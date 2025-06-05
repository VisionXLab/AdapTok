import itertools
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from easydict import EasyDict as edict
from collections import defaultdict

from safetensors.torch import load_file
from transformers import PretrainedConfig
from huggingface_hub import hf_hub_download, snapshot_download

import models
import utils
from models import register
from datasets.adap_priors import adap_each_block
from .mask_generator import (MaskSampler, generate_attention_mask, decoder_attn_mask_with_latent_mask, pad_latent_mask,
                             extract_padded_tokens, rearrange_drop_mask, solve_ilp_min, left_masking_by_group_adap,
                             remove_special_token_and_pad_blockwise)
from .embed import (PatchEmbed3D, VideoPatchEmbed,
                    get_1d_sincos_pos_embed_from_grid, get_3d_sincos_pos_embed)


def set_nested_value(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def convert_value(value):
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def get_orig_module(module):
    if hasattr(module, 'module'):
        module = module.module
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, temporal_patch_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, temporal_patch_size * patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        # x: [b, n, c]
        x = self.norm_final(x)
        x = self.linear(x)
        return x




@register('adaptok')
class AdapTok(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        bottleneck_token_num=1024,
        input_size=128,
        frame_num=16,
        temporal_patch_size=4,
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,

        use_encoder_patch_token_type_embed=False,
        use_encoder_latent_query_token_type_embed=False,
        use_decoder_latent_token_type_embed=False,
        use_decoder_patch_query_token_type_embed=False,

        encoder_query_gaussian_init=True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.frame_num = frame_num
        self.bottleneck_token_num = bottleneck_token_num
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.decoder_temporal_patch_size = decoder_temporal_patch_size
        self.decoder_patch_size = decoder_patch_size
        self.decoder_latent_len = bottleneck_token_num

        self.encoder_hidden_size = encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = decoder_hidden_size = int(decoder_hidden_size)
        self.encoder_num_heads = encoder_num_heads = int(encoder_num_heads)
        self.decoder_num_heads = decoder_num_heads = int(decoder_num_heads)

        self.latent_pe_scale_factor = latent_pe_scale_factor
        self.query_init_std = query_init_std

        if temporal_patch_size == 1:
            self.x_embedder = VideoPatchEmbed(input_size, patch_size, in_channels, encoder_hidden_size, bias=True, frame_num=frame_num)
        else:
            assert temporal_patch_size > 1
            self.x_embedder = PatchEmbed3D(input_size, frame_num, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True)
        self.token_h = token_h = self.token_w = token_w = int(self.x_embedder.num_spatial_patches ** 0.5)
        self.token_t = token_t = self.x_embedder.num_temporal_patches
        self.video_token_num = video_token_num = self.x_embedder.num_spatial_patches * token_t
        assert input_size % decoder_patch_size == 0, "input_size must be divisible by decoder_patch_size"
        self.decoder_token_t = decoder_token_t = frame_num // decoder_temporal_patch_size
        decoder_token_h = decoder_token_w = input_size // decoder_patch_size
        recon_num_patches_per_frame = decoder_token_h * decoder_token_w
        self.decoder_token_h = self.decoder_token_w = decoder_token_h
        self.recon_video_token_num = recon_video_token_num = recon_num_patches_per_frame * decoder_token_t


        # encoder patch PE
        self.learned_encoder_patch_pe = learned_encoder_patch_pe
        if self.learned_encoder_patch_pe:
            self.encoder_h_embed = nn.Parameter(torch.zeros(1, 1, token_h, 1, encoder_hidden_size), requires_grad=True)
            self.encode_w_embed = nn.Parameter(torch.zeros(1, 1, 1, token_w, encoder_hidden_size), requires_grad=True)
            self.encoder_t_embed = nn.Parameter(torch.zeros(1, token_t, 1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe_raw = lambda: (self.encoder_h_embed + self.encode_w_embed + self.encoder_t_embed).reshape(1, video_token_num, encoder_hidden_size)
        else:
            self.register_buffer('encoder_patch_pe', torch.zeros(1, video_token_num, encoder_hidden_size))
            self.get_encoder_patch_pe_raw = lambda: self.encoder_patch_pe
        self.use_encoder_patch_token_type_embed = use_encoder_patch_token_type_embed
        if self.use_encoder_patch_token_type_embed:
            self.encoder_patch_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe = lambda: self.get_encoder_patch_pe_raw() + self.encoder_patch_token_type_embed
        else:
            self.get_encoder_patch_pe = self.get_encoder_patch_pe_raw

        # encoder latent query embed
        self.learned_encoder_latent_query_embed = learned_encoder_latent_query_embed
        self.encoder_query_gaussian_init = encoder_query_gaussian_init
        if self.learned_encoder_latent_query_embed:
            self.encoder_latent_query_embed = nn.Parameter(torch.zeros(bottleneck_token_num, encoder_hidden_size), requires_grad=True)
        else:
            self.register_buffer('encoder_latent_query_embed', torch.zeros(bottleneck_token_num, encoder_hidden_size))
            assert not encoder_query_gaussian_init, "encoder_query_gaussian_init requires learned_encoder_latent_query_embed to be True"
        self.use_encoder_latent_query_token_type_embed = use_encoder_latent_query_token_type_embed
        if self.use_encoder_latent_query_token_type_embed:
            self.encoder_latent_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0) + self.encoder_latent_query_token_type_embed
        else:
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0)

        # decoder latent PE
        self.learned_decoder_latent_pe = learned_decoder_latent_pe
        if self.learned_decoder_latent_pe:
            self.decoder_latent_pe = nn.Parameter(torch.zeros(1, self.decoder_latent_len, decoder_hidden_size), requires_grad=True)
        else: 
            self.register_buffer('decoder_latent_pe', torch.zeros(1, self.decoder_latent_len, decoder_hidden_size))
        self.use_decoder_latent_token_type_embed = use_decoder_latent_token_type_embed
        if self.use_decoder_latent_token_type_embed:
            self.decoder_latent_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe + self.decoder_latent_token_type_embed
        else: 
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe

        # decoder patch query embed
        self.learned_decoder_patch_query_embed = learned_decoder_patch_query_embed
        if self.learned_decoder_patch_query_embed:
            self.decoder_h_embed = nn.Parameter(torch.zeros(1, 1, decoder_token_h, 1, decoder_hidden_size), requires_grad=True)
            self.decoder_w_embed = nn.Parameter(torch.zeros(1, 1, 1, decoder_token_w, decoder_hidden_size), requires_grad=True)
            self.decoder_t_embed = nn.Parameter(torch.zeros(1, decoder_token_t, 1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed_raw = lambda: (self.decoder_h_embed + self.decoder_w_embed + self.decoder_t_embed).reshape(1, recon_video_token_num, decoder_hidden_size)
        else:
            self.register_buffer('decoder_patch_query_embed', torch.zeros(1, recon_video_token_num, decoder_hidden_size))
            self.get_decoder_patch_query_embed_raw = lambda: self.decoder_patch_query_embed
        self.use_decoder_patch_query_token_type_embed = use_decoder_patch_query_token_type_embed
        if self.use_decoder_patch_query_token_type_embed:
            self.decoder_patch_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed = lambda: self.get_decoder_patch_query_embed_raw() + self.decoder_patch_query_token_type_embed
        else: 
            self.get_decoder_patch_query_embed = self.get_decoder_patch_query_embed_raw


        # Build encoder, decoder, and bottleneck
        if encoder_name is None or encoder_name.lower() in ['none', 'no', 'null', '']:
            encoder_name = transformer_name
        if decoder_name is None or decoder_name.lower() in ['none', 'no', 'null', '']:
            decoder_name = transformer_name

        encoder_args = {
            'name': encoder_name,
            'args': {
                'dim': encoder_hidden_size,
                'depth': encoder_depth,
                'n_head': encoder_num_heads,
                'head_dim': encoder_hidden_size // encoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        self.encoder = models.make(encoder_args)
        self.decoder = models.make(decoder_args)

        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
        bottleneck_args = {'token_nums': self.bottleneck_token_num, 'input_dim': encoder_hidden_size, 'output_dim': decoder_hidden_size}
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']
        self.final_layer = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)


        # Build prior model
        prior_model = edict(prior_model)
        if prior_model.get('name', '').lower() in ['none', 'no', 'null', '']:
            self.prior_model = None
        else:
            prior_model_additional_args = {'n_ind': self.bottleneck_dim,
                                           'n_classes': self.codebook_size,
                                           'max_seq_len':self.bottleneck_token_num
                                           }
            if prior_model.get('no_dropout', False):
                prior_model_additional_args['embd_pdrop'] = 0.0
                prior_model_additional_args['resid_pdrop'] = 0.0
                prior_model_additional_args['attn_pdrop'] = 0.0
                print(f"Warning: prior_loss is using no dropout")


            self.prior_model = models.make(prior_model, args=prior_model_additional_args)
            self.prior_n_rounds = prior_model.n_rounds
            self.prior_no_grad_before_last_round = prior_model.no_grad_before_last_round
            self.prior_avg_loss_over_rounds = prior_model.avg_loss_over_rounds
            self.use_mix_ss = prior_model.use_mix_ss
            self.mix_ss_max_ratio = prior_model.mix_ss_max_ratio
            self.mix_ss_peak_steps_ratio = prior_model.mix_ss_peak_steps_ratio
            self.prior_latent_ce_temperature = prior_model.latent_ce_temperature
        
        self.initialize_weights()
        
        # add encoder attention mask
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        encoder_attn_type = kwargs.get("encoder_attn_type", None)
        if encoder_attn_type is not None:
            num_groups = self.video_token_num // self.x_embedder.num_spatial_patches
            self.encoder_attn_mask = generate_attention_mask(num_img_tok_each=self.x_embedder.num_spatial_patches,
                                                             num_latent_each=self.bottleneck_token_num // num_groups,
                                                             num_groups=num_groups,
                                                             attn_type=encoder_attn_type,
                                                             latent_first=False).to(device)
        else:
            self.encoder_attn_mask = None
        
        scorer_attn_type = kwargs.get("scorer_attn_type", None)
        if scorer_attn_type is not None and scorer_attn_type not in ("none", "no", "null", ""):
            num_groups = self.video_token_num // self.x_embedder.num_spatial_patches
            self.scorer_attn_mask = generate_attention_mask(num_img_tok_each=self.bottleneck_token_num // num_groups,
                                                             num_latent_each=0,
                                                             num_groups=num_groups,
                                                             attn_type=scorer_attn_type,
                                                             latent_first=False).to(device)
        else:
            self.scorer_attn_mask = None

        # add decoder attention mask
        decoder_attn_type = kwargs.get("decoder_attn_type", None)
        if decoder_attn_type is not None:
            num_groups = self.recon_video_token_num // self.x_embedder.num_spatial_patches
            self.decoder_attn_mask = generate_attention_mask(num_img_tok_each=self.x_embedder.num_spatial_patches,
                                                             num_latent_each=self.bottleneck_token_num // num_groups,
                                                             num_groups=num_groups,
                                                             attn_type=decoder_attn_type,
                                                             latent_first=True).to(device)
        else:
            self.decoder_attn_mask = None


        print(f"encoder_attn_type={encoder_attn_type}")
        print(f"decoder_attn_type={decoder_attn_type}")
        print(f"scorer_attn_type={scorer_attn_type}")

        self.decode_mode = kwargs.get("decode_mode", "with_mask")

        self.mode = kwargs.get("mode", "train_adap")
        if kwargs.get('scorer_name', 'none').lower() not in ['none', 'no', 'null', '']:
            scorer_args = {
                'name': kwargs['scorer_name'],
                'args': {
                    'latent_len': self.bottleneck_token_num,
                    'step': kwargs['scorer_step'],
                    'dim': kwargs['scorer_hidden_size'],
                    'depth': kwargs['scorer_depth'],
                    'n_head': kwargs['scorer_num_heads'],
                    'head_dim': decoder_hidden_size // decoder_num_heads,
                },
            }
            self.scorer = models.make(scorer_args)
        
        self.mask_generator = MaskSampler(**kwargs["mask_generator"]) if "mask_generator" in kwargs else None
        self.score_thr = kwargs.get("score_thr", 10000)
        self.token_select_mode = kwargs.get("token_select_mode", "ilp")
        self.token_select_num = kwargs.get("token_select_num", self.mask_generator.max_toks)
        self.score_std = 1.0
        self.score_mean = 1.0
        self.eval_micro_bs = kwargs.get("eval_micro_bs", -1)
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        token_h, token_w = self.token_h, self.token_w

        # Initialize encoder patch PE
        if self.learned_encoder_patch_pe:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.token_t))
            self.encoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.encoder_h_embed))
            self.encode_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.encode_w_embed))
            self.encoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.encoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t)
            self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))
        if self.use_encoder_patch_token_type_embed:
            encoder_patch_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_patch_token_type_embed.data.copy_(encoder_patch_token_type_embed)

        # Initialize encoder latent query embed
        if self.learned_encoder_latent_query_embed:
            if self.encoder_query_gaussian_init:
                # from timm vision_transformer.py
                # https://github.com/huggingface/pytorch-image-models/blob/70ccf00c95a2d78a166cca24ef6adbca46f47c2a/timm/models/vision_transformer.py#L495
                query_embed = torch.randn(self.bottleneck_token_num, self.encoder_hidden_size) * self.query_init_std
            else:
                query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num))
                query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        else:
            query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num), self.latent_pe_scale_factor)
            query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        self.encoder_latent_query_embed.data.copy_(query_embed)
        if self.use_encoder_latent_query_token_type_embed:
            encoder_latent_query_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_latent_query_token_type_embed.data.copy_(encoder_latent_query_token_type_embed)

        # initialize decoder latent PE
        if self.learned_decoder_latent_pe:
            decoder_token_embed = torch.randn(1, self.decoder_latent_len, self.decoder_hidden_size) * .02
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        else:
            decoder_token_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_latent_len), self.latent_pe_scale_factor)
            decoder_token_embed = torch.from_numpy(decoder_token_embed).float().reshape(1, self.decoder_latent_len, self.decoder_hidden_size)
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        if self.use_decoder_latent_token_type_embed:
            decoder_latent_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_latent_token_type_embed.data.copy_(decoder_latent_token_type_embed)

        # initialize decoder patch query PE
        if self.learned_decoder_patch_query_embed:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_t))
            self.decoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.decoder_h_embed))
            self.decoder_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.decoder_w_embed))
            self.decoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.decoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.decoder_token_h, self.decoder_token_t)
            self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        if self.use_decoder_patch_query_token_type_embed:
            decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_last_layer(self):
        return self.final_layer.linear.weight

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decoder_parameters(self):
        decoder_params = itertools.chain(
            self.decoder.parameters(),
            self.final_layer.parameters()
        )

        if self.learned_decoder_patch_query_embed:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_h_embed, self.decoder_w_embed, self.decoder_t_embed]
            )

        if self.learned_decoder_latent_pe:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_latent_pe]
            )

        return decoder_params

    def decoder_requires_grad_(self, requires_grad):
        for param in self.decoder_parameters():
            param.requires_grad_(requires_grad)

    def others_parameters(self):
        decoder_params_set = set(self.decoder_parameters())
        return (p for p in self.parameters() if p not in decoder_params_set)

    def others_requires_grad_(self, requires_grad):
        for param in self.others_parameters():
            param.requires_grad_(requires_grad)

    def get_emb(self):
        emb = self.bottleneck.regularizer.get_emb()
        emb = emb.detach()
        return emb
    
    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True, version='sd', external_update_params=None):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        else:
            assert isinstance(
                ckpt, dict
            ), f"checkpoint must be a dict or a path to a checkpoint"

        kwargs = ckpt["model"]["args"]
            
        if external_update_params:
            assert len(external_update_params) % 2 == 0, "Invalid external_update_params format"
            for i in range(0, len(external_update_params), 2):
                key_path = external_update_params[i].split('.')
                value = convert_value(external_update_params[i + 1])
                set_nested_value(kwargs, key_path, value)

        model = cls(**kwargs)
        if load_state_dict:
            if version == 'sd':
                sd = ckpt["model"]["sd"]
            elif version.startswith('ema'):
                assert '_' in version, "ema version must be in the format 'ema_{alpha}'"
                alpha = float(version.split('_')[1])
                sd = ckpt["model"]['ema_sd'][alpha]
            else:
                raise ValueError(f"Unknown version: {version}")
            model.load_state_dict(sd, strict=True)
        return model
    
    @classmethod
    def from_pretrained(cls, ckpt, external_update_params=None):
        if not os.path.isdir(ckpt):
            config_path = snapshot_download(
                repo_id=ckpt,
                allow_patterns="config.json"
            )
            weight_path = hf_hub_download(
                repo_id=ckpt,
                filename="model.safetensors"
            )
        else:
            config_path = os.path.join(ckpt, "config.json")
            weight_path = os.path.join(ckpt, "model.safetensors")
            
        config = PretrainedConfig.from_pretrained(config_path)
        kwargs = config.to_dict()

        if external_update_params:
            assert len(external_update_params) % 2 == 0, "Invalid external_update_params format"
            for i in range(0, len(external_update_params), 2):
                key_path = external_update_params[i].split('.')
                value = convert_value(external_update_params[i + 1])
                set_nested_value(kwargs, key_path, value)

        model = cls(**kwargs)

        state_dict = load_file(weight_path)
        model.load_state_dict(state_dict)

        return model


    def forward(self, data, iter_mode="default", **kwargs):
        if iter_mode == "encode_only":
            return self.encode(data, **kwargs)
        elif iter_mode == "decode_only":
            return self.decode(data, **kwargs)
        elif iter_mode == "scorer_only":
            return self.decode_scorer(data, **kwargs)
        elif self.mode in ("train_scorer", "get_infer_scores"):
            return self._forward_scorer(data, **kwargs)
        else:
            return self._forward_adap(data, **kwargs)

    def _forward_adap(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        B = data.size(0)
        encode_output = self.encode(data)
        pred_frames = self.decode(encode_output['encoded'], encode_output['latent_mask']).contiguous() # [b, c, t, h, w]
        return_dict = {'pred_frames': pred_frames, **encode_output}

        if self.prior_model is not None:
            results = self.calculate_prior_loss_with_pred(encode_output, latent_mask=encode_output['latent_mask'], **kwargs)
            return_dict.update(results)

        return return_dict


    def _forward_scorer(self, data, **kwargs):
        with torch.no_grad():
            encode_output = self.encode(data)
        scorer_attn_mask = self.scorer_attn_mask.contiguous() if self.scorer_attn_mask is not None else None
        scores = self.scorer(encode_output['input'], encode_output['encoded'], scorer_attn_mask)
        
        return {'scores': scores, 'bottleneck_rep': encode_output['bottleneck_rep']}
    
    def decode_scorer(self, cont, quant, latent_mask=None):
        scorer_attn_mask = self.scorer_attn_mask.contiguous() if self.scorer_attn_mask is not None else None
        
        if latent_mask is not None:
            cont = cont.masked_fill(latent_mask.logical_not().unsqueeze(-1), float(0))   # add mask
            quant = quant.masked_fill(latent_mask.logical_not().unsqueeze(-1), float(0))   # add mask
            if scorer_attn_mask is None:
                scorer_attn_mask = torch.ones(self.bottleneck_token_num, self.bottleneck_token_num, device=self.device, dtype=torch.bool)
            scorer_attn_mask = decoder_attn_mask_with_latent_mask(scorer_attn_mask, latent_mask, bs=quant.shape[0], fill_diag=True)
            scorer_attn_mask = scorer_attn_mask.unsqueeze(1)
            
        scores = self.scorer(cont, quant, scorer_attn_mask)
        
        return {'scores': scores}

    def calculate_prior_loss_with_pred(self, encode_output, latent_mask=None, **kwargs):
        return_dict = {}
        B = encode_output['encoded'].size(0)
        ar_input = encode_output['regularized_z'] # (b, n, d=16) normalized


        if latent_mask is not None:
            k = latent_mask.shape[-1]
            pos_ids = torch.arange(k, device=self.device).unsqueeze(0).expand(B, k)
            
            new_latent_mask = rearrange_drop_mask(latent_mask, latent_mask)
            new_pos_ids = rearrange_drop_mask(pos_ids, latent_mask)
            ar_input = rearrange_drop_mask(ar_input, latent_mask)
            
            logits_all_rounds, _, _ = self.prior_ar_predict_n_rounds_ss(ar_input, latent_mask=new_latent_mask, pos_ids=new_pos_ids, **kwargs) # regularized_z_ss: (b, n, d=16)
            
            labels = rearrange_drop_mask(encode_output['bottleneck_rep'], latent_mask)
            labels = labels[:, 1:].contiguous() # (b, n - 1)
            labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1,).contiguous() # (n_rounds or 1, b, n - 1)
            new_latent_mask_all_rounds = new_latent_mask[:, :-1].unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous()

            loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size)[new_latent_mask_all_rounds.view(-1)],
                                             labels_all_rounds.view(-1)[new_latent_mask_all_rounds.view(-1)])
            topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0][new_latent_mask[:, :-1]],
                                                            labels[new_latent_mask[:, :-1]], topk=(1, 5), prepend='prior_')

        else:
            labels = encode_output['bottleneck_rep'][:, 1:].contiguous() # (b, n - 1)
            logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(ar_input, **kwargs) # regularized_z_ss: (b, n, d=16)
            labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1,).contiguous() # (n_rounds or 1, b, n - 1)    
            loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size), labels_all_rounds.view(-1))
            topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_')
        return_dict['loss_latent_ce'] = loss_latent_ce
        return_dict.update(topk_accuracies)

        return return_dict

    def logits_to_token_embedding_with_ss(self, logits, ar_input_staring_from_idx_1, mask=None, **kwargs):
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (kwargs['global_step'] / (kwargs['max_steps'] * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
            mask = mask.expand(-1, -1, self.bottleneck_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.codebook_size), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits.mul_(1 / self.prior_latent_ce_temperature)
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont

    def prior_ar_predict_n_rounds_ss(self, ar_input, latent_mask=None, pos_ids=None, **kwargs):
        b, n, _ = ar_input.size()
        ar_attn_mask = torch.triu(torch.ones(self.bottleneck_token_num, self.bottleneck_token_num, device=self.device), diagonal=1).logical_not()
        ar_attn_mask = decoder_attn_mask_with_latent_mask(ar_attn_mask, latent_mask, bs=b, fill_diag=True)[:, :-1, :-1]
        ar_attn_mask = ar_attn_mask.unsqueeze(1)
        
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        n_minus_1 = n - 1
        if self.use_mix_ss:
            global_step = kwargs['global_step']
            max_steps = kwargs['max_steps']
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32)
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32)

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, self.bottleneck_dim) # (b, n - 1, d=16)

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                prior_model_output = prior_model.ar_predict(next_ar_input.detach(), attn_mask=ar_attn_mask, pos_ids=pos_ids) # (b, n - 1, codebook_size)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                prior_model.requires_grad_(True)
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input, attn_mask=ar_attn_mask, pos_ids=pos_ids) # (b, n - 1, codebook_size) or (b, n, d=16)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(logits, ar_input[:, 1:], mask=mask_ss, **kwargs) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input


    def get_latent_mask(self, N):
        latent_mask = self.mask_generator(N, diff_batch=True, is_training=self.training) if self.mask_generator is not None else None
        return latent_mask
        
    def encode(self, x, with_latent_mask=True):
        latent_mask = self.mask_generator(len(x), diff_batch=True, is_training=self.training) if with_latent_mask and (self.mask_generator is not None) else None
        x = self.x_embedder(x) + self.get_encoder_patch_pe() # (b, n, d)
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) # (b, n, d)
        z = self.encoder(x, q_emb, self.encoder_attn_mask.contiguous())
        bottleneck_out = self.bottleneck(z, latent_mask)
        encoded = bottleneck_out.pop('output')
        return {'input': z, 'encoded': encoded, 'latent_mask': latent_mask, **bottleneck_out}

    def batch_encode(self, x, with_latent_mask=True):
        if self.eval_micro_bs > 0 and self.eval_micro_bs < len(x):
            outputs = defaultdict(list)
            for i in range(0, len(x), self.eval_micro_bs):
                mini_outputs = self.encode(x[i:i+self.eval_micro_bs], with_latent_mask)
                for k, v in mini_outputs.items():
                    outputs[k].append(v)
            
            for k, v in outputs.items():
                if isinstance(v[0], torch.Tensor):
                    outputs[k] = torch.cat(v, dim=0) if v[0].dim() > 1 else torch.tensor(v).mean()
                elif isinstance(v[0], float):
                    outputs[k] = sum(v) / len(v)
                elif v[0] is None:
                    outputs[k] = None
                else:
                    raise NotImplementedError

        else:
            outputs = self.encode(x, with_latent_mask)

        return outputs


    def batch_decode(self, z, latent_mask):
        if self.eval_micro_bs > 0:
            preds = []
            for i in range(0, len(z), self.eval_micro_bs):
                mini_mask = latent_mask[i:i+self.eval_micro_bs] if latent_mask is not None else None
                preds.append(self.decode(z[i:i+self.eval_micro_bs], mini_mask))
            preds = torch.cat(preds, dim=0)
        else:
            preds = self.decode(z, latent_mask)

        return preds

    def decode(self, z, latent_mask):
        if self.decode_mode == "with_mask":
            return self.decode_with_mask(z, latent_mask)
        elif self.decode_mode == "with_drop":
            return self.decode_with_drop(z, latent_mask)

    def decode_with_mask(self, z, latent_mask):
        """decoder with attention mask"""
        # z: (b, n, d)
        if latent_mask is not None:
            z = z.masked_fill(latent_mask.logical_not().unsqueeze(-1), float(0))   # add mask
            decoder_attn_mask = decoder_attn_mask_with_latent_mask(self.decoder_attn_mask, latent_mask, bs=z.shape[0], fill_diag=True)
        else:
            decoder_attn_mask = decoder_attn_mask_with_latent_mask(self.decoder_attn_mask, latent_mask, bs=z.shape[0], fill_diag=False)
        
        bs, seq_len, _ = decoder_attn_mask.shape
        decoder_attn_mask = decoder_attn_mask.unsqueeze(1)
        
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed

        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(bs, -1, -1)
        x = self.decoder(z, decoder_pos_embed, decoder_attn_mask.contiguous())
        x = self.final_layer(x)
        x = self.unpatchify(x)
        
        return x


    def encode_eval(self, x, with_latent_mask=True, avg_block_tok=None):
        if self.mode in ("infer_with_online_scorer", "get_ar_annotations", "get_ar_fp_annotations"):
            outputs = self.batch_encode(x, with_latent_mask=False)
            tot_toks, max_toks, num_groups = self.mask_generator.total_toks, self.mask_generator.max_toks, self.mask_generator.tot_groups
            step = self.scorer.step if hasattr(self, "scorer") else tot_toks // scores.shape[1]
            n_toks = tot_toks // step

            # adaptive per block assign
            latent_num = torch.zeros(len(x), num_groups, device=self.device, dtype=torch.int16)
            for block_idx in range(num_groups):
                latent_num[:, block_idx] = tot_toks
                latent_mask = self.mask_generator._left_masking_by_group_adap(num_latents=latent_num, device=self.device).bool()
                scores = self.decode_scorer(outputs['input'], outputs['encoded'], latent_mask)['scores']
                scores = scores[:, block_idx*n_toks:(block_idx+1)*n_toks]
                
                if self.token_select_mode == "ilp":
                    tok_select_num = adap_each_block[f"ilp_t{tot_toks*num_groups}_b{num_groups}"][self.token_select_num][block_idx] if avg_block_tok is None else avg_block_tok[block_idx]
                    block_num_latents = solve_ilp_min(scores, tok_select_num, step=step, device=self.device)
                else:
                    raise NotImplementedError
                latent_num[:, block_idx] = block_num_latents
            outputs['latent_num'] = latent_num
            outputs['latent_mask'] = self.mask_generator._left_masking_by_group_adap(num_latents=latent_num, device=self.device).bool()
        
        else:
            outputs = self.encode(x, with_latent_mask=with_latent_mask)

        return outputs

    def unpatchify(self, x):
        """
        x: (b, n, t_patch_size * s_patch_size**2 * c)
        videos: (b, c, t, h, w)
        """
        c = self.out_channels
        pt = self.temporal_patch_size
        p = self.patch_size
        h = w = self.token_h
        t = x.size(1) // (h * w)

        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x


    def decode_eval(self, z, num_x_tokens=None, latent_mask=None):
        outputs = self.batch_decode(z, latent_mask)
        return outputs

    def decode_with_drop(self, z, latent_mask=None):
        """decoder with drop mask.  z: (b, n, d)"""
        if latent_mask is not None:
            # z: (b, n, d)
            b = z.size(0)
            z = z.masked_fill(latent_mask.logical_not().unsqueeze(-1), float(0))   # add mask

            padded_latent_mask, _ = pad_latent_mask(latent_mask)

            decoder_attn_mask = decoder_attn_mask_with_latent_mask(self.decoder_attn_mask, latent_mask, b)
            padded_latent_mask_for_attn = torch.cat([
                padded_latent_mask,
                torch.ones(b, self.recon_video_token_num, dtype=padded_latent_mask.dtype, device=padded_latent_mask.device)
            ], dim=-1)
            decoder_attn_mask = extract_padded_tokens(decoder_attn_mask, padded_latent_mask_for_attn)
            decoder_attn_mask = extract_padded_tokens(decoder_attn_mask.transpose(1, 2), padded_latent_mask_for_attn).transpose(1, 2)
            bs, seq_len, _ = decoder_attn_mask.shape
            indices = torch.arange(seq_len, device=decoder_attn_mask.device)    # set the diagonal elements to True !!!
            decoder_attn_mask[:, indices, indices] = True
            decoder_attn_mask = decoder_attn_mask.unsqueeze(1)
            
            decoder_token_embed = self.get_decoder_latent_pe()
            z = z + decoder_token_embed
            z = extract_padded_tokens(z, padded_latent_mask)    # drop masked token

        else:
            decoder_attn_mask = decoder_attn_mask_with_latent_mask(self.decoder_attn_mask, latent_mask, bs=z.size(0))
            decoder_attn_mask = decoder_attn_mask.unsqueeze(1)
            decoder_token_embed = self.get_decoder_latent_pe()
            z = z + decoder_token_embed 

        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(z.size(0), -1, -1)
        x = self.decoder(z, decoder_pos_embed, decoder_attn_mask.contiguous())
        x = self.final_layer(x)
        x = self.unpatchify(x)
        
        return x


    def decode_from_bottleneck(self, bottleneck_rep, latent_nums=None, block_sep_id=None):
        # This method is only used when this module is used as a first-stage model

        if latent_nums is not None and block_sep_id is not None:
            rep = remove_special_token_and_pad_blockwise(bottleneck_rep, latent_nums, self.mask_generator.total_toks, special_token=block_sep_id)
            z = self.bottleneck.decode(rep) # (b, n, c)
            latent_mask = left_masking_by_group_adap(latent_nums, self.mask_generator.total_toks, device=self.device)
            output = self.decode_with_drop(z, latent_mask)

        else:
            z = self.bottleneck.decode(bottleneck_rep)
            output = self.decode_with_drop(z, latent_mask=None)

        return output