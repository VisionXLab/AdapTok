import os
from copy import deepcopy

import einops
import imageio
import torch
import torch.distributed as dist
import wandb
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils
from models import get_model_cls
from models.embed import LabelEmbedder
from models.norm import RMSNorm
from trainers import register

from .base_trainer import BaseTrainer
from models.mask_generator import (left_masking_by_group_adap, rearrange_drop_mask,
                                   insert_special_token_before_rearrange)


optimizer_dict = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW
}

def save_video(video, path):
    imageio.mimwrite(path, video.numpy(), fps=25)


@register('adaptok_ar_fp_trainer')
class AdapTokARFramePredictionTrainer(BaseTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        self.num_samples = cfg['ar']['num_samples']
        self.num_save_wandb = cfg['ar']['num_save_wandb']
        self.sample_batch_size = cfg['ar']['sample_batch_size']
        self.cfg_scale = cfg['ar']['cfg_scale']
        self.cfg_interval = cfg['ar']['cfg_interval']
        self.sampling_temperture = cfg['ar']['temperature']
        self.sampling_top_k = cfg['ar']['top_k']
        self.sampling_top_p = cfg['ar']['top_p']
        self.num_frames = cfg['ar']['num_frames']
        self.num_cond_frames = cfg['ar']['num_cond_frames']
        self.log(f'Using {self.num_cond_frames} frames as condition')
        cfg['vae']['checkpoint'] = cfg['vae']['checkpoint'].strip("'").strip('"')

        self.offline_mode = cfg['vae'].get('offline_mode', False)
        self.mode = cfg.model.args.get("mode", "train_default")
        self.fixed_per_block = cfg['vae'].get('fixed_per_block', -1)
        if os.path.exists(cfg['vae']['checkpoint']):
            # load model from local checkpoint
            self.vae = (
                get_model_cls(cfg['vae']['name'])
                .from_checkpoint(cfg['vae']['checkpoint'], version=cfg['vae']['version'])
                .to(self.device)
            )
        else:
            # load model from huggingface hub
            self.vae = (
                get_model_cls(cfg['vae']['name'])
                .from_pretrained(cfg['vae']['checkpoint'])
                .to(self.device)
            )

        self.vae.eval()
        vae_eval_deterministic = cfg['vae'].get('eval_deterministic', False)
        if vae_eval_deterministic:
            self.vae.set_vq_eval_deterministic(True)
            self.log('Set VQ eval mode to deterministic. Only effective if stochastic VQ is used')

        self.log(f'Loaded VAE from {cfg["vae"]["checkpoint"]}')
        self.vae_force_fp32 = cfg['vae_force_fp32'] if 'vae_force_fp32' in cfg else False

        self.seq_length = self.vae.bottleneck_token_num
        if self.cfg['model']['args']['start_ar_block_idx'] > 0:
            start_blk = self.cfg['model']['args']['start_ar_block_idx']
            self.cfg['model']['args']['tot_blocks'] -= start_blk
            num_groups = self.vae.video_token_num // self.vae.x_embedder.num_spatial_patches
            toks_per_group = self.vae.bottleneck_token_num // num_groups
            self.seq_length -= toks_per_group * start_blk

        self.cfg['model']['args']['max_seq_len'] = self.seq_length # this only counts the generated tokens, not the condition tokens
        self.cfg['model']['args']['vocab_size'] = self.vae.codebook_size # add 1 for the sep token (separates the given frames and the predicted frames)
        self.cfg['model']['args']['frame_prediction'] = True
        if self.cfg['model']['args'].get("custom_cls_token_num", False):
            self.cfg['model']['args']['cls_token_num'] += 1
        else:
            self.cfg['model']['args']['cls_token_num'] = self.seq_length + 1 # add 1 for the sep token
        
        self.log(f'Using sequence length: {self.seq_length}')
        self.log(f'Using vocab size: {self.vae.codebook_size}')


    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        exp_name = f"{base_exp_name}/FP_"
        if len(cfg.vae.checkpoint) < 8:
            exp_name += cfg.vae.checkpoint + "_"

        if float(cfg.optimizer.args.lr) != 0.0001:
            exp_name += f"lr{cfg.optimizer.args.lr}_"

        if 'weight_decay' in cfg.optimizer.args:
            if cfg.optimizer.args.weight_decay != 0.0:
                exp_name += f"wd{cfg.optimizer.args.weight_decay}_"

        exp_name += f'{cfg.model.name}_'
        exp_name += f'_{args.tag}'
        return exp_name


    def make_model(self, model_spec=None, load_sd=False):
        super().make_model(model_spec, load_sd)
        vae_size_str = utils.compute_num_params(self.vae)
        self.log(f'vae size: {vae_size_str}')
        if self.enable_wandb:
            wandb.run.summary['vae_size'] = vae_size_str


    def make_datasets(self):
        super().make_datasets()
        def get_vislist(dataset, n_vis=128):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return Subset(dataset, ids.tolist())

        if hasattr(self, 'test_loader_dict'):
            vislist_test = []
            for k, test_loader in self.test_loader_dict.items():
                test_loader: DataLoader
                vislist_test.append(get_vislist(test_loader.dataset, n_vis=self.num_samples))
                num_workers = test_loader.num_workers
            self.vislist_test = ConcatDataset(vislist_test)
            sampler = DistributedSampler(self.vislist_test, shuffle=False) if self.distributed else None
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
            self.vis_loader = DataLoader(
                self.vislist_test, 
                batch_size=self.sample_batch_size // world_size, 
                drop_last=False,
                sampler=sampler, 
                num_workers=num_workers, 
                persistent_workers = True if num_workers > 0 else False,
                pin_memory=True
            )
            if self.distributed:
                sampler.set_epoch(0)


    def configure_optimizers(self, config, load_sd=False):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (
            torch.nn.LayerNorm, 
            torch.nn.Embedding,
            LabelEmbedder,
            RMSNorm
        )
        for mn, m in self.orig_model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root AR module as not decayed
        cases = ['pos_emb', 'abs_pe']
        for case in cases:
            if hasattr(self.orig_model, case) and isinstance(getattr(self.orig_model, case), torch.nn.Parameter):
                no_decay.add(case)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.orig_model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        config_no_decay = deepcopy(config)
        if 'weight_decay' in config_no_decay['args']:
            del config_no_decay['args']['weight_decay']

        optimizer_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': config['args']['weight_decay']},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_dict[config['name']](
            optimizer_groups,
            **config_no_decay['args']
        )

        if load_sd:
            optimizer.load_state_dict(config['sd'])

        self.optimizer = optimizer


    def forward_ar_model(self, z, c, mask=None, valid=None):
        input_tokens = z[:, :-1]
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            logits, loss = self.model_ddp(cond_idx=c, idx=input_tokens, targets=z, mask=mask, valid=valid)
        loss = loss.mean()
        return logits, loss


    def _iter_step(self, data, is_train):
        if self.mode == 'train_adap_ar_fp':
            if self.offline_mode:
                # load pre-annotated labels
                c = data['conditions'].to(dtype=torch.long, device=self.device, non_blocking=True)
                c = c[:, :self.model.cls_token_num-1]
                z = data['bottleneck_rep'].to(c, non_blocking=True)
                latent_nums = data['latent_nums'].to(c, non_blocking=True)
                if self.fixed_per_block > 0:
                    latent_nums[:,:] = self.fixed_per_block
            else:
                raise NotImplementedError
            
            sep_token = torch.full((c.shape[0], 1), self.model.cond_sep_id, device=c.device, dtype=c.dtype)
            c = torch.cat([c, sep_token], dim=1)

            toks_per_group = self.vae.bottleneck_token_num // latent_nums.shape[-1]

            if self.model.start_ar_block_idx > 0:
                start_blk = self.model.start_ar_block_idx
                z = z[:, start_blk * toks_per_group:].contiguous()
                latent_nums = latent_nums[:, start_blk:].contiguous()


            z = insert_special_token_before_rearrange(z, latent_nums, toks_per_group,
                                                      special_token=self.model.block_sep_id)
            
            latent_mask = left_masking_by_group_adap(
                num_latents=latent_nums + 1,
                toks_per_group=toks_per_group + 1,
                device=z.device
            ).bool()

            z = rearrange_drop_mask(z, latent_mask)
            latent_mask = rearrange_drop_mask(latent_mask, latent_mask)

            max_len = latent_nums.sum(dim=-1).max()
            z = z[:, :max_len].contiguous()
            latent_mask = latent_mask[:, :max_len].contiguous()

            logits, loss = self.forward_ar_model(z, c, valid=latent_mask)
            topk_accuracies = utils.calculate_topk_accuracy(
                logits.view(-1, logits.size(-1))[latent_mask.view(-1)],
                z.view(-1)[latent_mask.view(-1)], topk=(1, 5)
            )
        else:
            x = data.pop('gt').to(self.device, non_blocking=True)
            x_cond = utils.repeat_to_m_frames(x[:, :, :self.num_cond_frames], m=self.num_frames)

            # first stage model
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=not self.vae_force_fp32 and self.use_amp):
                with torch.no_grad():
                    x_x_cond = torch.cat([x, x_cond], dim=0)
                    z_c = self.vae.encode(x_x_cond)['bottleneck_rep'] # get the discrete token representation
                    z, c = torch.chunk(z_c, 2, dim=0) # (b, n)
                    assert z.shape == c.shape and z.ndim == 2
                    sep_token = torch.full((c.shape[0], 1), self.model.cond_sep_id, device=c.device, dtype=c.dtype)
                    c = torch.cat([c, sep_token], dim=1)

            # AR modeling
            logits, loss = self.forward_ar_model(z, c)
            topk_accuracies = utils.calculate_topk_accuracy(logits, z, topk=(1, 5))

        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for ema_decay, ema_model in self.ema_model_dict.items():
                self.update_ema(ema_model, decay=ema_decay)

        return_dict = {'loss': loss.item(), **topk_accuracies}
        return return_dict

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    @torch.inference_mode()
    def visualize_epoch(self, logging=True, use_ema=False, force_fp32=False):
        if use_ema:
            model = self.ema_model_dict[self.ema_decay_list[0]]
            self.log(f'Using EMA model {self.ema_decay_list[0]} for visualization')
        else:
            model = self.model
            self.log('Using current model for visualization')
        model = deepcopy(model)
        if not force_fp32:
            model = model.to(device=self.device, dtype=torch.bfloat16)
        model.eval()

        fp16_enabled = (not force_fp32) and self.use_amp
        self.log(f'Visualizing with fp16: {fp16_enabled}')

        out_dir = os.path.join(self.cfg['env']['save_dir'], 'visualize')
        vid_dir = os.path.join(out_dir, 'vid')
        if self.is_master:
            os.makedirs(vid_dir, exist_ok=True)

        sample_i3d_feats = None
        orig_i3d_feats = None

        pbar = self.vis_loader
        pbar = tqdm(pbar) if self.is_master else pbar
        block_sep_id = self.model.block_sep_id if hasattr(self.model, "block_sep_id") else None

        vis_res = []
        for i, data in enumerate(pbar):
            if self.mode == 'train_adap_ar_fp' and self.offline_mode:
                # load pre-annotated labels
                c = data['conditions'].to(dtype=torch.long, device=self.device, non_blocking=True)
                c = c[:, :self.model.cls_token_num-1]
                sep_token = torch.full((c.shape[0], 1), self.model.cond_sep_id, device=c.device, dtype=c.dtype)
                c = torch.cat([c, sep_token], dim=1)
                x = None    # in offline mode, we don't calculate fvd for this mini-batch
            else:
                assert not self.offline_mode, "not implemented yet"
                x = data['gt'].to(self.device, non_blocking=True)
                x_cond = utils.repeat_to_m_frames(x[:, :, :self.num_cond_frames], m=self.num_frames)

                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    with torch.no_grad():
                        c = self.vae.encode(x_cond)['bottleneck_rep'] # get the discrete token representation for conditioning
                        sep_token = torch.full((c.shape[0], 1), self.model.cond_sep_id, device=c.device, dtype=c.dtype)
                        c = torch.cat([c, sep_token], dim=1)
  
            sampled_seqs, latent_nums = model.sample(
                c=c,
                cfg_scale=self.cfg_scale, cfg_interval=self.cfg_interval,
                temperature=self.sampling_temperture, 
                top_k=self.sampling_top_k,
                top_p=self.sampling_top_p
            )

            if self.model.start_ar_block_idx > 0:
                start_blk = self.model.start_ar_block_idx
                toks_per_group = self.vae.bottleneck_token_num // (latent_nums.shape[-1] + start_blk)
                cond_latent_nums = torch.full((latent_nums.shape[0], start_blk), toks_per_group, device=self.device)
                cond_seqs = insert_special_token_before_rearrange(c[:, :toks_per_group *start_blk],
                                                                  cond_latent_nums, toks_per_group,
                                                                  special_token=self.model.block_sep_id)
            

                sampled_seqs = torch.cat([cond_seqs, sampled_seqs], dim=-1)
                latent_nums = torch.cat([cond_latent_nums, latent_nums], dim=-1)

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                sampled_batch: torch.FloatTensor = self.vae.decode_from_bottleneck(sampled_seqs, latent_nums, block_sep_id=block_sep_id) # (b, c, t, h, w)

            sampled_batch = sampled_batch.clamp(0., 1.)
            sampled_batch = sampled_batch.to(torch.float32)

            sampled_batch_new = sampled_batch
            orig_video_new = x

            if sampled_batch_new.shape[2] >= 10 and x is not None: # FVD calc requires at least 10 frames
                sample_i3d_feats = self.fvd_calculator.get_feature_stats_for_batch(sampled_batch_new, sample_i3d_feats)
                orig_i3d_feats = self.fvd_calculator.get_feature_stats_for_batch(orig_video_new, orig_i3d_feats)

            if len(vis_res) < self.num_save_wandb and self.is_master:
                vis_res.append(sampled_batch_new * 255.)

        del model

        if dist.is_initialized():
            dist.barrier()

        if sample_i3d_feats is not None:
            assert sample_i3d_feats.num_items == len(self.vislist_test), f"sample_i3d_feats.num_items={sample_i3d_feats.num_items}, len(self.vislist_test)={len(self.vislist_test)}"

        fvd, fid = None, None

        if self.is_master:
            vis_res = torch.cat(vis_res, dim=0)[:self.num_save_wandb]
            vis_res = einops.rearrange(vis_res, 'b c t h w -> b t c h w').cpu()
            vis_res = vis_res.type(torch.uint8)

            assert self.num_save_wandb % 8 == 0, "num_save_wandb must be divisible by 8"
            col = 8
            row = self.num_save_wandb // col

            # concat all samples into a video, 4 rows, 8 columns
            if vis_res.shape[0] < row * col:
                col = 4
                row = vis_res.shape[0] // col

            vis_res = einops.rearrange(vis_res, '(b row col) t c h w -> b t c (row h) (col w)', row=row, col=col)

            if self.enable_wandb:
                if vis_res.shape[1] >= 2: # videos
                    wandb.log({'samples': [wandb.Video(v, fps=4, format="mp4") for v in vis_res], "epoch": self.epoch, "iter": self.global_step}, step=self.global_step)
                else: # images
                    vis_res = vis_res.squeeze(1)
                    wandb.log({'samples': [wandb.Image(v) for v in vis_res], "epoch": self.epoch, "iter": self.global_step}, step=self.global_step)

            if sample_i3d_feats is not None:
                fvd = self.fvd_calculator.calculate_fvd(
                    sample_i3d_feats, 
                    orig_i3d_feats
                )

                if isinstance(fvd, torch.Tensor):
                    fvd = fvd.item()

                if logging:
                    logtext = f' sample_fvd={fvd:.4f}'
                    self.log_temp_scalar(f'test/fvd', fvd, self.global_step)
                    self.log_buffer.append(logtext)

                if hasattr(self, 'current_fvd'):
                    self.current_fvd = fvd

        if dist.is_initialized():
            dist.barrier()

        return {
            'fvd': fvd,
            'fid': fid
        }
