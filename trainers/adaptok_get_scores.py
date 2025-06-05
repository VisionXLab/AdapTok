from trainers import register
from .adaptok_trainer import AdapTokTrainer
from .base_trainer import map_location_fn

import os
import random

import utils
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import copy
import lpips
import einops
from pytorch_msssim import ssim

@register('adaptok_score_labeler')
class AdapTokScoreLabeler(AdapTokTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)

    def run(self):
        self.make_datasets(drop_last_train=False, drop_last_val=False, shuffle_train=False, shuffle_val=False)
        self.starting_epoch = 1
        self.global_step = 0
        assert self.init_checkpoint is not None and os.path.exists(self.init_checkpoint), "ckpt required!"

        # resume training from a checkpoint
        self.log(f'Loading from {self.init_checkpoint}')
        latest_ckt = torch.load(self.init_checkpoint, map_location=map_location_fn)
        model_spec = copy.deepcopy(self.cfg['model'])
        model_spec['sd'] = latest_ckt['model']['sd']
        self.make_model(model_spec, load_sd=True)

        if 'loss' in latest_ckt:
            self.make_loss(latest_ckt['loss'], load_sd=True)
        
        self.configure_optimizers(self.cfg['optimizer'], load_sd=False)
        self.configure_scalers(load_sd=False)

        self.starting_epoch = latest_ckt['epoch'] + 1
        self.scores = defaultdict(list)

        if 'rng_states_per_rank' in latest_ckt and self.rank in latest_ckt['rng_states_per_rank']:
            local_rng_state_dict = latest_ckt['rng_states_per_rank'][self.rank]
            torch.set_rng_state(local_rng_state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(local_rng_state_dict['torch_cuda_rng_state'])
            np.random.set_state(local_rng_state_dict['numpy_rng_state'])
            random.setstate(local_rng_state_dict['python_rng_state'])

        self.perceptual_loss = lpips.LPIPS(net='vgg').cuda().eval()

        torch.cuda.empty_cache()
        self.config()
        self.get_scores()

    def set_perceptual_eval(self):
        self.perceptual_loss.eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad_(False)


    def get_scores(self):
        if self.mode == "get_gt_scores":
            self.get_scores_from_tokenizer()
        elif self.mode == "get_infer_scores":
            self.get_scores_from_scorer()
        elif self.mode == "get_ar_annotations":
            self.get_ar_annotations_from_scorer()
        elif self.mode == "get_ar_fp_annotations":
            self.get_ar_fp_annotations_from_scorer()
        else:
            raise NotImplementedError

    def get_ar_annotations_from_scorer(self):
        SAVE_ITER = 500
        MERGE_ALL = True
        save_dir = os.path.join(self.cfg['env']['save_dir'], 'annotations')
        os.makedirs(save_dir, exist_ok=True)
        self.model_ddp.eval()
        local_latent_nums = []
        local_ids = []
        local_bottleneck_rep = []
        
        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc=f'get_priors', leave=True)
        mini_i = 0

        for i, inputs in enumerate(pbar):
            data = inputs['gt'].to(self.device)

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                if self.distributed:
                    model_output = self.model_ddp.module.encode_eval(data)
                else:
                    model_output = self.model_ddp.encode_eval(data)
                # model_output = self.model.encode_eval(data)
                
            local_latent_nums.extend(model_output['latent_num'].cpu().tolist())
            local_ids.extend([f"{s}_{e}_frames__{os.path.basename(i)}" for i,s,e in zip(inputs["path"], inputs["frame_start"], inputs["frame_end"])])
            local_bottleneck_rep.extend(model_output['bottleneck_rep'].cpu().tolist())

            if ((i + 1) % SAVE_ITER == 0) or (i + 1 == len(self.train_loader)):
                if self.distributed:
                    world_size = dist.get_world_size()
                    all_latent_nums = [None] * world_size
                    all_ids = [None] * world_size
                    all_bottleneck_rep = [None] * world_size

                    dist.all_gather_object(all_latent_nums, local_latent_nums)
                    dist.all_gather_object(all_ids, local_ids)
                    dist.all_gather_object(all_bottleneck_rep, local_bottleneck_rep)

                    if self.rank == 0:
                        merged_latent_nums = [latent_num for sublist in all_latent_nums for latent_num in sublist]
                        merged_latent_nums = torch.tensor(merged_latent_nums, dtype=torch.int16)
                        merged_ids = [id_ for sublist in all_ids for id_ in sublist]
                        merged_bottleneck_rep = [toks for sublist in all_bottleneck_rep for toks in sublist]
                        merged_bottleneck_rep = torch.tensor(merged_bottleneck_rep, dtype=torch.int16)

                        save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt') + f'_sub{mini_i}.pt')
                        save_dict = {'id': merged_ids, 'latent_nums': merged_latent_nums, 'bottleneck_rep': merged_bottleneck_rep}
                        torch.save(save_dict, save_path)
                        print(f"Saved score results to {save_path}")
                else:
                    merged_latent_nums = torch.tensor(local_latent_nums, dtype=torch.int16)
                    merged_ids = local_ids
                    merged_bottleneck_rep = torch.tensor(local_bottleneck_rep, dtype=torch.int16)
                    
                    save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt') + f'_sub{mini_i}.pt')
                    save_dict = {'id': merged_ids, 'latent_nums': merged_latent_nums, 'bottleneck_rep': merged_bottleneck_rep}
                    torch.save(save_dict, save_path)
                    print(f"Saved score results to {save_path}")

                # reset the local infos
                local_latent_nums = []
                local_ids = []
                local_bottleneck_rep = []
                mini_i += 1

        # merge all
        if MERGE_ALL:
            save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt'))
            mini_paths = glob(save_path + "_sub*.pt")
            if self.rank == 0:
                all_ids = []
                all_latent_nums = []
                all_bottleneck_rep = []
                for path in sorted(mini_paths):
                    sub_dict = torch.load(path)
                    all_ids.extend(sub_dict['id'])
                    all_latent_nums.append(sub_dict['latent_nums'])
                    all_bottleneck_rep.append(sub_dict['bottleneck_rep'])

                merged_latent_nums = torch.cat(all_latent_nums, dim=0)
                merged_bottleneck_rep = torch.cat(all_bottleneck_rep, dim=0)

                save_dict = {
                    'id': all_ids,
                    'latent_nums': merged_latent_nums,
                    'bottleneck_rep': merged_bottleneck_rep
                }
                torch.save(save_dict, save_path)
                print(f"Merged and saved final score results to {save_path}")
        
    
    def get_ar_fp_annotations_from_scorer(self):
        SAVE_ITER = 100
        MERGE_ALL = False
        save_dir = os.path.join(self.cfg['env']['save_dir'], 'annotations')
        os.makedirs(save_dir, exist_ok=True)
        self.model_ddp.eval()
        local_latent_nums = []
        local_ids = []
        local_bottleneck_rep = []
        local_conditions = []
        num_cond_frames = self.cfg.get('num_cond_frames', 5)
        
        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc=f'get_priors', leave=True)
        mini_i = 0

        for i, inputs in enumerate(pbar):
            data = inputs['gt'].to(self.device)
            x_cond = utils.repeat_to_m_frames(data[:, :, :num_cond_frames], m=data.shape[2])

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                if self.distributed:
                    model_output = self.model_ddp.module.encode_eval(data)
                    cond_output = self.model_ddp.module.encode(x_cond, with_latent_mask=False)
                else:
                    model_output = self.model_ddp.encode_eval(data)
                    cond_output = self.model_ddp.encode(x_cond, with_latent_mask=False)
                # model_output = self.model.encode_eval(data)
                
            local_latent_nums.extend(model_output['latent_num'].cpu().tolist())
            local_ids.extend([f"{s}_{e}_frames__{os.path.basename(i)}" for i,s,e in zip(inputs["path"], inputs["frame_start"], inputs["frame_end"])])
            local_bottleneck_rep.extend(model_output['bottleneck_rep'].cpu().tolist())
            local_conditions.extend(cond_output['bottleneck_rep'].cpu().tolist())

            if (i > 0 and i % SAVE_ITER == 0) or (i + 1 == len(self.train_loader)):
                if self.distributed:
                    world_size = dist.get_world_size()
                    all_latent_nums = [None] * world_size
                    all_ids = [None] * world_size
                    all_bottleneck_rep = [None] * world_size
                    all_conditions = [None] * world_size

                    dist.all_gather_object(all_latent_nums, local_latent_nums)
                    dist.all_gather_object(all_ids, local_ids)
                    dist.all_gather_object(all_bottleneck_rep, local_bottleneck_rep)
                    dist.all_gather_object(all_conditions, local_conditions)

                    if self.rank == 0:
                        merged_latent_nums = [latent_num for sublist in all_latent_nums for latent_num in sublist]
                        merged_latent_nums = torch.tensor(merged_latent_nums, dtype=torch.int16)
                        merged_ids = [id_ for sublist in all_ids for id_ in sublist]
                        merged_bottleneck_rep = [toks for sublist in all_bottleneck_rep for toks in sublist]
                        merged_bottleneck_rep = torch.tensor(merged_bottleneck_rep, dtype=torch.int16)
                        merged_conditions = [toks for sublist in all_conditions for toks in sublist]
                        merged_conditions = torch.tensor(merged_conditions, dtype=torch.int16)

                        save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt') + f'_sub{mini_i}.pt')
                        save_dict = {'id': merged_ids, 'latent_nums': merged_latent_nums,
                                     'bottleneck_rep': merged_bottleneck_rep, 'conditions': merged_conditions}
                        torch.save(save_dict, save_path)
                        print(f"Saved score results to {save_path}")
                else:
                    merged_latent_nums = torch.tensor(local_latent_nums, dtype=torch.int16)
                    merged_ids = local_ids
                    merged_bottleneck_rep = torch.tensor(local_bottleneck_rep, dtype=torch.int16)
                    merged_conditions = torch.tensor(local_conditions, dtype=torch.int16)
                    
                    save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt') + f'_sub{mini_i}.pt')
                    save_dict = {'id': merged_ids, 'latent_nums': merged_latent_nums,
                                 'bottleneck_rep': merged_bottleneck_rep, 'conditions': merged_conditions}
                    torch.save(save_dict, save_path)
                    print(f"Saved score results to {save_path}")

                # reset the local infos
                local_latent_nums = []
                local_ids = []
                local_bottleneck_rep = []
                local_conditions = []
                mini_i += 1

        # merge all
        if MERGE_ALL:
            save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt'))
            mini_paths = glob(save_path + "_sub*.pt")
            if self.rank == 0:
                all_ids = []
                all_latent_nums = []
                all_bottleneck_rep = []
                all_conditions = []
                for path in sorted(mini_paths):
                    sub_dict = torch.load(path)
                    all_ids.extend(sub_dict['id'])
                    all_latent_nums.append(sub_dict['latent_nums'])
                    all_bottleneck_rep.append(sub_dict['bottleneck_rep'])
                    all_conditions.append(sub_dict['conditions'])

                merged_latent_nums = torch.cat(all_latent_nums, dim=0)
                merged_bottleneck_rep = torch.cat(all_bottleneck_rep, dim=0)
                merged_conditions = torch.cat(all_conditions, dim=0)

                save_dict = {
                    'id': all_ids,
                    'latent_nums': merged_latent_nums,
                    'bottleneck_rep': merged_bottleneck_rep,
                    'conditions': merged_conditions
                }
                torch.save(save_dict, save_path)
                print(f"Merged and saved final score results to {save_path}")
    

    def get_scores_from_scorer(self):
        save_dir = os.path.join(self.cfg['env']['save_dir'], 'scores')
        os.makedirs(save_dir, exist_ok=True)
        self.model_ddp.eval()
        local_mses = []
        local_ids = []
        
        mean, std, score_type = self.train_loader.dataset.score_mean, self.train_loader.dataset.score_std, self.train_loader.dataset.score_type
        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc=f'get_scores', leave=True)

        for i, inputs in enumerate(pbar):
            data = inputs['gt'].to(self.device)

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                model_output = self.model_ddp(
                    data=data,
                    global_step=self.global_step,
                    max_steps=self.max_steps,
                )
                
            local_mses.extend(model_output['scores'].cpu().tolist())
            local_ids.extend([f"{s}_{e}_frames__{os.path.basename(i)}" for i,s,e in zip(inputs["path"], inputs["frame_start"], inputs["frame_end"])])

        if self.distributed:
            world_size = dist.get_world_size()
            all_preds = [None] * world_size
            all_ids = [None] * world_size

            dist.all_gather_object(all_preds, local_mses)
            dist.all_gather_object(all_ids, local_ids)

            if self.rank == 0:
                merged_preds = [mse for sublist in all_preds for mse in sublist]
                merged_preds = torch.tensor(merged_preds) * std + mean
                merged_ids = [id_ for sublist in all_ids for id_ in sublist]
                save_dict = {score_type: merged_preds, 'id': merged_ids}
                save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt'))
                torch.save(save_dict, save_path)
                print(f"Saved score results to {save_path}")
        else:
            merged_preds = local_mses
            merged_preds = torch.tensor(merged_preds) * std + mean
            merged_ids = local_ids
            save_dict = {score_type: merged_preds, 'id': merged_ids}
            save_path = os.path.join(save_dir, self.cfg.get('score_save_name', 'scores_predict.pt'))
            torch.save(save_dict, save_path)
            print(f"Saved score results to {save_path}")


    def get_scores_from_tokenizer(self):
        save_dir = os.path.join(self.cfg['env']['save_dir'], 'scores')
        os.makedirs(save_dir, exist_ok=True)
        tot_toks, tot_groups = self.model.mask_generator.total_toks, self.model.mask_generator.tot_groups
        for tok_num in range(self.cfg.tok_step, tot_toks * tot_groups + 1, self.cfg.tok_step):
            self.get_scores_by_fixed_toks(tok_num)

        if self.rank == 0:
            save_dict = defaultdict(list)
            for tok_num in range(self.cfg.tok_step, tot_toks * tot_groups + 1, self.cfg.tok_step):
                save_path = os.path.join(save_dir, f'scores_tok{tok_num}.pt')
                cur_dict = torch.load(save_path, weights_only=True)
                for k, v in cur_dict.items():
                    if k == 'id':
                        if save_dict['id']:
                            assert save_dict['id'] == v
                        else:
                            save_dict['id'] = v
                    else:
                        save_dict[k].append(torch.tensor(v).unsqueeze(1))

                # del save_path

            for k in save_dict.keys():
                if k != "id":
                    save_dict[k] = torch.cat(save_dict[k], dim=1)

            save_path = os.path.join(save_dir, f'scores.pt')
            torch.save(save_dict, save_path)
            print(f"Saved score results to {save_path}")


    def get_scores_by_fixed_toks(self, max_toks):
        print(f"Get scores with {max_toks} tokens each group.")
        self.model.mask_generator.eval_mask_type = "left_masking_fixed"
        self.model.mask_generator.max_toks = max_toks

        # k-th block's frames
        tot_toks, tot_groups = self.model.mask_generator.total_toks, self.model.mask_generator.tot_groups
        block_idx = (torch.arange(tot_toks, tot_groups * tot_toks + 1, tot_toks) >= max_toks).int().argmax().item()
        frame_step = self.train_loader.dataset.frame_num // tot_groups
        start_idx, end_idx = frame_step * block_idx, frame_step * block_idx + frame_step
        print(f"Start idx: {start_idx}; end idx: {end_idx}")

        self.model_ddp.eval()
        local_mses = []
        local_ids = []
        local_perceptuals = []
        local_psnrs = []
        local_ssims = []
        local_fake_logits = []

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc=f'get_scores_tok{max_toks}', leave=True)

        with torch.inference_mode():
            for i, inputs in enumerate(pbar):
                data = inputs['gt'].to(self.device) 
                B = data.shape[0]
                
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    model_output = self.model_ddp(
                        data=data,
                        global_step=self.global_step,
                        max_steps=self.max_steps,
                    )
                    assert isinstance(model_output, dict)
                    pred_frames = model_output['pred_frames']
                    logits_fake = self.loss.discriminator(pred_frames)

                # only calculate the related blocks
                pred_frames = pred_frames.float().clamp(0.0, 1.0)
                data = data[:,:, start_idx:end_idx]
                pred_frames = pred_frames[:,:, start_idx: end_idx]
                
                vb_frames = einops.rearrange(data, 'b c t h w -> (b t) c h w')
                rvb_frames = einops.rearrange(pred_frames, 'b c t h w -> (b t) c h w')
                perceptual = self.perceptual_loss(vb_frames, rvb_frames, normalize=True)
                perceptual = perceptual.view(B, -1).mean(dim=1)
                ssim_v = ssim(vb_frames, rvb_frames, size_average=False).view(B, -1).mean(dim=1)
            
                mses_per_frame = ((pred_frames - data)**2).mean(dim=(1, 3, 4))
                mses = mses_per_frame.mean(dim=-1)
                psnr =  (-10 * torch.log10(mses_per_frame)).mean(dim=-1)
                
                local_mses.extend(mses.cpu().tolist())
                local_perceptuals.extend(perceptual.cpu().tolist())
                local_psnrs.extend(psnr.cpu().tolist())
                local_ssims.extend(ssim_v.cpu().tolist())
                local_fake_logits.extend(logits_fake.cpu().tolist())
                local_ids.extend([os.path.basename(i) for i in inputs["path"]])

        if self.distributed:
            world_size = dist.get_world_size()
            all_mses = [None] * world_size
            all_perceptuals = [None] * world_size
            all_psnrs = [None] * world_size
            all_ssims = [None] * world_size
            all_fake_logits = [None] * world_size
            all_ids = [None] * world_size

            dist.all_gather_object(all_mses, local_mses)
            dist.all_gather_object(all_perceptuals, local_perceptuals)
            dist.all_gather_object(all_psnrs, local_psnrs)
            dist.all_gather_object(all_ssims, local_ssims)
            dist.all_gather_object(all_fake_logits, local_fake_logits)
            dist.all_gather_object(all_ids, local_ids)

            if self.rank == 0:
                merged_mses = [mse for sublist in all_mses for mse in sublist]
                merged_perceptuals = [p for sublist in all_perceptuals for p in sublist]
                merged_psnrs = [p for sublist in all_psnrs for p in sublist]
                merged_ssims = [p for sublist in all_ssims for p in sublist]
                merged_fake_logits = [p for sublist in all_fake_logits for p in sublist]
                merged_ids = [id_ for sublist in all_ids for id_ in sublist]
                save_path = os.path.join(self.cfg['env']['save_dir'], 'scores', f'scores_tok{max_toks}.pt')
                torch.save({'mse': torch.tensor(merged_mses),
                            'perceptual': torch.tensor(merged_perceptuals),
                            'psnr': torch.tensor(merged_psnrs),
                            'ssim': torch.tensor(merged_ssims),
                            'fake_logits': torch.tensor(merged_fake_logits),
                            'id': merged_ids}, save_path)
                print(f"Saved MSE results to {save_path}")
        else:
            merged_mses = local_mses
            merged_perceptuals = local_perceptuals
            merged_psnrs = local_psnrs
            merged_ssims = local_ssims
            merged_fake_logits = local_fake_logits
            merged_ids = local_ids
            save_path = os.path.join(self.cfg['env']['save_dir'], 'scores', f'scores_tok{max_toks}.pt')
            torch.save({'mse': torch.tensor(merged_mses),
                        'perceptual': torch.tensor(merged_perceptuals),
                        'psnr': torch.tensor(merged_psnrs),
                        'ssim': torch.tensor(merged_ssims),
                        'fake_logits': torch.tensor(merged_fake_logits),
                        'id': merged_ids}, save_path)
            print(f"Saved score results to {save_path}")


    def config(self):
        cfg = self.cfg
        max_epoch = cfg['max_epoch']
        self.n_steps_per_epoch = len(self.train_loader)
        self.max_steps = self.n_steps_per_epoch * max_epoch
        self.current_fvd = 99999.99
        self.current_fid = 99999.99
        self.epoch = self.starting_epoch
        self.model.set_vq_eval_deterministic(deterministic=True)
        print('Using deterministic VQ for evaluation')
        self.set_perceptual_eval()