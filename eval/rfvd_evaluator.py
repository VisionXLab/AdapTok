import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import einops
import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from utils import FVDCalculator


class UCFrFVDEvaluator:
    def __init__(
        self,
        model,
        dataset_csv,
        root_path='data/metadata',
        frame_num=16,
        crop_size=128,
        batch_size=4,
        num_workers=4,
        use_amp=True,
        amp_dtype=torch.float16,
        compile=False,
        token_subsample=None,
        repeat_to_16=False,
        score_path=None,
        score_type="mse",
        frame_rate='fixed'
    ):
        self.model = model.cuda().eval()
        if hasattr(self.model, 'x_embedder'):
            self.model.x_embedder.strict_vid_size = False
        if compile:
            self.model = torch.compile(self.model, dynamic=(os.environ.get('USE_DYNAMIC', "")=="True"))
        self.dataset_csv = dataset_csv
        self.root_path = root_path
        self.frame_num = frame_num
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.token_subsample = token_subsample
        self.repeat_to_16 = repeat_to_16

        self.psnr = lambda x, y: (-10 * torch.log10(F.mse_loss(x, y)))
        self.psnr_given_mse = lambda m: (-10 * torch.log10(m)).mean()
        self.lantent_toks = 0
        self.totol_num = 0

        self.perceptual_loss = lpips.LPIPS(net='vgg').cuda().eval()
        self.fvdc = FVDCalculator()

        self.dataset = datasets.make(
            {
                'name': 'video_dataset',
                'args': {
                    'root_path': self.root_path,
                    'frame_num': self.frame_num,
                    'cls_vid_num': '-1_-1',
                    'crop_size': self.crop_size,
                    'csv_file': self.dataset_csv,
                    'frame_rate': frame_rate,
                    'use_all_frames': False,
                    'score_path': score_path,
                    'with_scores': True,
                    'score_type': score_type,
                },
            }
        )

        if hasattr(self.dataset, "score_mean"):
            self.model.score_mean = self.dataset.score_mean
            self.model.score_std = self.dataset.score_std

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def repeat_to_16_frames(self, video_tensor):
        # video_tensor: (b, c, t, h, w)
        b, c, t, h, w = video_tensor.shape
        if t >= 16:
            return video_tensor
        else:
            repeated_tensor = torch.cat([video_tensor, video_tensor[:, :, -1:].repeat(1, 1, 16 - t, 1, 1)], dim=2)
            return repeated_tensor


    def evaluate(self, no_fvd=False, start=0, end=16):
        mse_l = []
        lpips_l = []
        fake_stats = None
        running_real_stats = None
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(self.loader)):
                vb = batch['gt'].cuda()
                n_frames = vb.size(2)
                if self.repeat_to_16:
                    vb = self.repeat_to_16_frames(vb)

                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    if hasattr(self.model, 'encode_eval'):
                        er = self.model.encode_eval(vb)
                    else:
                        er = self.model.encode(vb)
                    if 'encoded' in er:
                        de_input = er['encoded']
                    else:
                        de_input = er['bottleneck_rep']

                    if self.token_subsample is not None:
                        rvb = self.model.decode(de_input, token_subsample=self.token_subsample)
                    else:
                        num_x_tokens = er['num_x_tokens'] if 'num_x_tokens' in er else None
                        latent_mask = er['latent_mask'] if 'latent_mask' in er else None
                        if latent_mask is not None:
                            self.lantent_toks += latent_mask.sum().item()
                            self.totol_num += latent_mask.shape[0]
                        if hasattr(self.model, 'decode_eval'):
                            rvb = self.model.decode_eval(de_input, num_x_tokens=num_x_tokens, latent_mask=latent_mask)
                        else:
                            rvb = self.model.decode(de_input)

                if isinstance(rvb, dict):
                    rvb = rvb['pred_frames']

                # collect at most 16 frames
                # they have shape (b, c, t, h, w)
                rvb = rvb.float().clamp(0.0, 1.0)
                vb = vb[:, :, start:end]
                rvb = rvb[:, :, start:end]

                if self.repeat_to_16:
                    vb = vb[:, :, :n_frames] # back to original length
                    rvb = rvb[:, :, :n_frames] # back to original length

                mse_l.append(
                    F.mse_loss(vb, rvb, reduction='none').mean(dim=(1, 2, 3, 4))
                )
                vb_frames = einops.rearrange(vb, 'b c t h w -> (b t) c h w')
                rvb_frames = einops.rearrange(rvb, 'b c t h w -> (b t) c h w')

                if len(vb_frames) > 1024:
                    for ii in range(0, len(vb_frames), 1024):
                        lpips_l.append(self.perceptual_loss(vb_frames[ii:ii+1024], rvb_frames[ii:ii+1024], normalize=True))
                else:
                    lpips_l.append(self.perceptual_loss(vb_frames, rvb_frames, normalize=True))

                if vb.size(2) >= 12:
                    if len(vb) > 64:
                        for ii in range(0, len(vb), 128):
                            fake_stats = self.fvdc.get_feature_stats_for_batch(rvb[ii:ii+128], fake_stats)
                            running_real_stats = self.fvdc.get_feature_stats_for_batch(
                                vb[ii:ii+128], running_real_stats
                            )
                    else:
                        fake_stats = self.fvdc.get_feature_stats_for_batch(rvb, fake_stats)
                        running_real_stats = self.fvdc.get_feature_stats_for_batch(
                            vb, running_real_stats
                        )

        mse = torch.cat(mse_l) #(n,)
        assert mse.ndim == 1
        lpips_val = torch.cat(lpips_l).mean()
        psnr_val = self.psnr_given_mse(mse)
        avg_toks = self.lantent_toks / self.totol_num if self.totol_num != 0 else 0

        if no_fvd or fake_stats is None or running_real_stats is None:
            fvd = -1.0
        else:
            fvd = self.fvdc.calculate_fvd(fake_stats, running_real_stats)

        return mse.mean().item(), psnr_val, fvd, lpips_val, avg_toks

