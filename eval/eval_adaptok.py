import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import torch
import pandas as pd
from eval.rfvd_evaluator import UCFrFVDEvaluator
from models.adaptok import AdapTok

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--dataset_csv', type=str, default='ucf101_train.csv')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--version', type=str, default='sd')
    parser.add_argument('--amp_dtype', type=str, default='float16')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no_fvd', action='store_true')
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--token_subsample', type=int, default=None)
    parser.add_argument('--repeat_to_16', action='store_true')
    parser.add_argument('--model_type', type=str, default='adaptok')
    parser.add_argument('--score_path', type=str, default=None, help="path to gt_scores")
    parser.add_argument('--score_type', type=str, default="mse")
    parser.add_argument('--pre_compute', action='store_true')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--frame_rate', type=str, default='fixed')
    parser.add_argument('--end_frame', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='output_metrics.csv')
    
    args, external_update_params = parser.parse_known_args()

    return args, external_update_params

model_dict = {
    "adaptok": AdapTok
}

def main(args, external_update_params):
    assert args.tokenizer is not None

    if os.path.exists(args.tokenizer) and os.path.isfile(args.tokenizer):
        # Load tokenizer from local checkpoint
        model = model_dict[args.model_type].from_checkpoint(args.tokenizer, external_update_params=external_update_params)
    else:
        # Load tokenizer from HuggingFace Hub
        model = model_dict[args.model_type].from_pretrained(args.tokenizer)

    if args.det and hasattr(model, 'set_vq_eval_deterministic'):
        model.set_vq_eval_deterministic(deterministic=True)
        print('Using deterministic VQ for evaluation')

    if args.use_amp:
        if args.amp_dtype == 'float16':
            amp_dtype = torch.float16
        elif args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        else:
            raise ValueError(f'Unknown AMP dtype: {args.amp_dtype}')
    else:
        amp_dtype = torch.float32
    
    evaluator = UCFrFVDEvaluator(
        model=model,
        dataset_csv=args.dataset_csv,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        compile=args.compile,
        frame_num=args.num_frames,
        batch_size=args.batch_size,
        num_workers=4,
        token_subsample=args.token_subsample,
        repeat_to_16=args.repeat_to_16,
        score_path=args.score_path,
        score_type=args.score_type,
        frame_rate=args.frame_rate,
    )
    mse, psnr_val, fvd, lpips_val, avg_toks = evaluator.evaluate(no_fvd=args.no_fvd, start=args.start_frame, end=args.end_frame)
    print(f'mse={mse}\npsnr_val={psnr_val}\nfvd={fvd}\nlpips_val={lpips_val}, avg_toks={avg_toks}')

    data = [{
        'mse': mse,
        'psnr_val': psnr_val.item(),
        'fvd': fvd.item(),
        'lpips_val': lpips_val.item(),
        'avg_toks': avg_toks
    }]

    if data:
        df = pd.DataFrame(data)
        df.to_csv(args.save_path, index=False)
        print(f"Metrics saved to {args.save_path}")


if __name__ == '__main__':
    args, external_update_params = get_args()
    main(args, external_update_params)