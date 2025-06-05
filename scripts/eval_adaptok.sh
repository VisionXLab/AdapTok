tokenizer_path=path_to_scorer.pth

python eval/eval_adaptok.py \
    --score_type perceptual \
    --tokenizer $tokenizer_path \
    --use_amp \
    --det \
    --model_type adaptok \
    --num_frames 16 \
    --end_frame 16 \
    --batch_size 64 \
    --dataset_csv ucf101_train.csv \
    mode infer_with_online_scorer \
    token_select_mode ilp \
    token_select_num 256 \
    mask_generator.eval_mask_type left_masking_by_group_adap
