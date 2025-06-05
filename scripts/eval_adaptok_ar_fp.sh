ar_ckpt=yeahhhh326/AdapTok-FP
tokenizer_ckpt=yeahhhh326/AdapTok
output_dir=path_to_save

python3 sample.py \
    --fp --num_cond_frames 5 \
    --ar_model $ar_ckpt \
    --tokenizer $tokenizer_ckpt \
    --output_dir $output_dir \
    --num_samples 50000 \
    --sample_batch_size 64 \
    --dtype bfloat16 \
    --dataset_csv k600_val.csv \
    --dataset_split_seed 42 \
    --replace \
    --stats_only \
    --model_type adaptok