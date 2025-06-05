ar_ckpt=path_to_adaptok_ar.pth
tokenizer_ckpt=path_to_adaptok.pth
output_dir=path_to_save

mkdir -p $output_dir

python3 sample.py \
    --ar_model $ar_ckpt \
    --tokenizer $tokenizer_ckpt \
    --output_dir $output_dir \
    --num_samples 10000 \
    --sample_batch_size 64 \
    --cfg_scale 1.0 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42 \
    --replace \
    --stats_only \
    --model_type adaptok