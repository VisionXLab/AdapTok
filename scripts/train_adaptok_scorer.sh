MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

export USE_DYNAMIC=True

project_name=adaptok_scorer
wandb_project_name=adaptok_scorer


init_checkpoint="path_to_tokenizer.pth"
score_type="perceptual"

echo "init_checkpoint=${init_checkpoint}"
echo "score_type=${score_type}"

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
            train.py --cfg cfgs/adaptok_scorer.yaml \
            --manualSeed 66667 --tag adaptok_scorer_t2048 \
            --csv_file k600_train.csv+ucf101_train.csv --out_path work_dir/adaptok/ \
            --name $project_name -b 128 -j 128 \
            --frame_num 16 --input_size 128   \
            --opts \
            test_dataset.csv_paths.ucf101_val ucf101_val.csv \
            train_dataset.args.score_type $score_type \
            test_dataset.args.score_type $score_type \
            model.args.mode train_online_scorer \
            model.args.decode_mode with_drop \
            model.args.bottleneck_token_num 2048 \
            model.args.mask_generator.mask_type left_masking_by_group_normal \
            model.args.mask_generator.tot_groups 4 \
            model.args.mask_generator.min_toks 32 \
            model.args.mask_generator.max_toks 512 \
            model.args.mask_generator.mean_toks 256 \
            model.args.mask_generator.std_toks 128 \
            model.args.mask_generator.total_toks 512 \
            model.args.scorer_step 64 \
            model.args.scorer_attn_type full_causal_type1 \
            model.args.encoder_hidden_size 768 \
            model.args.decoder_hidden_size 768 \
            model.args.encoder_depth 12 \
            model.args.decoder_depth 12 \
            model.args.encoder_num_heads 12 \
            model.args.decoder_num_heads 12 \
            model.args.bottleneck.args.regularizer.name vq \
            model.args.prior_model.name gptc-S \
            loss.args.disc_tran_hidden_size 512 \
            loss.args.disc_tran_n_heads 8 \
            loss.args.disc_tran_n_layers 12 \
            optimizer.args.lr 0.0001  \
            optimizer.loss_args.lr 0.00003 \
            optimizer.warmup_epoch 8 \
            optimizer.min_lr_mult 0.01 \
            optimizer.prior_lr_mult 50.0 \
            optimizer.lr_type cosine \
            compile true \
            use_amp true \
            compile_mode default \
            init_checkpoint $init_checkpoint \
            vis_epoch 1000000 eval_epoch 1  max_epoch 150 latest_interval 1 save_best true save_epoch 20 \
            stepwise_logging true stepwise_logging_interval 500 \
            --wandb-upload --wandb_project $wandb_project_name