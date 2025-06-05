MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

export USE_DYNAMIC=True

project_name=adaptok_scorer
init_checkpoint="path_to_scorer.pth"

# ===================== UCF-101 trainset =========================
token_select_mode=ilp
token_select_num=256
frames_step=1
bs_per_gpu=128
tot_bs=$((bs_per_gpu * NNODES * GPUS_PER_NODE))
score_save_name=ucf101_train_step${frames_step}_scorer_${token_select_mode}_tok${token_select_num}_b${bs_per_gpu}_$(date +"%Y-%m-%d_%H-%M-%S").pt

echo "save annotations to ${score_save_name}"
echo "total batch_size=${tot_bs}"

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
            train.py --cfg cfgs/adaptok_scorer.yaml \
            --manualSeed 66667 --tag adaptok_scorer_t2048 \
            --csv_file ucf101_train.csv --out_path work_dir/adaptok/ \
            --name $project_name -b $tot_bs -j 128 \
            --frame_num 16 --input_size 128   \
            --opts \
            init_checkpoint $init_checkpoint \
            train_dataset.args.use_all_frames true \
            train_dataset.args.use_all_frames_step $frames_step \
            train_dataset.args.score_type perceptual \
            test_dataset.csv_paths.ucf101_val ucf101_val.csv \
            test_dataset.args.score_type perceptual \
            score_save_name $score_save_name \
            trainer adaptok_score_labeler \
            model.args.scorer_attn_type full_causal_type1 \
            model.args.mode get_ar_annotations \
            model.args.token_select_mode $token_select_mode \
            model.args.token_select_num $token_select_num \
            model.args.scorer_step 64 \
            model.args.bottleneck_token_num 2048 \
            model.args.mask_generator.tot_groups 4 \
            model.args.mask_generator.max_toks 512 \
            model.args.mask_generator.total_toks 512 \
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
            compile true \
            use_amp true \
            compile_mode default


# ===================== UCF-101 valset =========================

token_select_mode=ilp
token_select_num=256
frames_step=100
bs_per_gpu=128
tot_bs=$((bs_per_gpu * NNODES * GPUS_PER_NODE))
score_save_name=ucf101_val_step${frames_step}_scorer_${token_select_mode}_tok${token_select_num}_b${bs_per_gpu}_$(date +"%Y-%m-%d_%H-%M-%S").pt

echo "save annotations to ${score_save_name}"
echo "total batch_size=${tot_bs}"

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
            train.py --cfg cfgs/adaptok_scorer.yaml \
            --manualSeed 66667 --tag adaptok_scorer_t2048 \
            --csv_file ucf101_val.csv --out_path work_dir/adaptok/ \
            --name $project_name -b $tot_bs -j 128 \
            --frame_num 16 --input_size 128   \
            --opts \
            init_checkpoint $init_checkpoint \
            train_dataset.args.use_all_frames true \
            train_dataset.args.use_all_frames_step $frames_step \
            train_dataset.args.score_type perceptual \
            test_dataset.csv_paths.ucf101_val ucf101_val.csv \
            test_dataset.args.score_type perceptual \
            score_save_name $score_save_name \
            trainer adaptok_score_labeler \
            model.args.scorer_attn_type full_causal_type1 \
            model.args.mode get_ar_annotations \
            model.args.token_select_mode $token_select_mode \
            model.args.token_select_num $token_select_num \
            model.args.scorer_step 64 \
            model.args.bottleneck_token_num 2048 \
            model.args.mask_generator.tot_groups 4 \
            model.args.mask_generator.max_toks 512 \
            model.args.mask_generator.total_toks 512 \
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
            compile true \
            use_amp true \
            compile_mode default