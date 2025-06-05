MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

project_name=adaptok_ar_fp

ckpt_path=path_to_adaptok.pth

# e.g., data/k600_annotations/k600_train_ilp_grouped
train_ann=path_to_fp_annotations_after_post_processing_trainset
test_ann=path_to_fp_annotations_after_post_processing_valset

torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
            train.py --cfg cfgs/adaptok_ar_fp.yaml \
            --manualSeed 66667 --tag adapar_fp_avgtok1024 \
            --csv_file k600_train.csv --out_path work_dir/adaptok_ar_fp/ \
            --name adaptok_ar_fp -b 64 -j 128 \
            --frame_num 16 --input_size 128 \
            --opts \
            test_dataset.csv_paths.k600_val k600_val.csv \
            train_dataset.args.ar_ann_path $train_ann \
            test_dataset.args.ar_ann_path $test_ann \
            model.name llama-abs-LP \
            model.args.start_ar_block_idx 1 \
            vae.name adaptok \
            vae.checkpoint $ckpt_path \
            ar.num_cond_frames 5 \
            ar.num_samples 128 \
            optimizer.name adamw \
            optimizer.args.weight_decay 0.05 \
            optimizer.warmup_epoch 1 \
            optimizer.args.lr 0.0006  \
            use_amp true \
            compile true \
            vis_epoch 1 eval_epoch 1 max_epoch 75 latest_interval 1 save_epoch 25 \
            stepwise_logging true stepwise_logging_interval 500 \
            --wandb-upload --wandb_project $project_name





