MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8

project_name=adaptok_ar

ckpt_path=path_to_adaptok.pth

# e.g., work_dir/adaptok/adaptok_scorer/*__adaptok_scorer_t2048/annotations/ucf101_val_step100_scorer_ilp_tok256_b128_(date).pt
train_ann=path_to_ar_annotations_trainset.pt
test_ann=path_to_ar_annotations_valset.pt


torchrun --nnodes=${NNODES} \
         --nproc_per_node=${GPUS_PER_NODE} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
            train.py --cfg cfgs/adaptok_ar.yaml \
            --manualSeed 66667 --tag adapar_avgtok1024 \
            --csv_file ucf101_train.csv --out_path work_dir/adaptok_ar/ \
            --name adaptok_ar -b 64 -j 128 \
            --frame_num 16 --input_size 128 \
            --opts \
            test_dataset.csv_paths.ucf101_val ucf101_val.csv \
            train_dataset.args.ar_ann_path $train_ann \
            test_dataset.args.ar_ann_path $test_ann \
            model.name llama-abs-LP \
            vae.checkpoint $ckpt_path \
            ar.num_samples 32 \
            optimizer.name adamw \
            optimizer.args.weight_decay 0.05 \
            optimizer.warmup_epoch 4 \
            optimizer.args.lr 0.0006  \
            use_amp true \
            compile true \
            vis_epoch 30 eval_epoch 30 max_epoch 3000 latest_interval 30 save_epoch 1000 \
            stepwise_logging true stepwise_logging_interval 500 \
            --wandb-upload --wandb_project $project_name
            




