# run this script for trainset and valset respectively.

ann_pattern=pattern_to_ar-annotations
output_dir=annotation_save_path

# # for example
# ann_pattern=work_dir/adaptok/adaptok_scorer/b128_btn2048_vq_rcs8192__adaptok_scorer_t2048/annotations/k600_val_step100_scorer_ilp_tok256_b128_2025-05-29_22-49-04.pt_sub*.pt
# output_dir=data/k600_annotations/k600_val_ilp_grouped

python data/split_annotations.py \
   --ann_pattern $ann_pattern \
   --output_dir $output_dir
   