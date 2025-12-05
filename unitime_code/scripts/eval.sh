export CUDA_VISIBLE_DEVICES=2,3
export DECORD_EOF_RETRY_MAX=20480


python inference1.py --model_local_path ../UniTime/checkpoints/model_local1_path \
    --model_finetune_path ../UniTime/checkpoints/Total_768_LoRA32_LR2e4_epoch2_clip_50_newGT \
    --video_root /media/vlilab/DATA/MEMBER/QIANG/data/Charades_v1/videos \
    --feat_folder path_to_feat_folder \
    --data_path ../UniTime/data/charades_sta/test1.json \
    --output_dir ../UniTime/results/test1/charades_1fps/ \
    --nf_short 128 \

    