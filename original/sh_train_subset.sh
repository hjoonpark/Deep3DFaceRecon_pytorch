# LR=2e-5
# N_EPOCH=10
# PRINT_FREQ=1
# DISP_FREQ=5
# EVAL_FREQ=5
# PRESET_AGE=0
# PRESET_GENDER=0

LR=1e-4
N_EPOCH=100000000000
PRINT_FREQ=60
DISP_FREQ=1200
DISP_OVERWRITE_FREQ=600
EVAL_FREQ=1800

python -W ignore train.py \
--flist 'datalist/ffhq7k_celeba3k/train/masks.txt' \
--flist_val 'datalist/ffhq7k_celeba3k/val/images.txt' \
--checkpoints_dir ../results/checkpointNetwork_$RECON_NETWORK_TYPE \
--net_recog_path '/home/joonp/1_Projects/gitlab/ngc/assets/checkpoints_BACKUP/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth' \
--batch_size 16 \
--batch_size_val 16 \
--print_freq $PRINT_FREQ \
--display_freq $DISP_FREQ \
--display_overwrite_freq $DISP_OVERWRITE_FREQ \
--evaluation_freq $EVAL_FREQ \
--n_epochs $N_EPOCH \
--lr $LR 
