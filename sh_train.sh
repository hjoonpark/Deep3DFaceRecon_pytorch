# clear;
# python train.py --name="JoonTest" --gpu_ids=0
python train.py \
--flist 'datalist/picked/train/masks.txt' \
--use_aug false \
--batch_size 5 \
--display_freq 5000 \
--evaluation_freq 5000 \
--n_epochs 20000 \