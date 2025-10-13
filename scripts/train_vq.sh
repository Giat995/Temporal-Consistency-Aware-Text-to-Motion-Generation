gpu=$1
exp_name=$2
ds=$3
clip=$4 # choices=['ViT-B/32', 'AltCLIP-m9', 'OpenCLIP-ViT-B/16', 'stuxlm']
dim=512
mkdir -p ./save/${exp_name}

echo "Training on ${ds} with clip version ${clip} and saving to save/${exp_name}, LatentDim = ${dim}" | tee -a save/${exp_name}/train.log

CUDA_VISIBLE_DEVICES=${gpu} python -m train.train_mdm --save_dir save/${exp_name} --dataset ${ds} --clip_version $clip --latent_dim ${dim} | tee -a save/${exp_name}/train.log
