gpu=$1
exp_name=$2
ds=$3
bs=64
vq_name=$4


echo "Training residual transformer on ${vq_name}" 

CUDA_VISIBLE_DEVICES=${gpu} python -m train_res_transformer --name ${exp_name} --dataset_name ${ds} --batch_size ${bs} --vq_name ${vq_name} --cond_drop_prob 0.2 --share_weight
