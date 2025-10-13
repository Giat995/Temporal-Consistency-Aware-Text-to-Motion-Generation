#python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
gpu=$1 
ds=$2
rtrans_name=$3
vq_name=$4
mtrans_name=$5
name=$6

echo "Eval on ${ds} on ${name}"


CUDA_VISIBLE_DEVICES=${gpu} python -m eval_t2m_trans_res --dataset_name ${ds} --rtrans_name ${rtrans_name}  --vq_name ${vq_name} --mtrans_name ${mtrans_name} --name ${name} --cond_scale 4 --time_steps 10 --ext evaluation