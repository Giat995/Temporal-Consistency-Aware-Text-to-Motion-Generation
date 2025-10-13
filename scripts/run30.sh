CUDA_VISIBLE_DEVICES=3 python train_vq_tcc.py --name t3 --dataset_name t2m  --code_dim 512 --nb_code 2048 --vq_act relu --num_quantizers 6  --quantize_dropout_prob 0.2 --gamma 0.05  --batch_size 256  
CUDA_VISIBLE_DEVICES=3 python train_vq_tcc.py --name t7 --dataset_name t2m  --code_dim 1024 --nb_code 1024 --vq_act relu --num_quantizers 6  --quantize_dropout_prob 0.2 --gamma 0.05  --batch_size 256
bash ./scripts/train_mtrans.sh 3 t3 t2m 2025-03-25-19-38-45_t3
bash ./scripts/train_rtrans.sh 3 t3 t2m 2025-03-25-19-38-45_t3
bash ./scripts/train_mtrans.sh 3 t7 t2m 2025-03-26-07-40-18_t7
bash ./scripts/train_rtrans.sh 3 t7 t2m 2025-03-26-07-40-18_t7