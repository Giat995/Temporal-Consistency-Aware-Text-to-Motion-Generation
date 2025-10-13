CUDA_VISIBLE_DEVICES=3 python train_vq_tcc.py --name t11 --dataset_name t2m  --code_dim 512 --nb_code 512 --vq_act relu --num_quantizers 6  --quantize_dropout_prob 0.2 --gamma 0.05  --batch_size 256  --shared_codebook
CUDA_VISIBLE_DEVICES=3 python train_vq_tcc.py --name t15 --dataset_name t2m  --code_dim 512 --nb_code 512 --vq_act relu --num_quantizers 6  --quantize_dropout_prob 0.4 --gamma 0.05  --batch_size 256 
bash ./scripts/train_mtrans.sh 3 t11 t2m 2025-03-25-19-39-02_t11
bash ./scripts/train_rtrans.sh 3 t11 t2m 2025-03-25-19-39-02_t11
bash ./scripts/train_mtrans.sh 3 t15 t2m 2025-03-26-06-19-11_t15
bash ./scripts/train_rtrans.sh 3 t15 t2m 2025-03-26-06-19-11_t15