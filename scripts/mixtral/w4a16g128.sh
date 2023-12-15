CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/Mixtral-8x7B-v0.1 \
--epochs 10 --output_dir ./log/mixtral-8x7b_w4a16g128 \
--wbits 4 --abits 16 --group_size 128 --lwc  \
--nsamples 128 --net mixtral-8x7b \
--real_quant --eval_ppl \
--save_dir ./checkpoint/mixtral-8x7b_w4a16g128