CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-125m --eval_ppl \
--epochs 20 --output_dir ./log/opt-125m-w4a16g128 \
--wbits 4 --abits 16 --group_size 128 --lwc --let