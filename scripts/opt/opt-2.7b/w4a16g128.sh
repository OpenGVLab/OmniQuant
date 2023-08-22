CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-2.7b --eval_ppl \
--epochs 20 --output_dir ./log/opt-2.7b-w4a16g128 \
--wbits 4 --abits 16 --group_size 128 --lwc --let