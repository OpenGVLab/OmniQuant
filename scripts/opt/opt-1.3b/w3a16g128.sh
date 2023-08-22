CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-1.3b  --eval_ppl \
--epochs 20 --output_dir ./log/opt-1.3b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --let