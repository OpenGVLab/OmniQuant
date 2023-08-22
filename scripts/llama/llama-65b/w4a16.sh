CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/llama-65b --eval_ppl \
--epochs 20 --output_dir ./log/llama-65b-w4a16 \
--wbits 4 --abits 16 --lwc