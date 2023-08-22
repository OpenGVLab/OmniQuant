CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/Llama-2-13b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-13b-w4a16 \
--wbits 4 --abits 16 --lwc