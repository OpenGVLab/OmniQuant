CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-65b --eval_ppl \
--epochs 20 --output_dir ./log/llama-65b-w3a16 \
--wbits 3 --abits 16 --lwc