CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-7b --eval_ppl \
--epochs 20 --output_dir ./log/llama-7b-w3a16 \
--wbits 3 --abits 16 --lwc