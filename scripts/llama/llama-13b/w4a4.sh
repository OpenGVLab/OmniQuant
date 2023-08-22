CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-13b --eval_ppl \
--epochs 20 --output_dir ./log/llama-13b-w4a4 \
--wbits 4 --abits 4 --lwc --let --aug_loss