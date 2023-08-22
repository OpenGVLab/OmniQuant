CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/Llama-2-13b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-13b-w6a6 \
--wbits 6 --abits 6 --lwc --let