CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-30b  \
--epochs 20 --output_dir ./log/llama-30b-w6a6 \
--wbits 6 --abits 6 --lwc --let