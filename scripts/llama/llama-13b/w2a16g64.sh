CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-13b --eval_ppl \
--epochs 40 --output_dir ./log/llama-13b-w2a16g64 \
--wbits 2 --abits 16 --group_size 64 --lwc --aug_loss