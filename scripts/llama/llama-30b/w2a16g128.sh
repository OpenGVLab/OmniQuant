CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-30b --eval_ppl \
--epochs 40 --output_dir ./log/llama-30b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss