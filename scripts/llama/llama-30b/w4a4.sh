CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/llama-30b --eval_ppl \
--epochs 20 --output_dir ./log/llama-30b-w4a4 \
--wbits 4 --abits 4 --lwc --let --alpha 0.75 \
--aug_loss