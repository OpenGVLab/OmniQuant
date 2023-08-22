CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/Llama-2-7b --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let  \
--let_lr 1e-3 --alpha 0.75