CUDA_VISIBLE_DEVICES=0 

python main.py \
--model meta-llama/Llama-2-7b-hf \
--epochs 20 \
--eval_ppl --wbits 3 --abits 16 --lwc