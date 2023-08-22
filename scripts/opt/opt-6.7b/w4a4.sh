CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-6.7b --eval_ppl \
--epochs 20 --output_dir ./log/opt-6.7b-w4a4 \
--wbits 4 --abits 4 --lwc --let --alpha 0.75