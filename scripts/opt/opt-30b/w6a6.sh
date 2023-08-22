CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-30b --eval_ppl \
--epochs 20 --output_dir ./log/opt-30b-w6a6 \
--wbits 6 --abits 6 --lwc --let --alpha 0.5