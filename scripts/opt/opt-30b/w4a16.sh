CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-30b --eval_ppl \
--epochs 20 --output_dir ./log/opt-30b-w4a16 \
--wbits 4 --abits 16 --lwc --let