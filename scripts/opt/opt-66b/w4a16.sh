CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-66b --eval_ppl \
--epochs 20 --output_dir ./log/opt-66b-w4a16 \
--wbits 4 --abits 16 --lwc --let