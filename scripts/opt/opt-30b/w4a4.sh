CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-30b  \
--epochs 20 --output_dir ./log/opt-30b-w4a4 \
--wbits 4 --abits 4 --lwc --let --alpha 0.75