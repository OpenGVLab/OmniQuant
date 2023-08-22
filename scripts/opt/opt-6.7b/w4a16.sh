CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-6.7b  \
--epochs 20 --output_dir ./log/opt-6.7b-w4a16 \
--wbits 4 --abits 16 --lwc --let