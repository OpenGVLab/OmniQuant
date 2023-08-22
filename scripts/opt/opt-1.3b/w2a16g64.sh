CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-1.3b  --eval_ppl \
--epochs 40 --output_dir ./log/opt-1.3b-w2a16g64 \
--wbits 2 --abits 16 --group_size 64 --lwc --let