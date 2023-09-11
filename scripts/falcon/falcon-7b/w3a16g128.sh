CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/Falcon/falcon-7b --eval_ppl \
--epochs 10 --output_dir ./log/falcon-7b-w3a16g64 \
--wbits 3 --abits 16 --group_size 64 --lwc --aug_loss