CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/Falcon/falcon-180b \
--epochs 40 --output_dir ./log/falcon-180b-w3a16g512 \
--wbits 3 --abits 16 --group_size 512 --lwc --aug_loss \
--nsamples 32