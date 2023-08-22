CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/LLaMa/Llama-2-13b-chat --eval_ppl \
--epochs 20 --output_dir ./log/Llama-2-13b-chat-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss \
--save_dir /PATH/TO/OmniQuant_Checkpoints/Llama-2-13b-chat-w3a16g128.pth