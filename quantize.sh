CUDA_VISIBLE_DEVICES=0 

python main.py \
--model meta-llama/Llama-2-7b-hf \
--epochs 20 \
--eval_ppl --wbits 3 --abits 4 --lwc --let \
--let_ortho --let_ortho_k 1 \
--act-scales /workspace/act_scales/Llama-2-7b-hf.pt --act-shifts /workspace/act_shifts/Llama-2-7b-hf.pt \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande