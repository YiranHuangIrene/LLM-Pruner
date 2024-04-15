#!/bin/bash
export PYTHONPATH='.'

ratio=(0 0.2)

for r in "${ratio[@]}"; 
do
    python lm-evaluation-harness/main.py --model_args checkpoint=/shared-local/aoq609/LLM-Pruner/LLMPruner/prune_log/llava-v1.5-7b_${r}/pytorch_model.bin,config_pretrained=liuhaotian/llava-v1.5-7b --tasks hellaswag,arc_challenge,piqa --device cuda --output_path /shared-local/aoq609/LLM-Pruner/lm-evaluation-harness/results/results_llava-v1.5-7b_${r}.json
done
