#!/bin/bash
# Step 1: UA-FFN identification on Llama-3.3-70B-Instruct (bf16, full precision)
# Uses 2x A100 80GB with device_map="auto" (weights split across cards)
# Output: ./results/llama70b_bf16_step1/

echo "Step 1: Identifying UA-FFN layers for Llama-3.3-70B-Instruct (bf16)"
echo "Using GPUs 0 and 1 for tensor-parallel bf16 loading (~140 GB)"

CUDA_VISIBLE_DEVICES=0,1 python3 step1_identify_ua_ffn_llama70b.py \
  --model_name /home/aied_test/models/Llama-3.3-70B-Instruct \
  --in_file_path ./data/func_data/draw_acctivations_shuf1k.jsonl \
  --output_dir ./results/llama70b_bf16_step1 \
  --model_type llama31 \
  --top_k 10
