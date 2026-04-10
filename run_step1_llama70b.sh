#!/bin/bash
echo "Step 1: Identifying UA-FFN layers for Llama-3.3-70B-Instruct-AWQ"

CUDA_VISIBLE_DEVICES=0 python3 step1_identify_ua_ffn_llama70b.py \
  --model_name /home/aied_test/models/Llama-3.3-70B-Instruct-AWQ \
  --in_file_path ./data/func_data/draw_acctivations_shuf1k.jsonl \
  --output_dir ./results/llama70b_awq_step1 \
  --model_type llama31 \
  --top_k 10
