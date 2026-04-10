#!/bin/bash

SYSTEM_NAME="hero2"
SPLIT="dev"
BASE_DIR="."
DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"

export HF_HOME="${BASE_DIR}/huggingface_cache"
export HUGGING_FACE_HUB_TOKEN="[YOUR TOKEN]"

mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

python hero2/hyde_fc_generation.py \
  --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
  --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
  --model "Qwen/Qwen2.5-7B-Instruct" || exit 1

python hero2/retrieval.py \
  --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
  --embedding_store_dir "${KNOWLEDGE_STORE}/${SPLIT}_embed" \
  --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}_summary" \
  --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
  --model "Alibaba-NLP/gte-base-en-v1.5" \
  --top_k 10 || exit 1

python hero2/question_generation.py \
  --reference_corpus "${DATA_STORE}/averitec/train.json" \
  --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
  --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
  --model "Qwen/Qwen2.5-7B-Instruct" || exit 1

python hero2/answer_rewriting.py \
  --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
  --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
  --model "Qwen/Qwen2.5-7B-Instruct" || exit 1

python hero2/veracity_prediction.py \
  --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_rewrite.json" \
  --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
  --model "Qwen/Qwen2.5-7B-Instruct" || exit 1
