import tqdm
import argparse
import torch
import transformers
import json
import sys
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

# ── ParamMute suppression: use the modified transformers from ParamMute ──────
PARAMMUTE_SRC = "/home/aied_test/ParamMute/src/transformers/src"
if PARAMMUTE_SRC not in sys.path:
    sys.path.insert(0, PARAMMUTE_SRC)
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM_w_act_inhibit
# ─────────────────────────────────────────────────────────────────────────────

LABEL = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

INHIBIT_RATIO = 0.75
INHIBIT_LAYERS = [6, 7, 8, 17, 18, 19, 20, 25, 26, 27]


def truncate_chat_prompt(prompt: str, tokenizer, max_len: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
        return tokenizer.decode(token_ids, add_special_tokens=False)
    else:
        return prompt


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def get_label_from_output(output: str) -> Optional[str]:
    if "Not Enough Evidence" in output:
        return "Not Enough Evidence"
    elif any(x in output for x in ["Conflicting Evidence/Cherrypicking", "Cherrypicking", "Conflicting Evidence"]):
        return "Conflicting Evidence/Cherrypicking"
    elif any(x in output for x in ["Supported", "supported"]):
        return "Supported"
    elif any(x in output for x in ["Refuted", "refuted"]):
        return "Refuted"
    return None


def prepare_prompts(examples: List[Dict], tokenizer) -> List[str]:
    base_prompt = (
        "Your task is to predict the verdict of a claim based on the provided "
        "question-answer pair evidence. Choose from the labels: 'Supported', "
        "'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking'. "
        "Disregard irrelevant question-answer pairs when assessing the claim. "
        "Justify your decision step by step using the provided evidence and "
        "select the appropriate label."
    )
    prepared_inputs = []
    for example in examples:
        example["input_str"] = (
            base_prompt
            + "\n\nClaim: "
            + example["claim"]
            + "\n\n"
            + "\n\n".join(
                [
                    f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}"
                    for i, qa in enumerate(example["evidence"])
                ]
            )
        )
        example["input_str"] = truncate_chat_prompt(
            example["input_str"], tokenizer, max_len=3584
        )
        messages = [{"role": "user", "content": example["input_str"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prepared_inputs.append(input_ids)
    return prepared_inputs


def generate_hf(model, tokenizer, prompt: str, device, max_new_tokens: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")
    print(f"Suppression ON — lambda={INHIBIT_RATIO}, layers={INHIBIT_LAYERS}")

    data_load_start = time.time()
    try:
        with open(args.target_data) as f:
            examples = json.load(f)
    except Exception:
        examples = []
        with open(args.target_data) as f:
            for line in f:
                examples.append(json.loads(line))
    print(f"Data loading took: {format_time(time.time() - data_load_start)}")
    print(f"Total examples: {len(examples)}")

    model_start = time.time()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading suppressed model on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen2ForCausalLM_w_act_inhibit.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        inhibit_strength=INHIBIT_RATIO,
        inhibit_layer_list=INHIBIT_LAYERS,
    )
    model.eval()
    print(f"Model loaded in {format_time(time.time() - model_start)}")

    predictions = []
    processing_start = time.time()
    batch_size = args.batch_size

    for batch_idx in tqdm.tqdm(range(0, len(examples), batch_size), desc="Batches"):
        batch_end = min(batch_idx + batch_size, len(examples))
        current_batch = examples[batch_idx:batch_end]
        batch_inputs = prepare_prompts(current_batch, tokenizer)

        for example, prompt in zip(current_batch, batch_inputs):
            output_text = generate_hf(model, tokenizer, prompt, device)
            label = get_label_from_output(output_text)

            retry_count = 0
            while label is None and retry_count < 3:
                retry_count += 1
                output_text = generate_hf(model, tokenizer, prompt, device)
                label = get_label_from_output(output_text)
                if label is None:
                    print(f"RAW OUTPUT: {output_text[:200]}")
                    print(f"  Retry {retry_count}: no label found.")

            predictions.append(
                {
                    "claim_id": example["claim_id"],
                    "claim": example["claim"],
                    "evidence": example["evidence"],
                    "pred_label": label or "Not Enough Evidence",
                    "llm_output": output_text,
                }
            )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    total_time = time.time() - script_start
    processing_time = time.time() - processing_start
    print(f"\nDone. Total: {format_time(total_time)}")
    print(f"Processing: {format_time(processing_time)}")
    print(f"Results -> {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("-i", "--target_data",
                        default="data_store/hero2/dev_top_k_qa_rewrite.json")
    parser.add_argument("-o", "--output_file",
                        default="data_store/hero2/dev_veracity_prediction_suppress_veracity.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
