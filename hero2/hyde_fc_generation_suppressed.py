import json
import torch
import sys
import time
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

# ── ParamMute suppression ─────────────────────────────────────────────────────
PARAMMUTE_SRC = "/home/aied_test/ParamMute/src/transformers/src"
if PARAMMUTE_SRC not in sys.path:
    sys.path.insert(0, PARAMMUTE_SRC)
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM_w_act_inhibit
# ─────────────────────────────────────────────────────────────────────────────

INHIBIT_RATIO = 0.5
INHIBIT_LAYERS = [6, 7, 8, 17, 18, 19, 20, 25, 26, 27]


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def prepare_prompt(claim: str, tokenizer) -> str:
    base_prompt = f"Please write a fact-checking article passage to support, refute, indicate not enough evidence, or present conflicting evidence regarding the claim.\nClaim: {claim}"
    messages = [{"role": "user", "content": base_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "Passage: "


def generate_hf(model, tokenizer, prompt: str, device, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(args):
    total_start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suppression ON — lambda={INHIBIT_RATIO}, layers={INHIBIT_LAYERS}")

    # Load data
    print("Loading data...")
    with open(args.target_data, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Load suppressed model
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
    print(f"Model loaded in {format_time(time.time() - total_start_time)}")

    # Process examples
    processed_data = []
    for example in tqdm(examples, desc="Processing examples"):
        prompt = prepare_prompt(example["claim"], tokenizer)
        # Generate n=8 hypothetical documents (same as original)
        hypo_docs = []
        for _ in range(8):
            output = generate_hf(model, tokenizer, prompt, device)
            hypo_docs.append(output)
        example['hypo_fc_docs'] = hypo_docs
        processed_data.append(example)

    # Save results
    total_time = time.time() - total_start_time
    print(f"\nSaving results...")
    for claim_id, example in enumerate(processed_data):
        if not example.get("claim_id"):
            example['claim_id'] = claim_id

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"Done. Total: {format_time(total_time)}")
    print(f"Results -> {args.json_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--target_data',
                        default='data_store/averitec/dev.json')
    parser.add_argument('-o', '--json_output',
                        default='data_store/hero2/dev_hyde_fc_suppress_hyde.json')
    parser.add_argument('-m', '--model',
                        default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)
