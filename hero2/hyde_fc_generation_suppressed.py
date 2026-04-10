import json
import torch
import time
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

from suppression_utils import load_model_with_suppression, add_suppression_args


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
    print(f"Suppression: layers={args.suppress_layers or 'none'}, ratio={args.inhibit_ratio}")

    print("Loading data...")
    with open(args.target_data, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")
    model, tokenizer, suppressor = load_model_with_suppression(
        args.model, device,
        suppress_layers=args.suppress_layers or None,
        inhibit_ratio=args.inhibit_ratio,
    )
    print(f"Model loaded in {format_time(time.time() - total_start_time)}")

    processed_data = []
    for example in tqdm(examples, desc="Processing examples"):
        prompt = prepare_prompt(example["claim"], tokenizer)
        hypo_docs = []
        for _ in range(8):
            output = generate_hf(model, tokenizer, prompt, device)
            hypo_docs.append(output)
        example['hypo_fc_docs'] = hypo_docs
        processed_data.append(example)

    total_time = time.time() - total_start_time
    print(f"\nSaving results...")
    for claim_id, example in enumerate(processed_data):
        if not example.get("claim_id"):
            example['claim_id'] = claim_id

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    if suppressor:
        suppressor.remove()

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
    add_suppression_args(parser)
    args = parser.parse_args()
    main(args)
