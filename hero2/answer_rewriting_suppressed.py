import json
import tqdm
import argparse
import torch
from datetime import datetime, timedelta
import time

from suppression_utils import load_model_with_suppression, add_suppression_args

prompt = """The following text provides an evidence obtained through web searches related to a specific question, used for verifying the accuracy of a claim.
Your task is to answer the question based on this evidence. Ensure your answer is Supported by relevant context from the evidence.

Claim: {}
Question: {}

Evidence: {}
"""


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def generate_hf(model, tokenizer, prompt_text: str, device, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
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
    script_start = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suppression: layers={args.suppress_layers or 'none'}, ratio={args.inhibit_ratio}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")
    model, tokenizer, suppressor = load_model_with_suppression(
        args.model, device,
        suppress_layers=args.suppress_layers or None,
        inhibit_ratio=args.inhibit_ratio,
    )
    print(f"Model loaded in {format_time(time.time() - script_start)}")

    data = []
    with open(args.target_data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    with open(args.json_output, "w", encoding="utf-8") as output_json:
        for d in tqdm.tqdm(data, desc="Claims"):
            all_prompts = []
            for item in d["evidence"]:
                target_prompt = prompt.format(d["claim"], item["question"], item["answer"])
                target_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": target_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                target_prompt += "Answer: "
                all_prompts.append(target_prompt)

            results = []
            for p in all_prompts:
                result = generate_hf(model, tokenizer, p, device)
                results.append(result)

            new_evidence = []
            for num, item in enumerate(d["evidence"]):
                if results[num].lower() != "none":
                    item["answer"] = results[num]
                    new_evidence.append(item)
            d["evidence"] = new_evidence

            output_json.write(json.dumps(d, ensure_ascii=False) + "\n")
            output_json.flush()

    if suppressor:
        suppressor.remove()

    total_time = time.time() - script_start
    print(f"\nDone. Total: {format_time(total_time)}")
    print(f"Results -> {args.json_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--target_data",
                        default="data_store/hero2/dev_top_k_qa.json")
    parser.add_argument("-o", "--json_output",
                        default="data_store/hero2/dev_top_k_qa_rewrite_suppress_rewrite.json")
    parser.add_argument("-m", "--model",
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=2)
    add_suppression_args(parser)
    args = parser.parse_args()
    main(args)
