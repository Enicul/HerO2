import copy
import json
import tqdm
import argparse
import torch
import sys
from datetime import datetime, timedelta
import time

# ── ParamMute suppression ─────────────────────────────────────────────────────
PARAMMUTE_SRC = "/home/aied_test/ParamMute/src/transformers/src"
if PARAMMUTE_SRC not in sys.path:
    sys.path.insert(0, PARAMMUTE_SRC)
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM_w_act_inhibit
# ─────────────────────────────────────────────────────────────────────────────

INHIBIT_RATIO = 0.5
INHIBIT_LAYERS = [6, 7, 8, 17, 18, 19, 20, 25, 26, 27]

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
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")
    print(f"Suppression ON — lambda={INHIBIT_RATIO}, layers={INHIBIT_LAYERS}")

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

    total_time = time.time() - script_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    args = parser.parse_args()
    main(args)
