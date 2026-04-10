import os
import argparse
import time
import json
import nltk
import torch
import numpy as np
from datetime import datetime, timedelta
from rank_bm25 import BM25Okapi

from suppression_utils import load_model_with_suppression, add_suppression_args


def download_nltk_data(package_name, download_dir='nltk_data'):
    os.makedirs(download_dir, exist_ok=True)
    nltk.data.path.append(download_dir)
    try:
        nltk.data.find(f'tokenizers/{package_name}')
    except LookupError:
        nltk.download(package_name, download_dir=download_dir)


def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))


def claim2prompts(example):
    claim = example["claim"]
    claim_str = "Example [NUMBER]:||Claim: " + claim + "||Evidence: "
    for question in example["questions"]:
        q_text = question["question"].strip()
        if len(q_text) == 0:
            continue
        if not q_text[-1] == "?":
            q_text += "?"
        answer_strings = []
        for a in question["answers"]:
            if a["answer_type"] in ["Extractive", "Abstractive"]:
                answer_strings.append(a["answer"])
            if a["answer_type"] == "Boolean":
                answer_strings.append(a["answer"] + ", because " + a["boolean_explanation"].lower().strip())
        for a_text in answer_strings:
            if not a_text[-1] in [".", "!", ":", "?"]:
                a_text += "."
            prompt_lookup_str = a_text
            this_q_claim_str = claim_str + a_text.strip() + "||Question: " + q_text
            yield (prompt_lookup_str, this_q_claim_str.replace("\n", " ").replace("||", "\n")[:1500])


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
    script_start = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suppression: layers={args.suppress_layers or 'none'}, ratio={args.inhibit_ratio}")
    print(f"Resuming from claim index: {args.start}")

    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')

    corpus_start = time.time()
    with open(args.reference_corpus, "r", encoding="utf-8") as f:
        train_examples = json.load(f)

    prompt_corpus, tokenized_corpus = [], []
    for example in train_examples:
        for lookup_str, prompt in claim2prompts(example):
            entry = nltk.word_tokenize(lookup_str)
            tokenized_corpus.append(entry)
            prompt_corpus.append(prompt)

    prompt_bm25 = BM25Okapi(tokenized_corpus)
    print(f"Reference corpus processed in: {format_time(time.time() - corpus_start)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")
    model, tokenizer, suppressor = load_model_with_suppression(
        args.model, device,
        suppress_layers=args.suppress_layers or None,
        inhibit_ratio=args.inhibit_ratio,
    )
    print(f"Model loaded in: {format_time(time.time() - script_start)}")

    target_examples = []
    with open(args.top_k_target_knowledge, "r", encoding="utf-8") as f:
        for line in f:
            target_examples.append(json.loads(line))

    if args.end == -1:
        args.end = len(target_examples)
    print(f"Processing claims {args.start} to {args.end} ({args.end - args.start} total)")

    with open(args.output_questions, "a", encoding="utf-8") as output_file:
        for idx, example in enumerate(target_examples[args.start:args.end], start=args.start):
            batch_start = time.time()
            claim = example["claim"]
            claim_id = example["claim_id"]
            top_k_sentences_urls = example[f"top_{args.top_k}"]

            evidence = []
            for sentences_urls in top_k_sentences_urls:
                prompt_lookup_str = sentences_urls["sentence"]
                url = sentences_urls["url"]

                prompt_s = prompt_bm25.get_scores(nltk.word_tokenize(prompt_lookup_str))
                prompt_top_n = np.argsort(prompt_s)[::-1][:10]
                prompt_docs = [prompt_corpus[i] for i in prompt_top_n]

                temp_prompt = "\n\n".join(prompt_docs)
                for k in range(1, temp_prompt.count("[NUMBER]") + 1):
                    temp_prompt = temp_prompt.replace("[NUMBER]", f"{k}", 1)

                claim_prompt = "Your task is to generate a question based on the given claim and evidence. The question should clarify the relationship between the evidence and the claim\n\n"
                ev_text = prompt_lookup_str.replace("\n", " ")
                full_prompt = claim_prompt + temp_prompt + "\n\nNow, generate a question that links the following claim and evidence:" + f"\n\nClaim: {claim}" + f"\nEvidence: {ev_text}"

                messages = [{"role": "user", "content": full_prompt}]
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) + "Question: "

                output_text = generate_hf(model, tokenizer, prompt_str, device)
                question = output_text.strip().split("?")[0].replace("\n", " ") + "?"

                evidence.append({
                    "question": question,
                    "answer": prompt_lookup_str,
                    "url": url
                })

            json_data = {
                "claim_id": claim_id,
                "claim": claim,
                "evidence": evidence
            }
            output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_file.flush()

            batch_time = time.time() - batch_start
            print(f"Processed example {idx+1}/{args.end} (claim_id={claim_id}). Time: {batch_time:.2f}s")

    if suppressor:
        suppressor.remove()

    total_time = time.time() - script_start
    print(f"\nDone. Total: {format_time(total_time)}")
    print(f"Results -> {args.output_questions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reference_corpus", default="data_store/averitec/train.json")
    parser.add_argument("-i", "--top_k_target_knowledge",
                        default="data_store/hero2/dev_retrieval_top_k.json")
    parser.add_argument("-o", "--output_questions",
                        default="data_store/hero2/dev_top_k_qa_suppress_qgen.json")
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("-e", "--end", type=int, default=-1)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    add_suppression_args(parser)
    args = parser.parse_args()
    main(args)
