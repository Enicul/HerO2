"""Closed-book evaluation: verifier model with claim only, no evidence."""

import json
import argparse
import os
from collections import Counter
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="humane-lab/Qwen3-32B-AWQ-HerO")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    args = parser.parse_args()

    with open(args.prediction_file) as f:
        predictions = json.load(f)
    with open(args.gold_file) as f:
        gold_data = json.load(f)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )

    LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

    results = []
    for idx, pred in enumerate(predictions):
        claim_id = pred["claim_id"]
        claim = pred["claim"]
        gold_label = gold_data[claim_id].get("label", "")

        prompt = f"""You are a fact-checking expert. Based ONLY on your own knowledge (no external evidence provided), classify the following claim into one of these categories:
- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence/Cherrypicking

Claim: {claim}

Provide your verdict as a single label from the categories above. Then provide a brief justification.

verdict: """

        messages = [{"role": "user", "content": prompt}]

        # Run multiple times
        raw_verdicts = []
        for run in range(args.num_runs):
            outputs = llm.generate([prompt], sampling_params)
            output_text = outputs[0].outputs[0].text.strip()

            # Extract verdict
            verdict = "Not Enough Evidence"  # default
            output_lower = output_text.lower()
            for label in LABELS:
                if label.lower() in output_lower:
                    verdict = label
                    break
            raw_verdicts.append(verdict)

        # Majority vote
        vote_counts = Counter(raw_verdicts)
        majority_verdict = vote_counts.most_common(1)[0][0]
        consistent = len(set(raw_verdicts)) == 1
        high_confidence = vote_counts.most_common(1)[0][1] == args.num_runs

        result = {
            "claim_id": claim_id,
            "claim": claim,
            "gold_verdict": gold_label,
            "full_context_verdict": pred["pred_label"],
            "closed_book_verdict": majority_verdict,
            "closed_book_verdicts_raw": raw_verdicts,
            "closed_book_consistent": consistent,
            "closed_book_high_confidence": high_confidence,
            "evidence": pred["evidence"],
            "justification": pred.get("llm_output", "")
        }
        results.append(result)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(predictions)}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n=== Closed-Book Summary ({args.num_runs} runs) ===")
    print(f"Total claims: {len(results)}")
    cb_correct = sum(1 for r in results if r["closed_book_verdict"].lower() == r["gold_verdict"].lower())
    fc_correct = sum(1 for r in results if r["full_context_verdict"].lower() == r["gold_verdict"].lower())
    consistent_count = sum(1 for r in results if r["closed_book_consistent"])
    hc_count = sum(1 for r in results if r["closed_book_high_confidence"])
    print(f"Full-context accuracy: {fc_correct}/{len(results)} = {fc_correct/len(results):.2%}")
    print(f"Closed-book accuracy: {cb_correct}/{len(results)} = {cb_correct/len(results):.2%}")
    print(f"Consistent (all same): {consistent_count}/{len(results)} = {consistent_count/len(results):.2%}")
    print(f"High-confidence: {hc_count}/{len(results)} = {hc_count/len(results):.2%}")

    cb_dist = Counter(r["closed_book_verdict"] for r in results)
    print(f"\nClosed-book distribution: {dict(cb_dist.most_common())}")

if __name__ == "__main__":
    main()
