"""
FFN Suppression Ablation: Test whether suppressing candidate FFN layers
changes parametric override behavior.

Tests:
1. No suppression (baseline)
2. Narrow late-layer suppression (31-35)
3. Broad late-layer suppression (27-35)
4. Random-layer suppression (5 random layers as control)
"""

import json
import argparse
import torch
import numpy as np
import random
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

def extract_verdict(text):
    text_lower = text.lower()
    for label in LABELS:
        if label.lower() in text_lower:
            return label
    return "Not Enough Evidence"

def build_prompt(claim, evidence=None):
    if evidence:
        evidence_text = "\n".join([
            f"Q: {e['question']}\nA: {e['answer']}"
            for e in evidence[:5]
        ])
        return f"""You are a fact-checking expert. Based on the provided evidence, classify the following claim into one of these categories:
- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence/Cherrypicking

Claim: {claim}

Evidence:
{evidence_text}

verdict: """
    else:
        return f"""You are a fact-checking expert. Based ONLY on your own knowledge, classify the following claim.

Claim: {claim}

verdict: """

class FFNSuppressor:
    """Hook-based FFN suppression for specified layers."""
    
    def __init__(self, model, suppress_layers, scale=0.0):
        self.hooks = []
        self.suppress_layers = set(suppress_layers)
        self.scale = scale
        
        for name, module in model.named_modules():
            # Hook into down_proj (output of FFN)
            if 'mlp.down_proj' in name:
                layer_num = self._extract_layer(name)
                if layer_num in self.suppress_layers:
                    h = module.register_forward_hook(self._make_hook(layer_num))
                    self.hooks.append(h)
    
    def _extract_layer(self, name):
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i+1 < len(parts):
                return int(parts[i+1])
        return -1
    
    def _make_hook(self, layer_num):
        scale = self.scale
        def hook_fn(module, input, output):
            return output * scale
        return hook_fn
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def run_evaluation(model, tokenizer, items, mode="full_context"):
    """Run evaluation on items, return verdicts."""
    verdicts = []
    for r in items:
        evidence = r.get('evidence', None) if mode == "full_context" else None
        prompt = build_prompt(r['claim'], evidence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
        
        output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        verdict = extract_verdict(output_text)
        verdicts.append(verdict)
    return verdicts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    # Load and bucket data
    with open(args.data_file) as f:
        data = json.load(f)

    buckets = {'faithful_strict': [], 'parametric_override': [], 'context_sensitive_wrong': []}
    for r in data:
        gold = r['gold_verdict'].lower().strip()
        fc = r['full_context_verdict'].lower().strip()
        cb = r['closed_book_verdict'].lower().strip()
        consistent = r['closed_book_consistent']
        
        if fc == gold and cb != fc and consistent:
            buckets['faithful_strict'].append(r)
        elif fc != gold and cb == fc and consistent:
            buckets['parametric_override'].append(r)
        elif fc != gold and cb != fc and consistent:
            buckets['context_sensitive_wrong'].append(r)

    # Limit samples
    for k in buckets:
        buckets[k] = buckets[k][:args.max_samples]

    print("=== Bucket sizes ===")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    # Get total layers
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Define suppression configs
    random.seed(42)
    random_layers = sorted(random.sample(range(num_layers), 5))
    
    configs = {
        'no_suppression': [],
        'narrow_late (31-35)': list(range(31, min(36, num_layers))),
        'broad_late (27-35)': list(range(27, min(36, num_layers))),
        'random_control': random_layers,
    }

    results = {}
    
    for config_name, suppress_layers in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name} (suppressing layers {suppress_layers})")
        print(f"{'='*60}")
        
        # Apply suppression
        suppressor = None
        if suppress_layers:
            suppressor = FFNSuppressor(model, suppress_layers, scale=0.0)
        
        config_results = {}
        for bucket_name, items in buckets.items():
            if not items:
                continue
            print(f"\n  Evaluating {bucket_name} ({len(items)} samples)...")
            verdicts = run_evaluation(model, tokenizer, items, mode="full_context")
            
            # Compare with gold and original
            correct = sum(1 for r, v in zip(items, verdicts) 
                         if v.lower() == r['gold_verdict'].lower())
            changed = sum(1 for r, v in zip(items, verdicts)
                         if v.lower() != r['full_context_verdict'].lower())
            
            config_results[bucket_name] = {
                'accuracy': correct / len(items),
                'changed_from_original': changed / len(items),
                'verdict_dist': dict(Counter(verdicts)),
                'n': len(items)
            }
            print(f"    Accuracy: {correct}/{len(items)} = {correct/len(items):.2%}")
            print(f"    Changed from original: {changed}/{len(items)} = {changed/len(items):.2%}")
            print(f"    Distribution: {dict(Counter(verdicts))}")
        
        results[config_name] = config_results
        
        # Remove hooks
        if suppressor:
            suppressor.remove()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUPPRESSION ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Bucket':<25} {'Accuracy':>10} {'Changed':>10}")
    print("-" * 70)
    for config_name, config_results in results.items():
        for bucket_name, stats in config_results.items():
            print(f"{config_name:<25} {bucket_name:<25} {stats['accuracy']:>10.2%} {stats['changed_from_original']:>10.2%}")

if __name__ == "__main__":
    main()
