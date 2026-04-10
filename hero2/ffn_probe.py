"""
FFN Activation Probing: Compare activation patterns across buckets.
Records per-layer FFN stats on verdict output tokens.
"""

import json
import argparse
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_ffn_hooks(model):
    """Register forward hooks on all FFN layers to capture activations."""
    activations = {}
    hooks = []
    
    # Find FFN layers - adapt for Qwen architecture
    for name, module in model.named_modules():
        if any(ffn_name in name for ffn_name in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']):
            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    activations[layer_name] = output.detach()
                return hook_fn
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
    
    return activations, hooks

def compute_ffn_stats(activation):
    """Compute activation statistics for a single layer."""
    # Take last token (verdict token) activations
    if len(activation.shape) == 3:
        act = activation[0, -1, :]  # [hidden_dim]
    else:
        act = activation[-1, :]
    
    act_np = act.float().cpu().numpy()
    
    return {
        'activation_ratio': float(np.mean(act_np != 0)),
        'mean_magnitude': float(np.mean(np.abs(act_np))),
        'l2_norm': float(np.linalg.norm(act_np)),
        'max_activation': float(np.max(np.abs(act_np))),
        'mean_value': float(np.mean(act_np)),
        'std_value': float(np.std(act_np))
    }

def build_prompt(claim, evidence=None):
    """Build prompt for probing - with or without evidence."""
    if evidence:
        evidence_text = "\n".join([
            f"Q: {e['question']}\nA: {e['answer']}" 
            for e in evidence[:5]  # Use top 5 evidence
        ])
        prompt = f"""You are a fact-checking expert. Based on the provided evidence, classify the following claim.

Claim: {claim}

Evidence:
{evidence_text}

verdict: """
    else:
        prompt = f"""You are a fact-checking expert. Based ONLY on your own knowledge, classify the following claim.

Claim: {claim}

verdict: """
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="humane-lab/Qwen3-32B-AWQ-HerO")
    parser.add_argument("--max_samples_per_bucket", type=int, default=50)
    parser.add_argument("--mode", type=str, default="full_context", choices=["full_context", "closed_book"])
    args = parser.parse_args()

    # Load data and assign buckets
    with open(args.data_file) as f:
        data = json.load(f)

    buckets = {'faithful_strict': [], 'parametric_override': [], 'context_sensitive_wrong': [], 'ambiguous': []}
    
    for r in data:
        gold = r['gold_verdict'].lower().strip()
        fc = r['full_context_verdict'].lower().strip()
        cb = r['closed_book_verdict'].lower().strip()
        consistent = r['closed_book_consistent']
        
        ob_correct = (fc == gold)
        cb_diff_fc = (cb != fc)
        cb_matches_fc = (cb == fc)

        if ob_correct and cb_diff_fc and consistent:
            buckets['faithful_strict'].append(r)
        elif not ob_correct and cb_matches_fc and consistent:
            buckets['parametric_override'].append(r)
        elif not ob_correct and cb_diff_fc and consistent:
            buckets['context_sensitive_wrong'].append(r)
        else:
            buckets['ambiguous'].append(r)

    print("=== Bucket sizes ===")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    # Limit samples per bucket
    for k in buckets:
        if len(buckets[k]) > args.max_samples_per_bucket:
            buckets[k] = buckets[k][:args.max_samples_per_bucket]

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

    # Register hooks
    activations, hooks = get_ffn_hooks(model)
    
    # Get layer names (sorted)
    layer_names = sorted(activations.keys()) if activations else []
    
    # Probe each bucket
    results = {}
    for bucket_name in ['faithful_strict', 'parametric_override', 'context_sensitive_wrong']:
        items = buckets[bucket_name]
        if not items:
            print(f"\nSkipping {bucket_name} - no items")
            continue
            
        print(f"\nProbing {bucket_name} ({len(items)} samples, mode={args.mode})...")
        bucket_stats = defaultdict(lambda: defaultdict(list))
        
        for idx, r in enumerate(items):
            claim = r['claim']
            evidence = r.get('evidence', None) if args.mode == 'full_context' else None
            prompt = build_prompt(claim, evidence)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            
            activations.clear()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Record stats for each layer
            for layer_name, act in activations.items():
                stats = compute_ffn_stats(act)
                for stat_name, stat_val in stats.items():
                    bucket_stats[layer_name][stat_name].append(stat_val)
            
            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(items)}")
        
        # Aggregate stats per layer
        results[bucket_name] = {}
        for layer_name in bucket_stats:
            results[bucket_name][layer_name] = {}
            for stat_name in bucket_stats[layer_name]:
                values = bucket_stats[layer_name][stat_name]
                results[bucket_name][layer_name][stat_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'n': len(values)
                }

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Print summary comparison
    print("\n=== Layer-wise Comparison (mean L2 norm) ===")
    print(f"{'Layer':<50} {'Faithful':>10} {'Override':>10} {'CtxWrong':>10} {'F-P diff':>10}")
    print("-" * 90)
    
    all_layers = set()
    for b in results:
        all_layers.update(results[b].keys())
    
    diffs = []
    for layer in sorted(all_layers):
        f_val = results.get('faithful_strict', {}).get(layer, {}).get('l2_norm', {}).get('mean', 0)
        p_val = results.get('parametric_override', {}).get(layer, {}).get('l2_norm', {}).get('mean', 0)
        c_val = results.get('context_sensitive_wrong', {}).get(layer, {}).get('l2_norm', {}).get('mean', 0)
        diff = f_val - p_val
        diffs.append((layer, f_val, p_val, c_val, diff))
    
    # Sort by absolute difference
    diffs.sort(key=lambda x: abs(x[4]), reverse=True)
    for layer, f_val, p_val, c_val, diff in diffs[:20]:
        print(f"{layer:<50} {f_val:>10.4f} {p_val:>10.4f} {c_val:>10.4f} {diff:>+10.4f}")

if __name__ == "__main__":
    main()
