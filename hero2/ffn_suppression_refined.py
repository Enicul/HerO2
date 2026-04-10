"""
Refined FFN Suppression: Narrower layers, weaker scales, selectivity metrics.

Tests:
1. No suppression (baseline)
2. Single-layer suppression (34, 33, 32, 31)
3. Narrow pairs (34-35, 32-33)
4. Weaker scales (0.0, 0.25, 0.5, 0.75)
5. Random single-layer control
"""

import json
import argparse
import torch
import numpy as np
import random
from collections import Counter
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
    def __init__(self, model, suppress_layers, scale=0.0):
        self.hooks = []
        self.suppress_layers = set(suppress_layers)
        self.scale = scale
        
        for name, module in model.named_modules():
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

def run_evaluation(model, tokenizer, items):
    verdicts = []
    for r in items:
        evidence = r.get('evidence', None)
        prompt = build_prompt(r['claim'], evidence)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
        output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        verdicts.append(extract_verdict(output_text))
    return verdicts

def compute_selectivity(results, config_name, baseline_name='no_suppression'):
    """Selectivity = change in override - change in faithful."""
    baseline = results.get(baseline_name, {})
    config = results.get(config_name, {})
    
    po_change = config.get('parametric_override', {}).get('changed_from_original', 0)
    fs_change = config.get('faithful_strict', {}).get('changed_from_original', 0)
    
    po_base = baseline.get('parametric_override', {}).get('changed_from_original', 0)
    fs_base = baseline.get('faithful_strict', {}).get('changed_from_original', 0)
    
    # Selectivity: how much more it changes override vs faithful
    selectivity = (po_change - po_base) - (fs_change - fs_base)
    return selectivity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

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

    for k in buckets:
        buckets[k] = buckets[k][:args.max_samples]

    print("=== Bucket sizes ===")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    random.seed(42)
    random_single = [random.randint(0, num_layers - 1)]

    # Configs: (name, layers, scale)
    configs = [
        ('no_suppression', [], 1.0),
        # Single layers
        ('single_L34', [34], 0.0),
        ('single_L33', [33], 0.0),
        ('single_L32', [32], 0.0),
        ('single_L31', [31], 0.0),
        # Narrow pairs
        ('pair_L34-35', [34, 35], 0.0),
        ('pair_L32-33', [32, 33], 0.0),
        # Weaker scales on narrow (31-35)
        ('narrow_scale0.0', list(range(31, min(36, num_layers))), 0.0),
        ('narrow_scale0.25', list(range(31, min(36, num_layers))), 0.25),
        ('narrow_scale0.5', list(range(31, min(36, num_layers))), 0.5),
        ('narrow_scale0.75', list(range(31, min(36, num_layers))), 0.75),
        # Random control
        ('random_single', random_single, 0.0),
    ]

    results = {}
    for config_name, suppress_layers, scale in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name} (layers={suppress_layers}, scale={scale})")
        print(f"{'='*60}")
        
        suppressor = None
        if suppress_layers and scale < 1.0:
            suppressor = FFNSuppressor(model, suppress_layers, scale=scale)
        
        config_results = {}
        for bucket_name, items in buckets.items():
            if not items:
                continue
            print(f"  {bucket_name} ({len(items)} samples)...", end=" ", flush=True)
            verdicts = run_evaluation(model, tokenizer, items)
            
            correct = sum(1 for r, v in zip(items, verdicts)
                         if v.lower() == r['gold_verdict'].lower())
            changed = sum(1 for r, v in zip(items, verdicts)
                         if v.lower() != r['full_context_verdict'].lower())
            
            # Track if override cases flipped to gold
            flipped_to_gold = 0
            if bucket_name == 'parametric_override':
                flipped_to_gold = sum(1 for r, v in zip(items, verdicts)
                                      if v.lower() == r['gold_verdict'].lower()
                                      and r['full_context_verdict'].lower() != r['gold_verdict'].lower())
            
            config_results[bucket_name] = {
                'accuracy': correct / len(items),
                'changed_from_original': changed / len(items),
                'flipped_to_gold': flipped_to_gold / len(items) if bucket_name == 'parametric_override' else None,
                'verdict_dist': dict(Counter(verdicts)),
                'n': len(items)
            }
            print(f"acc={correct/len(items):.0%} chg={changed/len(items):.0%}" +
                  (f" flip={flipped_to_gold/len(items):.0%}" if bucket_name == 'parametric_override' else ""))
        
        results[config_name] = config_results
        if suppressor:
            suppressor.remove()

    # Save
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary with selectivity
    print(f"\n{'='*90}")
    print("REFINED SUPPRESSION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<22} {'Faithful':>10} {'Override':>10} {'CtxWrong':>10} {'PO→Gold':>10} {'Select.':>10}")
    print(f"{'':22} {'acc/chg':>10} {'acc/chg':>10} {'acc/chg':>10} {'flip':>10} {'PO-FS':>10}")
    print("-" * 90)
    
    for config_name, _, _ in configs:
        cr = results[config_name]
        fs = cr.get('faithful_strict', {})
        po = cr.get('parametric_override', {})
        cw = cr.get('context_sensitive_wrong', {})
        
        fs_str = f"{fs.get('accuracy',0):.0%}/{fs.get('changed_from_original',0):.0%}" if fs else "—"
        po_str = f"{po.get('accuracy',0):.0%}/{po.get('changed_from_original',0):.0%}" if po else "—"
        cw_str = f"{cw.get('accuracy',0):.0%}/{cw.get('changed_from_original',0):.0%}" if cw else "—"
        flip_str = f"{po.get('flipped_to_gold',0):.0%}" if po and po.get('flipped_to_gold') is not None else "—"
        
        # Selectivity
        po_chg = po.get('changed_from_original', 0) if po else 0
        fs_chg = fs.get('changed_from_original', 0) if fs else 0
        selectivity = po_chg - fs_chg
        sel_str = f"{selectivity:+.0%}"
        
        print(f"{config_name:<22} {fs_str:>10} {po_str:>10} {cw_str:>10} {flip_str:>10} {sel_str:>10}")

if __name__ == "__main__":
    main()
