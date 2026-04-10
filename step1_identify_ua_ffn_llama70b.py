"""
Step 1: Identify UA-FFN layers for Llama-3.3-70B-Instruct-AWQ
Combined: activation collection + analysis
"""

import argparse
import gc
import json
import os

import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationCollector:
    def __init__(self, model, tokenizer, model_type, num_layers, num_neurons, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.device = device
        self.reset()

    def reset(self):
        self.layer_flag = 0
        self.save_flag = False
        self.data_idx = 0
        self.activation_matrix_total = [None] * self.num_layers
        self.activation_matrix_common = [None] * self.num_layers
        self.activation_matrix = [None] * self.num_layers
        self.max_activate = [None] * self.num_layers
        self.min_activate = [None] * self.num_layers
        self.avg_activate = [None] * self.num_layers

    def mlp_forward_hook(self, layer_idx):
        original_mlp_forward = self.model.model.layers[layer_idx].mlp.forward

        def hooked_forward(x):
            if self.save_flag:
                activations = self.model.model.layers[layer_idx].mlp.act_fn(
                    self.model.model.layers[layer_idx].mlp.gate_proj(x)
                )
                actmean = activations[:, self.data_idx:, :].sum(dim=1, keepdim=True)
                act_bin = (actmean > 0).squeeze().squeeze()

                if self.activation_matrix_total[layer_idx] is None:
                    self.activation_matrix_total[layer_idx] = act_bin.clone()
                    self.activation_matrix_common[layer_idx] = act_bin.clone()
                    self.activation_matrix[layer_idx] = act_bin.int().clone()
                    self.max_activate[layer_idx] = act_bin.sum()
                    self.min_activate[layer_idx] = act_bin.sum()
                    self.avg_activate[layer_idx] = act_bin.sum()
                else:
                    self.activation_matrix_total[layer_idx] |= act_bin
                    self.activation_matrix_common[layer_idx] &= act_bin
                    self.activation_matrix[layer_idx] += act_bin.int()
                    self.max_activate[layer_idx] = max(act_bin.sum(), self.max_activate[layer_idx])
                    self.min_activate[layer_idx] = min(act_bin.sum(), self.min_activate[layer_idx])
                    self.avg_activate[layer_idx] += act_bin.sum()

                down_proj = self.model.model.layers[layer_idx].mlp.down_proj(
                    activations * self.model.model.layers[layer_idx].mlp.up_proj(x)
                )
                self.layer_flag = (self.layer_flag + 1) % self.num_layers
            else:
                down_proj = original_mlp_forward(x)
            return down_proj

        return hooked_forward

    def attach_hooks(self):
        for layer_idx in range(self.num_layers):
            self.model.model.layers[layer_idx].mlp.forward = self.mlp_forward_hook(layer_idx)

    def find_output_start_idx(self, token_ids):
        sublists = {
            "llama31": [128006, 882, 128007, 271],
            "llama3": [128006, 78191, 128007, 271],
        }
        sublist = sublists.get(self.model_type, sublists["llama31"])
        start_indices = [
            i
            for i in range(len(token_ids) - len(sublist) + 1)
            if token_ids[i : i + len(sublist)] == sublist
        ]
        return start_indices[-1] + len(sublist) if start_indices else 0

    def evaluate_per_example(self, in_file_path):
        self.attach_hooks()
        res_data = []
        cnt = 0

        with jsonlines.open(in_file_path) as reader:
            all_datas = list(reader)

        for data in tqdm(all_datas, desc="Collecting activations"):
            cnt += 1
            cur_output = data["pred"]
            cur_input_w_context = data["prompt_w_context"]

            modes = {
                "ww": [
                    {"role": "user", "content": cur_input_w_context},
                    {"role": "assistant", "content": cur_output},
                ]
            }
            per_data_results = {}

            for mode_name, cur_data in modes.items():
                self.reset()
                cur_data_tokens = self.tokenizer.apply_chat_template(cur_data, tokenize=False)
                cur_data_ids = self.tokenizer.apply_chat_template(cur_data, tokenize=True)
                self.data_idx = self.find_output_start_idx(cur_data_ids)
                input_ids = self.tokenizer(cur_data_tokens, return_tensors="pt").to(self.device)
                self.save_flag = True

                with torch.inference_mode():
                    self.model(**input_ids)

                if cnt % 500 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                avg_activations_per_layer = [
                    (m.item() / self.num_neurons) if m is not None else None
                    for m in self.avg_activate
                ]
                per_data_results[mode_name] = avg_activations_per_layer

            data.update({"activations_avg": per_data_results})
            res_data.append(data)

        return res_data


def analyze_ua_ffn(res_data, top_k=10):
    faithful_acts, unfaithful_acts = [], []

    for d in res_data:
        acts = d["activations_avg"]["ww"]
        if d["is_faithful"] == 1:
            faithful_acts.append(acts)
        else:
            unfaithful_acts.append(acts)

    faithful = np.array(faithful_acts)
    unfaithful = np.array(unfaithful_acts)
    n_layers = faithful.shape[1]

    print(f"\nSamples: {len(faithful_acts)} faithful, {len(unfaithful_acts)} unfaithful, {n_layers} layers")

    R_f = faithful.mean(axis=0)
    R_u = unfaithful.mean(axis=0)
    delta = R_u - R_f

    all_acts = np.concatenate([faithful, unfaithful], axis=0)
    mean_R = all_acts.mean()
    rel = delta / (mean_R + 1e-10)
    zsc = (delta - delta.mean()) / (delta.std() + 1e-10)

    def top_k_layers(scores, k):
        return sorted(np.argsort(scores)[::-1][:k].tolist())

    top_raw = top_k_layers(delta, top_k)
    top_rel = top_k_layers(rel, top_k)
    top_zsc = top_k_layers(zsc, top_k)

    print(f"\n{'Method':<12} {'Top-' + str(top_k) + ' Layers':<45} {'Scores'}")
    print("=" * 80)
    for method, layers, scores in [
        ("raw DR", top_raw, delta),
        ("relative", top_rel, rel),
        ("z-score", top_zsc, zsc),
    ]:
        score_str = " ".join(f"{scores[l]:.4f}" for l in layers)
        print(f"{method:<12} {str(layers):<45} {score_str}")

    print(f"\n  {'Layer':<8} {'R_faithful':<14} {'R_unfaithful':<16} {'DR':<12} {'Relative':<12} {'Z-score'}")
    for l in range(n_layers):
        marker = " <--" if l in top_zsc else ""
        print(
            f"  {l:<8} {R_f[l]:<14.5f} {R_u[l]:<16.5f} {delta[l]:<12.5f} {rel[l]:<12.5f} {zsc[l]:.5f}{marker}"
        )

    print(f"\n{'=' * 80}")
    print(f"Suggested suppression command (z-score, Top-{top_k}):")
    layers_str = " ".join(str(l) for l in top_zsc)
    print(f"  --act_inhibit_layer_list {layers_str}")

    return {
        "n_layers": n_layers,
        "top_k": top_k,
        "raw": top_raw,
        "relative": top_rel,
        "zscore": top_zsc,
        "per_layer": [
            {
                "layer": l,
                "R_faithful": float(R_f[l]),
                "R_unfaithful": float(R_u[l]),
                "delta": float(delta[l]),
                "relative_gap": float(rel[l]),
                "zscore_gap": float(zsc[l]),
            }
            for l in range(n_layers)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="/home/aied_test/models/Llama-3.3-70B-Instruct-AWQ",
        type=str,
    )
    parser.add_argument(
        "--in_file_path",
        default="./data/func_data/draw_acctivations_shuf1k.jsonl",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="./results/llama70b_awq_step1",
        type=str,
    )
    parser.add_argument("--model_type", default="llama31", type=str)
    parser.add_argument("--top_k", default=10, type=int, help="Number of UA-FFN layers to select")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size
    print(f"Layers: {num_layers}, Neurons: {num_neurons}")

    collector = ActivationCollector(
        model, tokenizer, args.model_type, num_layers, num_neurons, device
    )
    res_data = collector.evaluate_per_example(args.in_file_path)

    act_path = os.path.join(args.output_dir, "activations_llama33_70b_awq.jsonl")
    with jsonlines.open(act_path, mode="w") as writer:
        for d in res_data:
            writer.write(d)
    print(f"\nActivations saved to {act_path}")

    results = analyze_ua_ffn(res_data, top_k=args.top_k)

    analysis_path = os.path.join(args.output_dir, "ua_ffn_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
