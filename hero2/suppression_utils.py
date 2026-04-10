"""
Shared FFN suppression utilities for HerO2 pipeline stages.
Uses forward hooks on mlp.down_proj — works with any HF model (Qwen, LLaMA, AWQ, etc.).
No dependency on ParamMute's modified transformers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class FFNSuppressor:
    """Registers forward hooks on mlp.down_proj to scale FFN output."""

    def __init__(self, model, suppress_layers, scale=0.0):
        self.hooks = []
        self.suppress_layers = set(suppress_layers)
        self.scale = scale

        for name, module in model.named_modules():
            if "mlp.down_proj" in name:
                layer_num = self._extract_layer(name)
                if layer_num in self.suppress_layers:
                    h = module.register_forward_hook(self._make_hook(layer_num))
                    self.hooks.append(h)

        print(f"FFN suppression: {len(self.hooks)} hooks on layers "
              f"{sorted(self.suppress_layers)}, scale={self.scale}")

    @staticmethod
    def _extract_layer(name):
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
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


def load_model_with_suppression(model_name, device, suppress_layers=None,
                                 inhibit_ratio=0.0, dtype=torch.bfloat16):
    """Load any HF model with optional FFN suppression via forward hooks."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    suppressor = None
    if suppress_layers:
        suppressor = FFNSuppressor(model, suppress_layers, scale=inhibit_ratio)
    else:
        print("No suppression (baseline mode)")

    return model, tokenizer, suppressor


def add_suppression_args(parser):
    """Add standard suppression CLI args to an argparse parser."""
    parser.add_argument("--suppress_layers", type=int, nargs="+", default=[],
                        help="Layer indices to suppress (empty = no suppression)")
    parser.add_argument("--inhibit_ratio", type=float, default=0.0,
                        help="Scale factor for suppressed layers (0.0 = full suppression)")
    return parser
