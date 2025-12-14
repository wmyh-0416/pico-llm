"""
Visualize attention weights (softmax or linear-kernel approximations) for a given input sequence.

Example:
python analysis/attention_viz.py --checkpoint results/baseline/run-*/baseline_checkpoint.pt --seq_type arithmetic --length 10
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.sequences import SequenceGenerationConfig, generate_single_sequence
from models.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attention heatmaps for a numeric sequence.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq_type", type=str, default="counting", help="counting/arithmetic/geometric/alternating/random_walk")
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--custom_seq", type=str, default=None, help="Optional space-separated integers to override generator.")
    parser.add_argument("--output_dir", type=str, default="results/attention")
    return parser.parse_args()


def build_sequence(args, cfg) -> List[int]:
    if args.custom_seq:
        return [int(x) for x in args.custom_seq.strip().split()]
    gen_cfg = SequenceGenerationConfig(
        vocab_size=cfg.vocab_size,
        min_length=args.length,
        max_length=args.length,
        seq_types=(args.seq_type,),
        seed=0,
    )
    import random

    return generate_single_sequence(args.seq_type, gen_cfg, random.Random(0))


def plot_attention(attn: torch.Tensor, tokens: List[int], layer_idx: int, head_idx: int, out_path: Path):
    """
    attn: (batch, heads, seq, seq) already normalized for both baseline and linear variant.
    """
    attn = attn[0, head_idx].cpu()
    plt.figure(figsize=(4, 3.5))
    plt.imshow(attn, cmap="magma", vmin=0.0, vmax=attn.max().item() + 1e-6)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = list(range(len(tokens)))
    plt.xticks(ticks, tokens, rotation=45, ha="right")
    plt.yticks(ticks, tokens)
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Layer {layer_idx} Head {head_idx}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cpu")
    model, cfg, model_type = load_checkpoint(Path(args.checkpoint), device)

    seq = build_sequence(args, cfg)
    tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)
    with torch.no_grad():
        logits, attn_maps, _ = model(tokens, return_attn=True)
    if not attn_maps:
        raise RuntimeError("No attention maps returned; ensure the model supports return_attn=True.")

    out_dir = Path(args.output_dir)
    for layer_idx, attn in enumerate(attn_maps):
        num_heads = attn.size(1)
        for head_idx in range(num_heads):
            out_path = out_dir / f"{model_type}_layer{layer_idx}_head{head_idx}.png"
            plot_attention(attn, seq, layer_idx, head_idx, out_path)
    print(f"Saved attention heatmaps under {out_dir}")


if __name__ == "__main__":
    main()
