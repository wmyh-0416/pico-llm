"""
Neuron activation statistics across sequence families.

Example:
python analysis/activations.py --checkpoint results/baseline/run-*/baseline_checkpoint.pt
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.sequences import SequenceGenerationConfig, sample_sequences
from models.checkpoint import load_checkpoint


def build_batches(seq_types: Tuple[str, ...], cfg, samples_per_type: int, seq_len: int, pad_token_id: int):
    """Return dict type -> tensor batch (batch, seq)"""
    batches: Dict[str, torch.Tensor] = {}
    for seq_type in seq_types:
        gen_cfg = SequenceGenerationConfig(
            vocab_size=cfg.vocab_size,
            min_length=seq_len,
            max_length=seq_len,
            seq_types=(seq_type,),
            seed=123 + hash(seq_type) % 100,
        )
        raw = sample_sequences(gen_cfg, {seq_type: samples_per_type})
        seqs = [torch.tensor(seq, dtype=torch.long) for seq, _ in raw]
        stacked = torch.full((len(seqs), seq_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            stacked[i, : seq.numel()] = seq
        batches[seq_type] = stacked
    return batches


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = tensor * mask
    denom = mask.sum(dim=(0, 1)).clamp(min=1.0)
    return masked.sum(dim=(0, 1)) / denom


def plot_heatmap(matrix: np.ndarray, seq_types: Tuple[str, ...], neuron_ids: List[int], title: str, out_path: Path):
    plt.figure(figsize=(max(6, len(neuron_ids) * 0.35), 3 + 0.2 * len(seq_types)))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(seq_types)), seq_types)
    plt.xticks(range(len(neuron_ids)), neuron_ids, rotation=45, ha="right")
    plt.xlabel("Neuron id (within layer)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Activation selectivity analysis.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq_types", type=str, default="counting,arithmetic,geometric,alternating,random_walk")
    parser.add_argument("--samples_per_type", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=12, help="How many selective neurons to visualize per layer.")
    parser.add_argument("--output_dir", type=str, default="results/activations")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg, model_type = load_checkpoint(Path(args.checkpoint), device)
    seq_types: Tuple[str, ...] = tuple(args.seq_types.split(","))

    batches = build_batches(seq_types, cfg, args.samples_per_type, args.seq_len, cfg.pad_token_id)
    all_stats: Dict[str, Dict] = {}
    out_dir = Path(args.output_dir)

    for seq_type, batch in batches.items():
        batch = batch.to(device)
        with torch.no_grad():
            logits, attn, hidden_states = model(batch, return_hidden=True)
        mask = (batch != cfg.pad_token_id).unsqueeze(-1).float()
        for layer_idx, h in enumerate(hidden_states):
            key = f"layer_{layer_idx}"
            if key not in all_stats:
                all_stats[key] = {}
            mean_vec = masked_mean(h, mask).cpu()
            all_stats[key].setdefault("means", {})[seq_type] = mean_vec

    summary = {}
    for key, entry in all_stats.items():
        means = entry["means"]  # type: ignore
        types = list(means.keys())
        mat = torch.stack([means[t] for t in types])  # (types, neurons)
        # Selectivity: best - second best per neuron.
        top_vals, top_idx = torch.topk(mat, k=2, dim=0)
        scores = top_vals[0] - top_vals[1]
        best_types_idx = top_idx[0]
        scores_np = scores.numpy()
        best_types = [types[i] for i in best_types_idx]
        top_k = min(args.top_k, scores_np.shape[0])
        top_neurons = scores_np.argsort()[-top_k:][::-1]
        top_info = []
        for nid in top_neurons:
            top_info.append(
                {
                    "neuron": int(nid),
                    "pref_type": best_types[nid],
                    "selectivity": float(scores_np[nid]),
                    "mean_activation": float(mat[types.index(best_types[nid]), nid]),
                }
            )
        summary[key] = {
            "avg_mean_per_type": {t: float(means[t].mean()) for t in types},
            "top_selective": top_info,
        }

        # Visualization matrix for top neurons.
        neuron_ids = [int(n) for n in top_neurons.tolist()]
        idx_tensor = torch.tensor(neuron_ids, device=mat.device)
        heat = mat.index_select(1, idx_tensor).detach().cpu().numpy()
        plot_heatmap(heat, tuple(types), neuron_ids, f"{model_type} {key} selective activations", out_dir / f"{key}_heatmap.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "activation_summary.json").open("w", encoding="utf-8") as f:
        import json

        json.dump({"model_type": model_type, "summary": summary}, f, indent=2)
    print(f"Saved activation stats and heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
