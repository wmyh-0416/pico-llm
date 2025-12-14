"""
Visualize token embeddings with PCA and simple color coding (parity / token id).

Example:
python analysis/embedding_viz.py --checkpoint results/baseline/run-*/baseline_checkpoint.pt
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.checkpoint import load_checkpoint


def compute_pca(x: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal PCA via SVD (no sklearn dependency)."""
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(x_centered, full_matrices=False)
    components = vh[:n_components]
    proj = np.dot(x_centered, components.T)
    return proj, components


def plot_embeddings(proj: np.ndarray, tokens: np.ndarray, pad_token_id: int, out_path: Path, title: str):
    colors = ["blue" if t % 2 == 0 else "orange" for t in tokens]
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, s=24, alpha=0.8, edgecolors="k", linewidths=0.4)
    for t, (x, y) in zip(tokens, proj):
        if t == pad_token_id:
            continue
        plt.text(x, y, str(int(t)), fontsize=6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Embedding PCA visualization for pico-llm numeric Transformer.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/embedding_pca.png")
    parser.add_argument("--max_tokens", type=int, default=None, help="Limit tokens plotted; default uses full vocab except pad.")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg, model_type = load_checkpoint(Path(args.checkpoint), device)
    embeddings = model.token_emb.weight.detach().cpu().numpy()

    num_tokens = embeddings.shape[0]
    pad_id = cfg.pad_token_id
    limit = args.max_tokens if args.max_tokens is not None else num_tokens
    idx = np.arange(min(limit, num_tokens))
    proj, _ = compute_pca(embeddings[idx], n_components=2)
    plot_embeddings(proj, idx, pad_id, Path(args.output), title=f"{model_type} embedding PCA")
    print(f"Saved embedding PCA to {args.output}")


if __name__ == "__main__":
    main()
