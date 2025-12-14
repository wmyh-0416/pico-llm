"""
Train (or load) baseline softmax vs. linear attention Transformers and compare metrics/interpretability.

Examples:
python experiments/compare_baseline_linear.py --device cpu --epochs 10
python experiments/compare_baseline_linear.py --baseline_ckpt path/to/baseline.pt --linear_ckpt path/to/linear.pt --skip_train
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.file_sequences import build_file_dataloaders, compute_stats, load_numeric_sequences, remap_to_contiguous_ids
from analysis.activations import build_batches, masked_mean, plot_heatmap
from analysis.attention_viz import plot_attention
from analysis.embedding_viz import compute_pca, plot_embeddings
from data.sequences import SequenceGenerationConfig, generate_single_sequence
from models.checkpoint import load_checkpoint, save_checkpoint
from models.transformer_baseline import TransformerConfig, init_baseline_model
from models.transformer_linear import init_linear_model
from train.trainer import evaluate, train_model
from train.utils import TrainingConfig, build_dataloaders, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs linear attention comparison.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--seq_min_len", type=int, default=8)
    parser.add_argument("--seq_max_len", type=int, default=32)
    parser.add_argument("--long_seq_len", type=int, default=48, help="Longer sequences for generalization eval.")
    parser.add_argument("--train_per_type", type=int, default=300)
    parser.add_argument("--val_per_type", type=int, default=60)
    parser.add_argument("--test_per_type", type=int, default=60)
    parser.add_argument("--seq_types", type=str, default="counting,arithmetic,geometric,alternating,random_walk")
    parser.add_argument("--data_file", type=str, default=None, help="Optional path to numeric sequence file (space-separated integers per line).")
    parser.add_argument("--file_val_ratio", type=float, default=0.1, help="Validation split ratio when using --data_file.")
    parser.add_argument("--file_test_ratio", type=float, default=0.1, help="Test split ratio when using --data_file.")
    parser.add_argument("--file_seq_cap", type=int, default=0, help="Optional truncate length for file sequences (0 = no cap).")
    parser.add_argument("--attn_seq_len", type=int, default=48, help="Max sequence length to visualize for attention when using file data.")
    parser.add_argument("--activation_seq_len", type=int, default=96, help="Sequence length cap when computing activation selectivity for file data.")
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--linear_ckpt", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true", help="Only load checkpoints; do not train.")
    parser.add_argument("--output_dir", type=str, default="results/compare")
    return parser.parse_args()


def make_model_cfg(args, data_cfg: TrainingConfig) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_seq_len=args.seq_max_len,
        pad_token_id=data_cfg.pad_token_id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        epochs=args.epochs,
        max_steps=-1,
    )


def prepare_file_dataset(args) -> Tuple[TrainingConfig, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader], Dict]:
    """Load 3seqs-style numeric data and build dataloaders + config."""
    file_path = Path(args.data_file)
    raw_sequences = load_numeric_sequences(file_path)
    remapped_sequences, id_map, sorted_vocab = remap_to_contiguous_ids(raw_sequences)

    if args.file_seq_cap and args.file_seq_cap > 0:
        raw_sequences = [seq[: args.file_seq_cap] for seq in raw_sequences]
        remapped_sequences = [seq[: args.file_seq_cap] for seq in remapped_sequences]

    vocab_size = len(sorted_vocab) + 1  # +1 for pad
    pad_token_id = vocab_size - 1
    stats = compute_stats(remapped_sequences, vocab_size, pad_token_id, max_token=sorted_vocab[-1])

    loaders = build_file_dataloaders(
        remapped_sequences,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        val_ratio=args.file_val_ratio,
        test_ratio=args.file_test_ratio,
        seed=42,
    )

    data_cfg = TrainingConfig(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seq_min_len=stats.min_len,
        seq_max_len=stats.max_len,
        train_per_type=0,
        val_per_type=0,
        test_per_type=0,
        seed=42,
    )
    return data_cfg, loaders, {"raw_sequences": raw_sequences, "remapped_sequences": remapped_sequences, "id_map": id_map, "stats": stats, "sorted_vocab": sorted_vocab}


def maybe_train_model(
    model_type: str,
    args,
    data_cfg: TrainingConfig,
    seq_types: Tuple[str, ...],
    out_root: Path,
    ckpt_override: str,
    loaders: Optional[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = None,
) -> Tuple[torch.nn.Module, TransformerConfig, Dict, Dict, Path]:
    device = torch.device(args.device)
    if ckpt_override:
        model, cfg, _ = load_checkpoint(Path(ckpt_override), device)
        cfg.device = args.device
        history = None
        metrics = {}
        return model, cfg, history, metrics, Path(ckpt_override)

    model_cfg = make_model_cfg(args, data_cfg)
    if model_type == "baseline":
        model = init_baseline_model(model_cfg).to(device)
    else:
        model = init_linear_model(model_cfg).to(device)

    if loaders is None:
        train_loader, val_loader, test_loader = build_dataloaders(data_cfg, seq_types)
    else:
        train_loader, val_loader, test_loader = loaders
    run_dir = out_root / model_type
    history = train_model(model, model_cfg, model_type, train_loader, val_loader, run_dir)
    test_loss, test_acc = evaluate(model, test_loader, model_cfg)
    metrics = {"history": history, "test_loss": test_loss, "test_acc": test_acc}
    save_checkpoint(model, model_cfg, model_type, run_dir / f"{model_type}_checkpoint.pt")
    return model, model_cfg, history, metrics, run_dir / f"{model_type}_checkpoint.pt"


def combined_loss_plot(b_hist: Dict, l_hist: Dict, out_path: Path):
    plt.figure()
    if b_hist:
        plt.plot(b_hist["val_loss"], label="baseline val")
    if l_hist:
        plt.plot(l_hist["val_loss"], label="linear val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def run_embedding_viz(model, cfg, model_type: str, out_path: Path):
    emb = model.token_emb.weight.detach().cpu().numpy()
    tokens = np.arange(emb.shape[0])
    proj, _ = compute_pca(emb, n_components=2)
    plot_embeddings(proj, tokens, cfg.pad_token_id, out_path, f"{model_type} embeddings")


def run_attention_viz(
    model,
    cfg,
    model_type: str,
    seq_type: str,
    length: int,
    out_dir: Path,
    tokens: Optional[List[int]] = None,
    display_tokens: Optional[List[int]] = None,
):
    import random

    if tokens is None:
        gen_cfg = SequenceGenerationConfig(
            vocab_size=cfg.vocab_size,
            min_length=length,
            max_length=length,
            seq_types=(seq_type,),
            seed=7,
        )
        seq = generate_single_sequence(seq_type, gen_cfg, random.Random(7))
        tokens = seq
        display_tokens = seq
    else:
        display_tokens = display_tokens or tokens

    token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        logits, attn_maps, _ = model(token_tensor, return_attn=True)
    for layer_idx, attn in enumerate(attn_maps):
        for head_idx in range(attn.size(1)):
            out_path = out_dir / f"{model_type}_layer{layer_idx}_head{head_idx}.png"
            plot_attention(attn, display_tokens, layer_idx, head_idx, out_path)


def build_file_batches(sequences: List[List[int]], pad_token_id: int, seq_len: int, samples: int = 16, seed: int = 0) -> Dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(sequences), generator=rng)[: min(samples, len(sequences))]
    batch = torch.full((indices.numel(), seq_len), pad_token_id, dtype=torch.long)
    for i, idx in enumerate(indices):
        seq = sequences[int(idx)]
        truncated = seq[:seq_len]
        batch[i, : len(truncated)] = torch.tensor(truncated, dtype=torch.long)
    return {"file": batch}


def run_activation_viz(
    model,
    cfg,
    model_type: str,
    seq_types: Tuple[str, ...],
    out_dir: Path,
    batches: Optional[Dict[str, torch.Tensor]] = None,
):
    if batches is None:
        batches = build_batches(seq_types, cfg, samples_per_type=16, seq_len=min(cfg.max_seq_len, 24), pad_token_id=cfg.pad_token_id)
    summary = {}
    for seq_type, batch in batches.items():
        batch = batch.to(cfg.device)
        with torch.no_grad():
            logits, attn, hidden_states = model(batch, return_hidden=True)
        mask = (batch != cfg.pad_token_id).unsqueeze(-1).float()
        for layer_idx, h in enumerate(hidden_states):
            key = f"layer_{layer_idx}"
            if key not in summary:
                summary[key] = {}
            summary[key].setdefault(seq_type, masked_mean(h, mask).cpu())

    for key, per_type in summary.items():
        types = list(per_type.keys())
        mat = torch.stack([per_type[t] for t in types])  # (types, neurons)
        # top neurons (handle single-type case)
        if mat.size(0) >= 2:
            top_vals, _ = torch.topk(mat, k=2, dim=0)
            scores = (top_vals[0] - top_vals[1]).numpy()
        else:
            scores = mat[0].numpy()
        top_neurons = scores.argsort()[-10:][::-1]
        neuron_ids = [int(n) for n in top_neurons.tolist()]
        idx_tensor = torch.tensor(neuron_ids, device=mat.device)
        heat = mat.index_select(1, idx_tensor).detach().cpu().numpy()
        plot_heatmap(heat, tuple(types), neuron_ids, f"{model_type} {key} selective", out_dir / f"{model_type}_{key}_heatmap.png")


def evaluate_long_range(model, cfg, seq_types: Tuple[str, ...], seq_len: int) -> float:
    temp_cfg = TrainingConfig(
        vocab_size=cfg.vocab_size,
        seq_min_len=seq_len,
        seq_max_len=seq_len,
        train_per_type=60,
        val_per_type=20,
        test_per_type=20,
        batch_size=32,
        seed=1234,
    )
    loaders = build_dataloaders(temp_cfg, seq_types)
    _, _, test_loader = loaders
    loss, acc = evaluate(model, test_loader, cfg)
    return loss, acc


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_root = Path(args.output_dir) / f"run-{timestamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    file_meta: Optional[Dict] = None
    loaders: Optional[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = None
    if args.data_file:
        data_cfg, loaders, file_meta = prepare_file_dataset(args)
        args.vocab_size = data_cfg.vocab_size
        args.seq_min_len = data_cfg.seq_min_len
        args.seq_max_len = data_cfg.seq_max_len
        seq_types: Tuple[str, ...] = ("file",)
    else:
        seq_types = tuple(args.seq_types.split(","))
        data_cfg = TrainingConfig(
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seq_min_len=args.seq_min_len,
            seq_max_len=args.seq_max_len,
            train_per_type=args.train_per_type,
            val_per_type=args.val_per_type,
            test_per_type=args.test_per_type,
        )

    baseline_model, b_cfg, b_hist, b_metrics, b_ckpt = maybe_train_model(
        "baseline", args, data_cfg, seq_types, out_root, args.baseline_ckpt if args.skip_train else None, loaders=loaders
    )
    linear_model, l_cfg, l_hist, l_metrics, l_ckpt = maybe_train_model(
        "linear", args, data_cfg, seq_types, out_root, args.linear_ckpt if args.skip_train else None, loaders=loaders
    )

    combined_loss_plot(b_hist or {}, l_hist or {}, out_root / "loss_compare.png")

    # Long-range eval
    long_range = {}
    if not args.data_file:
        long_loss_b, long_acc_b = evaluate_long_range(baseline_model, b_cfg, seq_types, args.long_seq_len)
        long_loss_l, long_acc_l = evaluate_long_range(linear_model, l_cfg, seq_types, args.long_seq_len)
        long_range = {
            "baseline": {"loss": long_loss_b, "acc": long_acc_b},
            "linear": {"loss": long_loss_l, "acc": long_acc_l},
        }

    # Interpretability snapshots
    run_embedding_viz(baseline_model, b_cfg, "baseline", out_root / "baseline_embedding_pca.png")
    run_embedding_viz(linear_model, l_cfg, "linear", out_root / "linear_embedding_pca.png")
    attn_dir = out_root / "attention"
    if file_meta:
        attn_tokens = file_meta["remapped_sequences"][0][: args.attn_seq_len]
        attn_display = file_meta["raw_sequences"][0][: args.attn_seq_len]
        run_attention_viz(baseline_model, b_cfg, "baseline", "file", len(attn_tokens), attn_dir, tokens=attn_tokens, display_tokens=attn_display)
        run_attention_viz(linear_model, l_cfg, "linear", "file", len(attn_tokens), attn_dir, tokens=attn_tokens, display_tokens=attn_display)
    else:
        run_attention_viz(baseline_model, b_cfg, "baseline", "arithmetic", 12, attn_dir)
        run_attention_viz(linear_model, l_cfg, "linear", "arithmetic", 12, attn_dir)
    act_dir = out_root / "activations"
    if file_meta:
        act_seq_len = min(args.activation_seq_len, data_cfg.seq_max_len)
        batches = build_file_batches(file_meta["remapped_sequences"], data_cfg.pad_token_id, seq_len=act_seq_len, samples=16)
        run_activation_viz(baseline_model, b_cfg, "baseline", ("file",), act_dir, batches=batches)
        run_activation_viz(linear_model, l_cfg, "linear", ("file",), act_dir, batches=batches)
    else:
        run_activation_viz(baseline_model, b_cfg, "baseline", seq_types, act_dir)
        run_activation_viz(linear_model, l_cfg, "linear", seq_types, act_dir)

    dataset_info = None
    if file_meta:
        stats = file_meta["stats"]
        dataset_info = {
            "source": str(Path(args.data_file)),
            "num_sequences": stats.num_sequences,
            "num_unique_tokens": stats.num_unique_tokens,
            "max_token": stats.max_token,
            "min_len": stats.min_len,
            "max_len": stats.max_len,
            "vocab_size": stats.vocab_size,
        }
    summary = {
        "baseline": b_metrics,
        "linear": l_metrics,
        "long_range": long_range,
        "dataset": dataset_info,
        "artifacts": {
            "baseline_ckpt": str(b_ckpt),
            "linear_ckpt": str(l_ckpt),
            "loss_plot": str(out_root / "loss_compare.png"),
        },
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison complete. Results under {out_root}")


if __name__ == "__main__":
    main()
