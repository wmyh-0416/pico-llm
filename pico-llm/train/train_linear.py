import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from models.transformer_baseline import TransformerConfig
from models.transformer_linear import init_linear_model
from train.trainer import evaluate, train_model
from train.utils import TrainingConfig, build_dataloaders, save_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear-attention Transformer on synthetic numeric sequences.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--train_per_type", type=int, default=400)
    parser.add_argument("--val_per_type", type=int, default=80)
    parser.add_argument("--test_per_type", type=int, default=80)
    parser.add_argument("--seq_types", type=str, default="counting,arithmetic,geometric,alternating,random_walk")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="results/linear")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    seq_types: Tuple[str, ...] = tuple(args.seq_types.split(","))
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
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, seq_types)

    model_cfg = TransformerConfig(
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
        device=str(device),
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    model = init_linear_model(model_cfg).to(device)

    out_dir = Path(args.output_dir) / f"run-{timestamp()}"
    history = train_model(model, model_cfg, "linear", train_loader, val_loader, out_dir)
    test_loss, test_acc = evaluate(model, test_loader, model_cfg)
    metrics = {
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "config": vars(args),
        "model_cfg": model_cfg.__dict__,
    }
    save_json(metrics, out_dir / "metrics.json")
    print(f"[linear] test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
