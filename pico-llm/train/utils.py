import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data.sequences import (
    NumericSequenceDataset,
    SequenceGenerationConfig,
    pad_collate,
    sample_sequences,
    split_dataset,
)


@dataclass
class TrainingConfig:
    """Lightweight container for experiment hyperparameters."""

    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    max_steps: int = -1
    linear_attention: bool = False
    # Data settings
    seq_min_len: int = 8
    seq_max_len: int = 32
    train_per_type: int = 400
    val_per_type: int = 80
    test_per_type: int = 80
    seed: int = 42

    @property
    def pad_token_id(self) -> int:
        return self.vocab_size - 1


def build_dataloaders(cfg: TrainingConfig, seq_types: Tuple[str, ...]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders for synthetic numeric sequences."""
    train_cfg = SequenceGenerationConfig(
        vocab_size=cfg.vocab_size,
        min_length=cfg.seq_min_len,
        max_length=cfg.seq_max_len,
        seq_types=seq_types,
        seed=cfg.seed,
    )
    val_cfg = SequenceGenerationConfig(
        vocab_size=cfg.vocab_size,
        min_length=cfg.seq_min_len,
        max_length=cfg.seq_max_len,
        seq_types=seq_types,
        seed=cfg.seed + 1,
    )
    test_cfg = SequenceGenerationConfig(
        vocab_size=cfg.vocab_size,
        min_length=cfg.seq_min_len,
        max_length=cfg.seq_max_len,
        seq_types=seq_types,
        seed=cfg.seed + 2,
    )

    counts = {t: cfg.train_per_type for t in seq_types}
    raw = sample_sequences(train_cfg, counts)

    # For validation/test, resample using offset seeds to avoid leakage.
    val_counts = {t: cfg.val_per_type for t in seq_types}
    test_counts = {t: cfg.test_per_type for t in seq_types}
    val_raw = sample_sequences(val_cfg, val_counts)
    test_raw = sample_sequences(test_cfg, test_counts)

    train_ds = NumericSequenceDataset(raw)
    val_ds = NumericSequenceDataset(val_raw)
    test_ds = NumericSequenceDataset(test_raw)

    collate_fn = lambda batch: pad_collate(batch, train_cfg.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
