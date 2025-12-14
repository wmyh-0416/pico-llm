"""
Utilities for loading numeric sequences from a plain-text file (space-separated integers per line)
and building dataloaders that match the synthetic-sequence pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.sequences import NumericSequenceDataset, pad_collate, split_dataset


@dataclass
class FileSequenceStats:
    vocab_size: int
    pad_token_id: int
    max_token: int
    min_len: int
    max_len: int
    num_sequences: int
    num_unique_tokens: int


def load_numeric_sequences(path: Path) -> List[List[int]]:
    """Read whitespace-separated integer sequences from a file (one sequence per line)."""
    sequences: List[List[int]] = []
    with path.open() as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            sequences.append([int(t) for t in tokens])
    return sequences


def remap_to_contiguous_ids(sequences: Iterable[List[int]]) -> Tuple[List[List[int]], Dict[int, int], List[int]]:
    """
    Map arbitrary integer tokens to contiguous ids starting at 0.
    Returns remapped sequences, the original->id map, and the sorted vocabulary.
    """
    unique_tokens = {}
    for seq in sequences:
        for tok in seq:
            unique_tokens[tok] = None
    sorted_vocab = sorted(unique_tokens.keys())
    id_map: Dict[int, int] = {tok: i for i, tok in enumerate(sorted_vocab)}
    remapped = [[id_map[tok] for tok in seq] for seq in sequences]
    return remapped, id_map, sorted_vocab


def build_file_dataloaders(
    sequences: List[List[int]],
    pad_token_id: int,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split remapped sequences and return train/val/test dataloaders."""
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    labeled = [(seq, "file") for seq in sequences]
    train_split, val_split, test_split = split_dataset(labeled, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    collate = lambda batch: pad_collate(batch, pad_token_id)
    train_loader = DataLoader(NumericSequenceDataset(train_split), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(NumericSequenceDataset(val_split), batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(NumericSequenceDataset(test_split), batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


def compute_stats(remapped_sequences: List[List[int]], vocab_size: int, pad_token_id: int, max_token: int) -> FileSequenceStats:
    lengths = [len(seq) for seq in remapped_sequences]
    return FileSequenceStats(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        max_token=max_token,
        min_len=min(lengths),
        max_len=max(lengths),
        num_sequences=len(remapped_sequences),
        num_unique_tokens=vocab_size - 1,
    )
