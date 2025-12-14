import random
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

try:
    import tiktoken
except ImportError:
    tiktoken = None

from data.sequences import NumericSequenceDataset, pad_collate, split_dataset


def load_tinystories_tokens(
    limit: int = 5000,
    max_length: int = 128,
    seed: int = 0,
    tokenizer=None,
) -> Tuple[List[List[int]], int, int, dict]:
    """
    Load TinyStories from HuggingFace, shuffle, take a subset, and encode with GPT-2 tokenizer.

    Returns:
        sequences: list of token id lists (truncated to max_length, length>=4)
        vocab_size: tokenizer vocab + 1 (we add a pad token)
        pad_token_id: id of the added pad token
        stats: summary dict
    """
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds = ds.shuffle(seed=seed)
    if limit > 0:
        ds = ds.select(range(limit))

    if tokenizer is not None:
        enc = tokenizer
        if enc.pad_token is None:
            enc.pad_token = enc.eos_token
        pad_token_id = enc.pad_token_id
        vocab_size = len(enc)
        encode_fn = lambda text: enc.encode(text, add_special_tokens=False)
    else:
        if tiktoken is None:
            raise ImportError("tiktoken is required when no HF tokenizer is provided.")
        enc = tiktoken.get_encoding("gpt2")
        base_vocab = enc.n_vocab
        pad_token_id = base_vocab
        vocab_size = base_vocab + 1
        encode_fn = lambda text: enc.encode(text)

    sequences: List[List[int]] = []
    for sample in ds:
        tokens = encode_fn(sample["text"])
        tokens = tokens[:max_length]
        if len(tokens) >= 4:
            sequences.append(tokens)

    stats = {
        "num_sequences": len(sequences),
        "base_vocab_size": base_vocab,
        "pad_token_id": pad_token_id,
        "max_length": max_length,
        "limit": limit,
    }
    return sequences, vocab_size, pad_token_id, stats


def build_tinystories_dataloaders(
    sequences: List[List[int]],
    pad_token_id: int,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split TinyStories token sequences and return train/val/test loaders."""
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    labeled = [(seq, "tinystories") for seq in sequences]
    train_split, val_split, test_split = split_dataset(labeled, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    collate = lambda batch: pad_collate(batch, pad_token_id)
    train_loader = DataLoader(NumericSequenceDataset(train_split), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(NumericSequenceDataset(val_split), batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(NumericSequenceDataset(test_split), batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader
