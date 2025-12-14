import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


@dataclass
class SequenceGenerationConfig:
    """Configuration for synthetic numeric sequence generation."""

    vocab_size: int = 128
    min_length: int = 8
    max_length: int = 32
    # Sequence families we sample from by default.
    seq_types: Tuple[str, ...] = ("counting", "arithmetic", "geometric")
    seed: int = 42

    @property
    def pad_token_id(self) -> int:
        # Reserve the last token id for padding so we never collide with real tokens.
        return self.vocab_size - 1

    @property
    def max_token_id(self) -> int:
        return self.vocab_size - 2


def _sample_length(cfg: SequenceGenerationConfig, rng: random.Random) -> int:
    return rng.randint(cfg.min_length, cfg.max_length)


def _clamp_token(value: int, cfg: SequenceGenerationConfig) -> int:
    return max(0, min(value, cfg.max_token_id))


def generate_single_sequence(seq_type: str, cfg: SequenceGenerationConfig, rng: random.Random) -> List[int]:
    """Generate one numeric sequence of the requested type."""
    length = _sample_length(cfg, rng)

    if seq_type == "counting":
        start = rng.randint(0, 3)
        step = 1
        seq = [start + step * i for i in range(length)]
    elif seq_type == "arithmetic":
        start = rng.randint(0, 8)
        step = rng.randint(1, 5)
        seq = [start + step * i for i in range(length)]
    elif seq_type == "geometric":
        start = rng.randint(1, 3)
        ratio = rng.randint(2, 3)
        seq = [start * (ratio ** i) for i in range(length)]
    elif seq_type == "alternating":
        a = rng.randint(0, cfg.max_token_id // 4 + 1)
        b = a + 1
        seq = [a if i % 2 == 0 else b for i in range(length)]
    elif seq_type == "random_walk":
        current = rng.randint(0, 4)
        seq = []
        for _ in range(length):
            seq.append(current)
            current = max(0, current + rng.choice([-1, 0, 1, 2]))
    else:
        raise ValueError(f"Unknown sequence type: {seq_type}")

    # Clamp values to stay within the vocabulary range (reserve last id for padding).
    seq = [_clamp_token(x, cfg) for x in seq]
    return seq


def sample_sequences(
    cfg: SequenceGenerationConfig,
    num_per_type: Dict[str, int],
) -> List[Tuple[List[int], str]]:
    """Generate a labeled pool of sequences for all requested types."""
    rng = random.Random(cfg.seed)
    all_types = num_per_type.keys()
    samples: List[Tuple[List[int], str]] = []
    for seq_type in all_types:
        count = num_per_type[seq_type]
        for _ in range(count):
            seq = generate_single_sequence(seq_type, cfg, rng)
            samples.append((seq, seq_type))

    rng.shuffle(samples)
    return samples


def split_dataset(
    data: Sequence[Tuple[List[int], str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Tuple[List[int], str]], List[Tuple[List[int], str]], List[Tuple[List[int], str]]]:
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_split = data[:n_train]
    val_split = data[n_train : n_train + n_val]
    test_split = data[n_train + n_val :]
    return train_split, val_split, test_split


class NumericSequenceDataset(torch.utils.data.Dataset):
    """Torch dataset that returns padded numeric sequences and their type label."""

    def __init__(self, data: Iterable[Tuple[List[int], str]]):
        super().__init__()
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        seq, label = self.data[idx]
        return torch.tensor(seq, dtype=torch.long), label


def pad_collate(batch: List[Tuple[torch.Tensor, str]], pad_token_id: int) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Pad a list of 1D tensors to the max length in the batch."""
    seqs, labels = zip(*batch)
    max_len = max(seq.size(0) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    lengths = torch.tensor([seq.size(0) for seq in seqs], dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : seq.size(0)] = seq
    return padded, list(labels), lengths
