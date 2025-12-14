import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from data.sequences import SequenceGenerationConfig, generate_single_sequence, split_dataset


@dataclass
class PreferenceSample:
    """Container for one prompt with chosen/rejected continuations."""

    prompt: List[int]
    chosen: List[int]
    rejected: List[int]
    seq_type: str


def _clamp_token(value: int, cfg: SequenceGenerationConfig) -> int:
    return max(0, min(value, cfg.max_token_id))


def _corrupt_continuation(target: List[int], cfg: SequenceGenerationConfig, rng: random.Random) -> List[int]:
    """Create a noised continuation as the rejected sample."""
    if not target:
        return [rng.randint(0, cfg.max_token_id)]

    corrupted: List[int] = []
    for tok in target:
        if rng.random() < 0.6:
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            corrupted.append(_clamp_token(tok + delta, cfg))
        else:
            corrupted.append(rng.randint(0, cfg.max_token_id))

    # Sometimes drop the last token to create shorter (worse) endings.
    if rng.random() < 0.3 and len(corrupted) > 1:
        corrupted = corrupted[:-1]

    # Ensure we do not accidentally match the chosen continuation.
    if corrupted == target:
        idx = rng.randrange(len(corrupted))
        corrupted[idx] = _clamp_token(corrupted[idx] + 1, cfg)
    return corrupted


def _make_preference_samples(
    cfg: SequenceGenerationConfig,
    seq_types: Sequence[str],
    pairs_per_type: int,
    prompt_frac_range: Tuple[float, float] = (0.4, 0.7),
) -> List[PreferenceSample]:
    rng = random.Random(cfg.seed)
    samples: List[PreferenceSample] = []
    low, high = prompt_frac_range
    for seq_type in seq_types:
        for _ in range(pairs_per_type):
            full_seq = generate_single_sequence(seq_type, cfg, rng)
            if len(full_seq) < 4:
                continue
            frac = rng.uniform(low, high)
            split = max(2, min(len(full_seq) - 2, int(len(full_seq) * frac)))
            prompt = full_seq[:split]
            chosen = full_seq[split:]
            rejected = _corrupt_continuation(chosen, cfg, rng)
            samples.append(PreferenceSample(prompt, chosen, rejected, seq_type))
    rng.shuffle(samples)
    return samples


class PreferenceDataset(torch.utils.data.Dataset):
    """Returns (prompt, chosen, rejected, label) as tensors."""

    def __init__(self, data: Iterable[PreferenceSample]):
        super().__init__()
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        return (
            torch.tensor(sample.prompt, dtype=torch.long),
            torch.tensor(sample.chosen, dtype=torch.long),
            torch.tensor(sample.rejected, dtype=torch.long),
            sample.seq_type,
        )


def preference_collate(batch, pad_token_id: int):
    prompts, chosens, rejecteds, labels = zip(*batch)
    bsz = len(prompts)
    max_prompt = max(p.numel() for p in prompts)
    max_chosen = max(c.numel() for c in chosens)
    max_rejected = max(r.numel() for r in rejecteds)

    prompt_tensor = torch.full((bsz, max_prompt), pad_token_id, dtype=torch.long)
    chosen_tensor = torch.full((bsz, max_chosen), pad_token_id, dtype=torch.long)
    rejected_tensor = torch.full((bsz, max_rejected), pad_token_id, dtype=torch.long)

    prompt_lens = torch.zeros(bsz, dtype=torch.long)
    chosen_lens = torch.zeros(bsz, dtype=torch.long)
    rejected_lens = torch.zeros(bsz, dtype=torch.long)

    for i, (p, c, r) in enumerate(zip(prompts, chosens, rejecteds)):
        prompt_len = p.numel()
        chosen_len = c.numel()
        rejected_len = r.numel()
        prompt_tensor[i, :prompt_len] = p
        chosen_tensor[i, :chosen_len] = c
        rejected_tensor[i, :rejected_len] = r
        prompt_lens[i] = prompt_len
        chosen_lens[i] = chosen_len
        rejected_lens[i] = rejected_len

    return {
        "prompt": prompt_tensor,
        "chosen": chosen_tensor,
        "rejected": rejected_tensor,
        "prompt_lens": prompt_lens,
        "chosen_lens": chosen_lens,
        "rejected_lens": rejected_lens,
        "labels": list(labels),
    }


def build_preference_dataloaders(
    vocab_size: int,
    seq_min_len: int,
    seq_max_len: int,
    seq_types: Tuple[str, ...],
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    batch_size: int,
    seed: int = 0,
    prompt_frac_range: Tuple[float, float] = (0.4, 0.7),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pad_token_id = vocab_size - 1
    train_cfg = SequenceGenerationConfig(
        vocab_size=vocab_size,
        min_length=seq_min_len,
        max_length=seq_max_len,
        seq_types=seq_types,
        seed=seed,
    )
    val_cfg = SequenceGenerationConfig(
        vocab_size=vocab_size,
        min_length=seq_min_len,
        max_length=seq_max_len,
        seq_types=seq_types,
        seed=seed + 1,
    )
    test_cfg = SequenceGenerationConfig(
        vocab_size=vocab_size,
        min_length=seq_min_len,
        max_length=seq_max_len,
        seq_types=seq_types,
        seed=seed + 2,
    )

    train_samples = _make_preference_samples(train_cfg, seq_types, train_pairs, prompt_frac_range)
    val_samples = _make_preference_samples(val_cfg, seq_types, val_pairs, prompt_frac_range)
    test_samples = _make_preference_samples(test_cfg, seq_types, test_pairs, prompt_frac_range)

    collate_fn = lambda batch: preference_collate(batch, pad_token_id)

    train_loader = DataLoader(PreferenceDataset(train_samples), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PreferenceDataset(val_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PreferenceDataset(test_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def _make_preference_from_sequences(
    sequences: Sequence[List[int]],
    vocab_size: int,
    pairs: int,
    seed: int,
    prompt_frac_range: Tuple[float, float],
    label: str = "text",
) -> List[PreferenceSample]:
    """Create preference samples by splitting real token sequences into prompt/continuations."""
    cfg = SequenceGenerationConfig(vocab_size=vocab_size)
    rng = random.Random(seed)
    low, high = prompt_frac_range
    samples: List[PreferenceSample] = []

    for _ in range(pairs):
        if not sequences:
            break
        full_seq = rng.choice(sequences)
        if len(full_seq) < 4:
            continue
        frac = rng.uniform(low, high)
        split = max(2, min(len(full_seq) - 1, int(len(full_seq) * frac)))
        prompt = full_seq[:split]
        chosen = full_seq[split:]
        rejected = _corrupt_continuation(chosen, cfg, rng)
        samples.append(PreferenceSample(prompt, chosen, rejected, label))

    rng.shuffle(samples)
    return samples


def build_text_preference_dataloaders(
    sequences: List[List[int]],
    vocab_size: int,
    batch_size: int,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    prompt_frac_range: Tuple[float, float] = (0.4, 0.7),
    label: str = "tinystories",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build preference dataloaders from real token sequences (e.g., TinyStories).
    Splits sequences, then samples pairs with replacement for each split.
    """
    pad_token_id = vocab_size - 1
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    labeled = [(seq, label) for seq in sequences]
    train_split, val_split, test_split = split_dataset(labeled, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    train_samples = _make_preference_from_sequences([s for s, _ in train_split], vocab_size, train_pairs, seed, prompt_frac_range, label)
    val_samples = _make_preference_from_sequences([s for s, _ in val_split], vocab_size, val_pairs, seed + 1, prompt_frac_range, label)
    test_samples = _make_preference_from_sequences([s for s, _ in test_split], vocab_size, test_pairs, seed + 2, prompt_frac_range, label)

    collate_fn = lambda batch: preference_collate(batch, pad_token_id)
    train_loader = DataLoader(PreferenceDataset(train_samples), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PreferenceDataset(val_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PreferenceDataset(test_samples), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
