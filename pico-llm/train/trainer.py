import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.checkpoint import save_checkpoint
from models.transformer_baseline import TransformerConfig


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    logits: (batch, seq, vocab)
    targets: (batch, seq)
    """
    if logits.size(1) < 2:
        return torch.tensor(0.0, device=logits.device)
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    gold = targets[:, 1:].reshape(-1)
    return F.cross_entropy(pred, gold, ignore_index=pad_token_id)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> float:
    if logits.size(1) < 2:
        return 0.0
    pred = logits[:, :-1, :].argmax(dim=-1)
    gold = targets[:, 1:]
    mask = gold != pad_token_id
    correct = (pred == gold) & mask
    total = mask.sum().item()
    return float(correct.sum().item()) / max(total, 1)


def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: TransformerConfig) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            tokens, _, _ = batch
            tokens = tokens.to(cfg.device)
            logits, _, _ = model(tokens)
            loss = next_token_loss(logits, tokens, cfg.pad_token_id)
            losses.append(loss.item())
            accs.append(compute_accuracy(logits, tokens, cfg.pad_token_id))
    return sum(losses) / len(losses), sum(accs) / len(accs)


def train_model(
    model: torch.nn.Module,
    cfg: TransformerConfig,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(cfg, "learning_rate", 3e-4),
        weight_decay=getattr(cfg, "weight_decay", 0.0),
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time": []}

    for epoch in range(getattr(cfg, "epochs", 1)):
        model.train()
        start = time.time()
        train_losses: List[float] = []
        for step, batch in enumerate(train_loader):
            tokens, _, _ = batch
            tokens = tokens.to(cfg.device)
            logits, _, _ = model(tokens)
            loss = next_token_loss(logits, tokens, cfg.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

            max_steps = getattr(cfg, "max_steps", -1)
            if max_steps > 0 and (step + 1) >= max_steps:
                break

        epoch_time = time.time() - start
        val_loss, val_acc = evaluate(model, val_loader, cfg)
        history["train_loss"].append(sum(train_losses) / max(len(train_losses), 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        print(
            f"[{model_type}] epoch {epoch+1} loss={history['train_loss'][-1]:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} time={epoch_time:.1f}s"
        )

    # Save checkpoint
    ckpt_path = output_dir / f"{model_type}_checkpoint.pt"
    save_checkpoint(model, cfg, model_type, ckpt_path)

    # Plot loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_loss_curve.png", dpi=200)
    plt.close()

    return history
