import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.preferences import build_preference_dataloaders, build_text_preference_dataloaders
from data.tinystories import build_tinystories_dataloaders, load_tinystories_tokens
from models.checkpoint import load_checkpoint, save_checkpoint
from models.transformer_baseline import TransformerConfig, init_baseline_model
from train.trainer import train_model
from train.utils import TrainingConfig, build_dataloaders, save_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) with TinyStories (default) or synthetic numeric sequences.")
    parser.add_argument("--data_source", choices=["tinystories", "synthetic"], default="tinystories", help="tinystories uses HF TinyStories; synthetic uses toy numeric sequences.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=5, help="DPO epochs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Policy learning rate during DPO.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature.")
    parser.add_argument("--length_normalize", action="store_true", help="Average logprobs by response length.")
    parser.add_argument("--vocab_size", type=int, default=128, help="Only used for synthetic mode.")
    parser.add_argument("--seq_min_len", type=int, default=8, help="Synthetic mode only.")
    parser.add_argument("--seq_max_len", type=int, default=128, help="Max tokens per sample.")
    parser.add_argument("--train_pairs", type=int, default=500, help="Preference pairs for train split.")
    parser.add_argument("--val_pairs", type=int, default=120, help="Preference pairs for val split.")
    parser.add_argument("--test_pairs", type=int, default=120, help="Preference pairs for test split.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio for TinyStories/text.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio for TinyStories/text.")
    parser.add_argument("--tinystories_limit", type=int, default=5000, help="How many TinyStories samples to load (shuffled).")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Patience in epochs for early stopping (0 disables).")
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="pref_acc",
        choices=["pref_acc", "loss"],
        help="Metric to monitor for early stopping (higher pref_acc is better; lower loss is better).",
    )
    parser.add_argument(
        "--seq_types",
        type=str,
        default="counting,arithmetic,geometric,alternating,random_walk",
        help="Comma-separated sequence types to sample.",
    )
    parser.add_argument("--sft_ckpt", type=str, default=None, help="Path to a pretrained SFT checkpoint to use as reference.")
    parser.add_argument("--sft_epochs", type=int, default=4, help="If no checkpoint is passed, train SFT for these epochs.")
    parser.add_argument("--sft_learning_rate", type=float, default=3e-4)
    parser.add_argument("--sft_train_per_type", type=int, default=300)
    parser.add_argument("--sft_val_per_type", type=int, default=60)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional per-epoch step cap for DPO.")
    parser.add_argument("--output_dir", type=str, default="results/dpo")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def concat_with_mask(
    prompt: torch.Tensor,
    response: torch.Tensor,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build concatenated prompt+response sequences plus a mask over response positions.
    Returns:
        concat: (batch, max_len)
        response_mask: (batch, max_len) bool
    """
    bsz = prompt.size(0)
    max_len = int((prompt_lens + response_lens).max().item())
    concat = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=prompt.device)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=prompt.device)
    for i in range(bsz):
        p_len = int(prompt_lens[i])
        r_len = int(response_lens[i])
        concat[i, :p_len] = prompt[i, :p_len]
        concat[i, p_len : p_len + r_len] = response[i, :r_len]
        mask[i, p_len : p_len + r_len] = True
    return concat, mask


def response_logprobs(
    model: torch.nn.Module,
    concat: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
    length_normalize: bool = False,
) -> torch.Tensor:
    logits, _, _ = model(concat)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = concat[:, 1:]
    pred_log_probs = log_probs[:, :-1, :]
    mask = response_mask[:, 1:]  # align with next-token targets
    selected = pred_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    masked = selected * mask
    token_counts = mask.sum(dim=1).clamp_min(1)
    log_sums = masked.sum(dim=1)
    if length_normalize:
        log_sums = log_sums / token_counts
    return log_sums


def dpo_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: TransformerConfig,
    beta: float,
    length_normalize: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    prompt = batch["prompt"].to(cfg.device)
    chosen = batch["chosen"].to(cfg.device)
    rejected = batch["rejected"].to(cfg.device)
    prompt_lens = batch["prompt_lens"].to(cfg.device)
    chosen_lens = batch["chosen_lens"].to(cfg.device)
    rejected_lens = batch["rejected_lens"].to(cfg.device)

    chosen_concat, chosen_mask = concat_with_mask(prompt, chosen, prompt_lens, chosen_lens, cfg.pad_token_id)
    rejected_concat, rejected_mask = concat_with_mask(prompt, rejected, prompt_lens, rejected_lens, cfg.pad_token_id)

    chosen_logprob = response_logprobs(policy, chosen_concat, chosen_mask, cfg.pad_token_id, length_normalize)
    rejected_logprob = response_logprobs(policy, rejected_concat, rejected_mask, cfg.pad_token_id, length_normalize)

    with torch.no_grad():
        ref_chosen = response_logprobs(reference, chosen_concat, chosen_mask, cfg.pad_token_id, length_normalize)
        ref_rejected = response_logprobs(reference, rejected_concat, rejected_mask, cfg.pad_token_id, length_normalize)

    pi_diff = chosen_logprob - rejected_logprob
    ref_diff = ref_chosen - ref_rejected
    logits = beta * (pi_diff - ref_diff)
    loss = -F.logsigmoid(logits).mean()
    pref_acc = (logits > 0).float().mean().item()
    stats = {
        "policy_margin": pi_diff.mean().item(),
        "ref_margin": ref_diff.mean().item(),
        "pref_acc": pref_acc,
    }
    return loss, stats


def evaluate_policy(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    loader: DataLoader,
    cfg: TransformerConfig,
    beta: float,
    length_normalize: bool,
) -> Dict[str, float]:
    policy.eval()
    reference.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for batch in loader:
            loss, stats = dpo_loss(policy, reference, batch, cfg, beta, length_normalize)
            losses.append(loss.item())
            accs.append(stats["pref_acc"])
    return {"loss": sum(losses) / max(len(losses), 1), "pref_acc": sum(accs) / max(len(accs), 1)}


def train_sft_baseline(
    args: argparse.Namespace,
    device: torch.device,
    out_root: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    pad_token_id: int,
    max_seq_len: int,
):
    """Train a quick SFT baseline if a checkpoint is not provided."""
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        learning_rate=args.sft_learning_rate,
        weight_decay=args.weight_decay,
        device=str(device),
        epochs=args.sft_epochs,
        max_steps=-1,
    )
    model = init_baseline_model(model_cfg).to(device)
    history = train_model(model, model_cfg, "sft", train_loader, val_loader, out_root / "sft")
    ckpt_path = out_root / "sft" / "sft_checkpoint.pt"
    return model, model_cfg, history, ckpt_path


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    run_dir = Path(args.output_dir) / f"run-{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data prep: TinyStories by default, synthetic numeric as fallback.
    if args.data_source == "tinystories":
        sequences, vocab_size, pad_token_id, tiny_stats = load_tinystories_tokens(
            limit=args.tinystories_limit,
            max_length=args.seq_max_len,
            seed=args.seed,
        )
        if len(sequences) == 0:
            raise ValueError("TinyStories dataset is empty after loading/encoding. Increase --tinystories_limit.")

        sft_train_loader, sft_val_loader, sft_test_loader = build_tinystories_dataloaders(
            sequences,
            pad_token_id=pad_token_id,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_loader, val_loader, test_loader = build_text_preference_dataloaders(
            sequences=sequences,
            vocab_size=vocab_size,
            batch_size=args.batch_size,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            test_pairs=args.test_pairs,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            prompt_frac_range=(0.4, 0.7),
            label="tinystories",
        )
        data_meta = {"source": "tinystories", "tinystories_stats": tiny_stats}
        max_seq_len = args.seq_max_len
    else:
        seq_types: Tuple[str, ...] = tuple(args.seq_types.split(","))
        data_cfg = TrainingConfig(
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            epochs=args.sft_epochs,
            learning_rate=args.sft_learning_rate,
            weight_decay=args.weight_decay,
            seq_min_len=args.seq_min_len,
            seq_max_len=args.seq_max_len,
            train_per_type=args.sft_train_per_type,
            val_per_type=args.sft_val_per_type,
            test_per_type=args.sft_val_per_type,
            seed=args.seed,
        )
        sft_train_loader, sft_val_loader, sft_test_loader = build_dataloaders(data_cfg, seq_types)
        train_loader, val_loader, test_loader = build_preference_dataloaders(
            vocab_size=args.vocab_size,
            seq_min_len=args.seq_min_len,
            seq_max_len=args.seq_max_len,
            seq_types=seq_types,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            test_pairs=args.test_pairs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        data_meta = {"source": "synthetic", "seq_types": seq_types}
        vocab_size = args.vocab_size
        pad_token_id = data_cfg.pad_token_id
        max_seq_len = args.seq_max_len

    # Reference model
    if args.sft_ckpt:
        ref_model, ref_cfg, _ = load_checkpoint(Path(args.sft_ckpt), device)
        ref_cfg.device = args.device
        if pad_token_id != ref_cfg.pad_token_id:
            raise ValueError(f"pad_token_id mismatch: data={pad_token_id}, checkpoint={ref_cfg.pad_token_id}")
        vocab_size = ref_cfg.vocab_size
        pad_token_id = ref_cfg.pad_token_id
        max_seq_len = ref_cfg.max_seq_len
        ref_history = None
        ref_ckpt_path = Path(args.sft_ckpt)
    else:
        ref_model, ref_cfg, ref_history, ref_ckpt_path = train_sft_baseline(
            args,
            device,
            run_dir,
            sft_train_loader,
            sft_val_loader,
            vocab_size,
            pad_token_id,
            max_seq_len,
        )

    # Policy model starts from reference weights.
    policy_cfg = TransformerConfig(**ref_cfg.__dict__)
    policy_cfg.device = args.device
    policy_cfg.learning_rate = args.learning_rate
    policy_cfg.weight_decay = args.weight_decay
    policy_cfg.epochs = args.epochs
    policy_cfg.max_steps = args.max_steps
    policy_model = init_baseline_model(policy_cfg).to(device)
    policy_model.load_state_dict(ref_model.state_dict())

    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {"train_loss": [], "train_pref_acc": [], "val_loss": [], "val_pref_acc": []}
    best_state = None
    best_val = None
    patience_ctr = 0

    for epoch in range(args.epochs):
        policy_model.train()
        epoch_losses = []
        epoch_accs = []
        for step, batch in enumerate(train_loader):
            loss, stats = dpo_loss(policy_model, ref_model, batch, policy_cfg, args.beta, args.length_normalize)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
            epoch_accs.append(stats["pref_acc"])
            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break

        val_metrics = evaluate_policy(policy_model, ref_model, val_loader, policy_cfg, args.beta, args.length_normalize)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        train_acc = sum(epoch_accs) / max(len(epoch_accs), 1)
        history["train_loss"].append(train_loss)
        history["train_pref_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_pref_acc"].append(val_metrics["pref_acc"])
        print(
            f"[dpo] epoch {epoch+1} train_loss={train_loss:.4f} train_pref_acc={train_acc:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_pref_acc={val_metrics['pref_acc']:.3f}"
        )

        # Early stopping on chosen metric
        monitor = val_metrics["pref_acc"] if args.early_stop_metric == "pref_acc" else val_metrics["loss"]
        improved = False
        if best_val is None:
            improved = True
        else:
            if args.early_stop_metric == "pref_acc":
                improved = monitor > best_val + 1e-6
            else:
                improved = monitor < best_val - 1e-6
        if improved:
            best_val = monitor
            patience_ctr = 0
            # Keep a copy of the best policy weights
            best_state = {k: v.detach().cpu().clone() for k, v in policy_model.state_dict().items()}
        else:
            patience_ctr += 1
            if args.early_stop_patience > 0 and patience_ctr >= args.early_stop_patience:
                print(f"[dpo] early stopping at epoch {epoch+1} (no improve {args.early_stop_metric} for {patience_ctr} epochs)")
                break

    # Restore best model if we tracked it
    if best_state:
        policy_model.load_state_dict(best_state)

    test_metrics = evaluate_policy(policy_model, ref_model, test_loader, policy_cfg, args.beta, args.length_normalize)

    ckpt_path = run_dir / "dpo_policy.pt"
    save_checkpoint(policy_model, policy_cfg, "dpo_policy", ckpt_path)

    metrics = {
        "history": history,
        "val_metrics": {"loss": history["val_loss"][-1] if history["val_loss"] else None, "pref_acc": history["val_pref_acc"][-1] if history["val_pref_acc"] else None},
        "test_metrics": test_metrics,
        "beta": args.beta,
        "length_normalize": args.length_normalize,
        "config": vars(args),
        "ref_checkpoint": str(ref_ckpt_path),
        "ref_history": ref_history,
        "data": data_meta,
        "sft_val_loss": ref_history["val_loss"][-1] if ref_history and "val_loss" in ref_history else None,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_patience": args.early_stop_patience,
        "best_val_metric": best_val,
    }
    save_json(metrics, run_dir / "dpo_metrics.json")
    print(f"[dpo] test_pref_acc={test_metrics['pref_acc']:.3f} loss={test_metrics['loss']:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
