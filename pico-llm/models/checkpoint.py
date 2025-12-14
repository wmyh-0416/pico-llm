from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch

from models.transformer_baseline import TransformerConfig, init_baseline_model
from models.transformer_linear import init_linear_model
from models.hf_wrapper import HFConfig, init_hf_model


def save_checkpoint(model: torch.nn.Module, cfg, model_type: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_type,
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[torch.nn.Module, object, str]:
    ckpt = torch.load(path, map_location=device)
    model_type = ckpt.get("model_type", "baseline")
    cfg_dict = ckpt.get("config", {})

    if model_type == "hf":
        cfg = HFConfig(**cfg_dict)
        cfg.device = str(device)
        model = init_hf_model(cfg)
    elif model_type in ("baseline", "sft", "dpo_policy"):
        cfg = TransformerConfig(**cfg_dict)
        model = init_baseline_model(cfg)
    elif model_type == "linear":
        cfg = TransformerConfig(**cfg_dict)
        model = init_linear_model(cfg)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, model_type
