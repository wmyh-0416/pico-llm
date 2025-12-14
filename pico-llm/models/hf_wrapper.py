from dataclasses import dataclass, asdict
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFConfig:
    """Lightweight config for HF causal LM models."""

    model_name: str = "gpt2"
    pad_token_id: int = 50256
    device: str = "cpu"
    max_seq_len: int = 1024
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0


class HFCausalLM(torch.nn.Module):
    """Wrapper to make HF causal LM conform to our (logits, None, None) interface."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor):
        outputs = self.hf_model(input_ids)
        logits = outputs.logits
        return logits, None, None


def load_hf_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad token exists; default to eos if missing.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def init_hf_model(cfg: HFConfig) -> HFCausalLM:
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(model.config.vocab_size)  # ensure embeddings align after pad handling
    model.to(cfg.device)
    return HFCausalLM(model)
