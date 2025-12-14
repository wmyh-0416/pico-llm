import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    max_seq_len: int = 256
    pad_token_id: int = -1
    positional_embedding: str = "sinusoidal"  # "sinusoidal" or "learned"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    device: str = "cpu"
    epochs: int = 10
    max_steps: int = -1

    def __post_init__(self):
        if self.pad_token_id < 0:
            self.pad_token_id = self.vocab_size - 1


def _build_sinusoidal(max_len: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(0, max_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        out = self.out_proj(out)
        attn_to_return = attn if return_attn else None
        return out, attn_to_return


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, int(cfg.d_model * cfg.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(cfg.d_model * cfg.mlp_ratio), cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, return_attn: bool = False):
        attn_out, attn_map = self.attn(self.ln1(x), causal_mask, return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_map


class TinyTransformer(nn.Module):
    """Minimal Transformer decoder for numeric sequences."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positional_embedding == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("sinusoidal_pe", None, persistent=False)
        else:
            self.pos_emb = None
            self.register_buffer("sinusoidal_pe", _build_sinusoidal(cfg.max_seq_len, cfg.d_model, torch.device("cpu")), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Shape: (1, 1, seq_len, seq_len) for broadcasting over batch and heads.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq)
        Returns:
            logits: (batch, seq, vocab)
            attn_maps: list of (batch, heads, seq, seq) or empty
            hidden_states: list of (batch, seq, d_model) per layer if requested
        """
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        tok = self.token_emb(input_ids)
        if self.pos_emb is not None:
            max_pos = self.pos_emb.num_embeddings
            positions = torch.arange(seq_len, device=device).clamp_max(max_pos - 1).unsqueeze(0)
            pos = self.pos_emb(positions)
        else:
            if seq_len <= self.sinusoidal_pe.size(0):
                pos = self.sinusoidal_pe[:seq_len, :].unsqueeze(0).to(device=device, dtype=tok.dtype)
            else:
                pos = _build_sinusoidal(seq_len, self.cfg.d_model, device=device).unsqueeze(0).to(dtype=tok.dtype)

        x = self.drop(tok + pos)

        attn_maps: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []
        causal_mask = self._causal_mask(seq_len, device)

        for block in self.blocks:
            x, attn = block(x, causal_mask, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_maps.append(attn.detach())
            if return_hidden:
                hidden_states.append(x.detach())

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, attn_maps, hidden_states


def init_baseline_model(cfg: TransformerConfig) -> TinyTransformer:
    model = TinyTransformer(cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
