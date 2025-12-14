import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.transformer_baseline import TransformerConfig


def _feature_map(x: torch.Tensor) -> torch.Tensor:
    # Performer-style non-negative feature map to keep causal prefix sums stable.
    return torch.nn.functional.elu(x) + 1.0


class LinearCausalAttention(nn.Module):
    """Kernelized causal attention with O(n) time and space."""

    def __init__(self, dim: int, n_heads: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq, dim)
            causal_mask: (1,1,seq,seq) boolean lower-triangular mask
        """
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,s,d)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        phi_q = _feature_map(q)
        phi_k = _feature_map(k)

        # Prefix sums for denominator and numerator.
        k_cum = phi_k.cumsum(dim=2)  # (b,h,s,d)
        kv = torch.einsum("bhsi,bhsj->bhsij", phi_k, v)  # (b,h,s,d,d)
        kv_cum = kv.cumsum(dim=2)  # (b,h,s,d,d)

        numer = torch.einsum("bhsi,bhsij->bhsj", phi_q, kv_cum)  # (b,h,s,d)
        denom = torch.einsum("bhsi,bhsi->bhs", phi_q, k_cum) + self.eps  # (b,h,s)
        attn_out = numer / denom.unsqueeze(-1)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        attn_out = self.out_proj(attn_out)

        if not return_attn:
            return attn_out, None

        # Approximate attention weights (normalized causal kernel similarities).
        raw = torch.einsum("bhqd,bhkd->bhqk", phi_q, phi_k)  # (b,h,seq,seq)
        # Apply causal mask (True => keep).
        raw = raw * causal_mask
        raw_sum = raw.sum(dim=-1, keepdim=True) + self.eps
        attn_weights = raw / raw_sum
        attn_weights = torch.clamp(attn_weights, min=0.0)
        attn_weights = self.dropout(attn_weights)
        return attn_out, attn_weights


class LinearTransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = LinearCausalAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
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


class LinearAttentionTransformer(nn.Module):
    """Tiny Transformer variant with Performer-style linear attention."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positional_embedding == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("sinusoidal_pe", None, persistent=False)
        else:
            self.pos_emb = None
            self.register_buffer(
                "sinusoidal_pe",
                torch.zeros(cfg.max_seq_len, cfg.d_model),
                persistent=False,
            )
        self._init_sinusoidal()
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LinearTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _init_sinusoidal(self):
        if self.pos_emb is not None:
            return
        # Initialize sinusoidal table on CPU; moved to device at runtime.
        pe = torch.zeros(self.cfg.max_seq_len, self.cfg.d_model)
        position = torch.arange(0, self.cfg.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.cfg.d_model, 2) * (-math.log(10000.0) / self.cfg.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.sinusoidal_pe.copy_(pe)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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
                position = torch.arange(0, seq_len, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.cfg.d_model, 2, device=device) * (-math.log(10000.0) / self.cfg.d_model))
                pe = torch.zeros(seq_len, self.cfg.d_model, device=device, dtype=tok.dtype)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pos = pe.unsqueeze(0)
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


def init_linear_model(cfg: TransformerConfig) -> LinearAttentionTransformer:
    model = LinearAttentionTransformer(cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
