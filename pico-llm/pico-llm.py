# starter code by matus & o1-pro
import argparse
import time
import random
import math
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

# Central place to collect experiment knobs / 集中解析实验所有参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory where trained model checkpoints are stored.")

    # Model selection and training hyperparameters
    parser.add_argument("--models", nargs="+", default=["lstm_seq"],
                        help="Which models to train. Options include: kgram_mlp_seq, lstm_seq, transformer.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch size for DataLoader. Default=16.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs. Default=3.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer. Default=1e-3.")
    parser.add_argument("--train_subset_size", type=int, default=20000,
                        help="How many TinyStories samples to load at most. Default=20000.")
    parser.add_argument("--log_interval_steps", type=int, default=100,
                        help="Print loss every N batches. Default=100.")
    parser.add_argument("--sample_interval_seconds", type=int, default=30,
                        help="Generate sample text every N seconds. Default=30.")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data reserved for validation. Default=0.1.")
    parser.add_argument("--test_split", type=float, default=0.0,
                        help="Fraction of data reserved for held-out testing when studying overfitting. Default=0.0.")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Random seed for dataset split and shuffles.")
    parser.add_argument("--overfit_report_path", type=str, default=None,
                        help="If set, save per-epoch train/val/test loss and sample diversity metrics to this JSON file.")
    parser.add_argument("--overfit_sample_tokens", type=int, default=40,
                        help="Number of tokens to generate per epoch when logging overfitting samples.")
    parser.add_argument("--overfit_sampler_top_ps", type=float, nargs="*",
                        default=[0.0, 0.95, 1.0],
                        help="Top-p cutoffs to use when logging overfitting samples; use 0 for greedy sampling.")

    # Transformer-specific hyperparameters
    parser.add_argument("--transformer_d_model", type=int, default=512,
                        help="Hidden size of the transformer.")
    parser.add_argument("--transformer_num_heads", type=int, default=4,
                        help="Number of attention heads.")
    parser.add_argument("--transformer_num_layers", type=int, default=4,
                        help="Number of transformer blocks.")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=4.0,
                        help="Expansion ratio for the transformer MLP.")
    parser.add_argument("--transformer_dropout", type=float, default=0.0,
                        help="Dropout applied inside the transformer.")
    parser.add_argument("--transformer_max_seq_len", type=int, default=1024,
                        help="Maximum context length for the transformer.")
    parser.add_argument("--positional_embedding", type=lambda s: s.lower(), default="learned",
                        choices=["learned", "sinusoidal", "rope", "none", "nope"],
                        help="Type of positional embedding (learned, sinusoidal, RoPE, or NoPE).")
    parser.add_argument("--rope_base", type=float, default=10000.0,
                        help="Base frequency for RoPE positional embedding (if selected).")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """Simple sampler that mixes TinyStories and custom text."""
    # Keep TinyStories & custom pools and sample by probability / 保存两类语料并按概率采样
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Probabilistically pick data source / 先掷硬币决定来自哪种语料
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    # Pad ragged sequences so downstream models share shape / 用0填充不同长度的序列以方便并行处理
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################


def split_sequence_pool(seq_list, val_ratio, rng):
    """
    Splits a list of sequences into train/validation portions.
    Ensures we keep at least one element for training when possible.
    """
    # Randomly split sequences; light safeguard to avoid empty train / 随机切分并保证训练集不至为零
    seq_list = list(seq_list)
    if val_ratio <= 0.0 or len(seq_list) <= 1:
        return seq_list, []

    n_val = max(1, int(len(seq_list) * val_ratio))
    if n_val >= len(seq_list):
        n_val = len(seq_list) - 1

    indices = list(range(len(seq_list)))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val])
    train_split = [seq_list[i] for i in range(len(seq_list)) if i not in val_indices]
    val_split = [seq_list[i] for i in range(len(seq_list)) if i in val_indices]
    return train_split, val_split


def create_sequence_loader(tiny, other, p_tiny, batch_size, shuffle):
    """
    Helper to only build a loader when data is available.
    """
    # Skip loader construction when both pools empty / 当两类语料都空时直接返回None
    if len(tiny) == 0 and len(other) == 0:
        return None

    dataset = MixedSequenceDataset(
        tinystories_seqs=tiny,
        other_seqs=other,
        p_tiny=p_tiny
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=seq_collate_fn
    )


def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    # Align model outputs with future tokens / 通过平移一个时间步来对齐预测与真值
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


def save_model_checkpoint(model, checkpoint_path, metadata):
    """
    Persist a trained model's weights together with minimal config metadata.
    """
    # Create directory + torch.save payload / 先建目录再保存配置与权重
    directory = os.path.dirname(checkpoint_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "model_type": metadata.get("model_type"),
        "config": metadata,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    print(f"[checkpoint] Saved {metadata.get('model_type')} weights to {checkpoint_path}")


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        input_dim = self.k * self.vocab_size
        hidden_dim = self.embed_size
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(max(self.num_inner_layers - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, self.vocab_size))
        # Simple feedforward pipeline / 多层感知机将拼接的one-hot上下文映射到词表logits
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        # Explicit python loops save memory, though slower / 通过显式循环换取更低显存需求
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

# Standard token embedding + LSTM decoder / 标准嵌入+LSTM结构
class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        # Run embedding->LSTM->linear pipeline / 依次通过嵌入、LSTM与线性层
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################
    
# Lightweight RMS normalization / 均方根归一化层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return self.weight * x_norm

# Implements RoPE helper / RoPE位置编码辅助类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dimension per head.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotated = torch.zeros_like(x)
        rotated[..., ::2] = -x[..., 1::2]
        rotated[..., 1::2] = x[..., ::2]
        return (x * cos) + (rotated * sin)


# Basic causal self-attention block / 基本的多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rotary = use_rotary

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=rope_base) if use_rotary else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Project to QKV -> apply mask/softmax -> mix heads / 投影得到QKV后加mask做加权求和
        seq_len, batch_size, _ = x.shape

        qkv = self.qkv(x.transpose(0, 1))  # (batch, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            if rope_cache is None:
                raise ValueError("RoPE cache must be provided when use_rotary=True.")
            cos, sin = rope_cache
            q = RotaryEmbedding.apply_rotary(q, cos, sin)
            k = RotaryEmbedding.apply_rotary(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context).transpose(0, 1)
        return output


# Transformer MLP sub-layer / Transformer中的前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Single transformer layer = attention + FFN / 单层Transformer由注意力+前馈组成
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rotary: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rotary = use_rotary
        self.attn_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            rope_base=rope_base,
        )
        hidden_dim = int(d_model * mlp_ratio)
        self.ff = FeedForward(d_model, hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        attn_input = self.attn_norm(x)
        x = x + self.attn(attn_input, attn_mask, rope_cache)
        ff_input = self.ff_norm(x)
        x = x + self.ff(ff_input)
        return x



class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        positional_embedding: str = "learned",
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.positional_embedding = positional_embedding
        self.max_seq_len = max_seq_len
        self.use_rope = positional_embedding.lower() == "rope"
        self.rope_base = rope_base

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        pos_type = positional_embedding.lower()
        # Allow multiple positional encoding options / 支持多种位置编码配置
        if pos_type == "learned":
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        elif pos_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("sinusoidal_pe", pe, persistent=False)
            self.pos_embedding = None
        elif pos_type == "rope":
            self.pos_embedding = None
        elif pos_type in {"none", "nope"}:
            self.pos_embedding = None
        else:
            raise ValueError(f"Unknown positional_embedding type: {positional_embedding}")

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rotary=self.use_rope,
                    rope_base=rope_base,
                )
                for _ in range(n_blocks)
            ]
        )
        if self.use_rope and len(self.blocks) == 0:
            raise ValueError("RoPE positional embedding requires at least one transformer block.")
        self.final_norm = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        causal_mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # Store upper-triangular mask to block future positions / 构建上三角mask禁止看未来token
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = tokens_seq.shape
        if seq_len > self.max_seq_len:
            tokens_seq = tokens_seq[-self.max_seq_len :]
            seq_len = tokens_seq.shape[0]

        tok_emb = self.token_embedding(tokens_seq)  # (seq_len, batch, d_model)

        pos_type = self.positional_embedding.lower()
        if pos_type == "learned":
            positions = torch.arange(seq_len, device=tokens_seq.device).unsqueeze(1)
            pos_emb = self.pos_embedding(positions).expand(-1, batch_size, -1)
            x = tok_emb + pos_emb
        elif pos_type == "sinusoidal":
            pos_emb = self.sinusoidal_pe[:seq_len, :].to(device=tokens_seq.device, dtype=tok_emb.dtype).unsqueeze(1)
            x = tok_emb + pos_emb
        else:  # rope or none / RoPE或无显式位置编码
            x = tok_emb

        x = self.dropout(x)

        attn_mask = self.causal_mask[:seq_len, :seq_len]
        attn_mask = attn_mask.to(tokens_seq.device)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        rope_cache = None
        if self.use_rope and len(self.blocks) > 0:
            # Precompute cos/sin tables once per forward / 每次前向只计算一次RoPE旋转表
            rope_cache = self.blocks[0].attn.rotary_emb(seq_len, tokens_seq.device, tok_emb.dtype)

        for block in self.blocks:
            # Sequentially apply transformer blocks / 逐层应用Transformer块
            x = block(x, attn_mask, rope_cache)

        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    # Placeholder: hook for interpretability work / 占位函数，可接入可解释性分析
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    # Sample token from nucleus set / 使用top-p（核采样）从概率尾部截断后的集合中采样
    if p is None or p <= 0.0:
        return torch.argmax(logits).item()

    probs = torch.softmax(logits, dim=-1)

    if p >= 1.0:
        choice = torch.multinomial(probs, num_samples=1)
        return choice.item()

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff_idx = torch.nonzero(cumulative >= p, as_tuple=False)
    if cutoff_idx.numel() == 0:
        cutoff = sorted_probs.size(0)
    else:
        cutoff = cutoff_idx[0].item() + 1

    top_probs = sorted_probs[:cutoff]
    top_indices = sorted_indices[:cutoff]
    top_probs = top_probs / top_probs.sum()

    sampled_idx = torch.multinomial(top_probs, num_samples=1)
    return top_indices[sampled_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        # Keep growing context for autoregressive sampling / 自回归采样不停扩大上下文
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


def compute_text_diversity(text, enc):
    """
    Compute simple diversity statistics for a generated text sample.
    """
    # Quick heuristic stats (unique tokens etc.) / 粗略统计重复率、独特词数
    if enc is None or not text:
        return {
            "num_tokens": 0,
            "unique_tokens": 0,
            "type_token_ratio": 0.0,
            "repeat_ratio": 0.0,
        }

    token_ids = enc.encode(text)
    num_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    type_token_ratio = float(unique_tokens) / num_tokens if num_tokens > 0 else 0.0
    repeat_ratio = 1.0 - type_token_ratio if num_tokens > 0 else 0.0
    return {
        "num_tokens": num_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": type_token_ratio,
        "repeat_ratio": repeat_ratio,
    }


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    val_loader=None,
                    test_loader=None,
                    overfit_options=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    # Shared training controller / 通用训练主循环
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    overfit_history = [] if overfit_options is not None else None

    for epoch in range(1, epochs + 1):
        # One full pass over loader / 开始新一轮epoch
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            # Move batch to device & run forward/backward / 将批次送入设备并完成一次更新
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if log_steps > 0 and batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    # Periodically sample text for qualitative check / 定期生成文本方便观察模型学习情况
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        val_loss = None
        test_loss = None
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device)
        if test_loader is not None:
            test_loss = evaluate_model(model, test_loader, device)

        msg = f"[{model_name}] *** End of Epoch {epoch} *** Train Loss: {avg_loss:.4f}"
        if val_loss is not None:
            val_gap = val_loss - avg_loss
            msg += f" | Val Loss: {val_loss:.4f} (Gap={val_gap:.4f})"
        if test_loss is not None:
            test_gap = test_loss - avg_loss
            msg += f" | Test Loss: {test_loss:.4f} (Gap={test_gap:.4f})"
        print(msg)

        if overfit_history is not None and enc is not None:
            # Capture short generations for overfitting study / 记录样本以分析过拟合
            sample_prompt = overfit_options.get("prompt", prompt)
            sample_tokens = overfit_options.get("sample_tokens", 20)
            sampler_top_ps = overfit_options.get("top_ps", [None])
            sample_records = []
            for raw_top_p in sampler_top_ps:
                sampler_top_p = None if raw_top_p is None or raw_top_p <= 0.0 else float(raw_top_p)
                text_out, _ = generate_text(
                    model,
                    enc,
                    sample_prompt,
                    max_new_tokens=sample_tokens,
                    device=device,
                    top_p=sampler_top_p,
                    monosemantic_info=monosemantic_info,
                    do_monosemantic=False,
                )
                if sampler_top_p is None:
                    label = "greedy"
                else:
                    label = f"top_p_{min(1.0, sampler_top_p):.2f}"
                sample_records.append({
                    "label": label,
                    "top_p": sampler_top_p,
                    "text": text_out,
                    "diversity": compute_text_diversity(text_out, enc),
                })

            overfit_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_val_gap": (val_loss - avg_loss) if val_loss is not None else None,
                "train_test_gap": (test_loss - avg_loss) if test_loss is not None else None,
                "samples": sample_records,
                "timestamp": time.time(),
            })

    return overfit_history


def evaluate_model(model, loader, device):
    """
    Compute the average next-token prediction loss over a dataloader.
    """
    if loader is None:
        return float("nan")

    was_training = model.training
    model.eval()
    # No grad + running loss average / 关闭梯度，简单算平均损失
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_tokens in loader:
            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            total_loss += loss.item()
            count += 1
    model.train(was_training)

    return total_loss / max(count, 1)


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()
    # CLI -> python dict of settings / 将命令行参数整理成本地配置

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    block_size = args.block_size
    train_subset_size = args.train_subset_size
    log_interval_steps = args.log_interval_steps
    sample_interval_seconds = args.sample_interval_seconds

    transformer_max_seq_len = min(args.transformer_max_seq_len, block_size)
    val_split = max(0.0, min(args.val_split, 0.9))
    test_split = max(0.0, min(args.test_split, 0.5))
    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.data_seed)
    # Deterministic-ish splits / 固定随机种子以复现实验
    rng = random.Random(args.data_seed)
    requested_models = [m.lower() for m in args.models] if args.models else ["lstm_seq"]
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    # Report important hyperparameters / 打印主要训练配置方便检查
    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")
    print(f"Training hyperparameters: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}, "
          f"val_split={val_split}, test_split={test_split}")
    print(f"Requested models: {requested_models}")
    print(f"Transformer config: d_model={args.transformer_d_model}, heads={args.transformer_num_heads}, "
          f"layers={args.transformer_num_layers}, pos_emb={args.positional_embedding}, max_seq_len={transformer_max_seq_len}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        # Convert HF samples to token id lists / 将TinyStories文本编码成token序列
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny > 0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    tiny_train_pool, tiny_test = split_sequence_pool(tinystories_seqs, test_split, rng)
    other_train_pool, other_test = split_sequence_pool(other_seqs, test_split, rng)

    tiny_train, tiny_val = split_sequence_pool(tiny_train_pool, val_split, rng)
    other_train, other_val = split_sequence_pool(other_train_pool, val_split, rng)

    split_summary = {
        "train": {"tinystories": len(tiny_train), "custom": len(other_train)},
        "val": {"tinystories": len(tiny_val), "custom": len(other_val)},
        "test": {"tinystories": len(tiny_test), "custom": len(other_test)},
    }

    print(f"Train split: TinyStories={len(tiny_train)}, Custom={len(other_train)}")
    if val_split > 0.0:
        print(f"Val split: TinyStories={len(tiny_val)}, Custom={len(other_val)}")
    if test_split > 0.0:
        print(f"Test split: TinyStories={len(tiny_test)}, Custom={len(other_test)}")

    # Build DataLoaders for each phase / 为训练、验证、测试建立迭代器
    train_loader = create_sequence_loader(
        tiny=tiny_train,
        other=other_train,
        p_tiny=p_tiny,
        batch_size=batch_size,
        shuffle=True
    )
    if train_loader is None:
        raise ValueError("No training data available after splitting.")

    val_loader = create_sequence_loader(
        tiny=tiny_val,
        other=other_val,
        p_tiny=p_tiny,
        batch_size=batch_size,
        shuffle=False
    ) if (len(tiny_val) + len(other_val)) > 0 else None
    if val_loader is None and val_split > 0.0:
        print("Validation loader unavailable (not enough validation data after split).")

    test_loader = create_sequence_loader(
        tiny=tiny_test,
        other=other_test,
        p_tiny=p_tiny,
        batch_size=batch_size,
        shuffle=False
    ) if (len(tiny_test) + len(other_test)) > 0 else None
    if test_loader is None and test_split > 0.0:
        print("Test loader unavailable (not enough test data after split).")

    overfit_logging_enabled = bool(args.overfit_report_path)
    overfit_options = None
    overfit_reports = {}
    if overfit_logging_enabled:
        # Normalize sampler top-p list / 规范化overfit日志采样参数
        sampler_values = args.overfit_sampler_top_ps or [0.0]
        sanitized_top_ps = []
        for tp in sampler_values:
            if tp is None:
                sanitized_top_ps.append(None)
                continue
            value = float(tp)
            if value <= 0.0:
                sanitized_top_ps.append(None)
            else:
                sanitized_top_ps.append(min(value, 1.0))
        if not sanitized_top_ps:
            sanitized_top_ps = [None]
        overfit_options = {
            "prompt": args.prompt,
            "sample_tokens": max(1, args.overfit_sample_tokens),
            "top_ps": sanitized_top_ps,
        }

    ############################################################################
    # Models
    ############################################################################
    models = {}
    model_configs = {}
    for name in requested_models:
        if name in models:
            continue

        if name == "kgram_mlp_seq":
            config = {
                "model_type": "kgram_mlp_seq",
                "vocab_size": vocab_size,
                "kgram_k": k,
                "embed_size": embed_size,
                "num_inner_layers": num_inner_layers,
                "chunk_size": chunk_size,
            }
            # Instantiate sliding-window MLP / 构建k-gram MLP模型
            models[name] = KGramMLPSeqModel(
                vocab_size=vocab_size,
                k=k,
                embed_size=embed_size,
                num_inner_layers=num_inner_layers,
                chunk_size=chunk_size
            ).to(device)
            model_configs[name] = config
        elif name == "lstm_seq":
            config = {
                "model_type": "lstm_seq",
                "vocab_size": vocab_size,
                "embed_size": embed_size,
                "hidden_size": embed_size,
            }
            # Instantiate stacked LSTM / 构建LSTM序列模型
            models[name] = LSTMSeqModel(
                vocab_size=vocab_size,
                embed_size=embed_size,
                hidden_size=embed_size
            ).to(device)
            model_configs[name] = config
        elif name in {"transformer", "transformer_seq", "kvcache_transformer"}:
            canonical_name = "transformer"
            if canonical_name in models:
                continue
            config = {
                "model_type": canonical_name,
                "vocab_size": vocab_size,
                "d_model": args.transformer_d_model,
                "n_heads": args.transformer_num_heads,
                "n_blocks": args.transformer_num_layers,
                "mlp_ratio": args.transformer_mlp_ratio,
                "dropout": args.transformer_dropout,
                "max_seq_len": transformer_max_seq_len,
                "positional_embedding": args.positional_embedding,
                "rope_base": args.rope_base,
            }
            # Instantiate toy Transformer / 构建自定义Transformer模型
            models[canonical_name] = TransformerModel(
                vocab_size=vocab_size,
                d_model=args.transformer_d_model,
                n_heads=args.transformer_num_heads,
                n_blocks=args.transformer_num_layers,
                mlp_ratio=args.transformer_mlp_ratio,
                dropout=args.transformer_dropout,
                max_seq_len=transformer_max_seq_len,
                positional_embedding=args.positional_embedding,
                rope_base=args.rope_base,
            ).to(device)
            model_configs[canonical_name] = config
        else:
            print(f"Unknown model name '{name}' requested; skipping.")

    if not models:
        raise ValueError("No valid models selected. Choose from: kgram_mlp_seq, lstm_seq, transformer.")


    ############################################################################
    # Train each model
    ############################################################################
    # Train/eval each requested model sequentially / 依次训练并评估每一种模型
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        history = train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            val_loader=val_loader,
            test_loader=test_loader,
            overfit_options=overfit_options if overfit_logging_enabled else None,
        )

        if overfit_logging_enabled:
            overfit_reports[model_name] = {
                "model_config": model_configs.get(model_name, {"model_type": model_name}),
                "history": history or [],
            }

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        # Print multilingual-friendly summary of outputs / 打印不同采样策略生成的最终文本
        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")

        # Persist weights for later reuse / 保存模型权重供复现
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_final.pt")
        checkpoint_meta = model_configs.get(model_name, {"model_type": model_name})
        save_model_checkpoint(model, checkpoint_path, checkpoint_meta)
        print("--------------------------------------------------")

    if overfit_logging_enabled and args.overfit_report_path:
        # Serialize report for later inspection / 将过拟合分析写成JSON
        report_payload = {
            "generated_at": time.time(),
            "data_splits": split_summary,
            "train_args": {
                "block_size": block_size,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "val_split": val_split,
                "test_split": test_split,
                "tinystories_weight": args.tinystories_weight,
                "input_files": args.input_files,
                "prompt": args.prompt,
            },
            "overfit_logging": overfit_options,
            "models": overfit_reports,
        }
        with open(args.overfit_report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Saved overfitting report to {args.overfit_report_path}")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
