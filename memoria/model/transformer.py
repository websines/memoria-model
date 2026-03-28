"""GPT Transformer blocks, ported from autoresearch/train.py (Karpathy).

Includes: RoPE, QK-Norm, ReLU², value embeddings, per-layer residual scalars,
logit softcapping. Muon optimizer setup.

This is the frozen language backbone. State interface layers are inserted
in memoria_model.py, not here.

Reference: github.com/karpathy/autoresearch (train.py)
Reference: github.com/KellerJordan/modded-nanogpt
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import asdict

from .config import TransformerConfig


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # GQA: expand kv heads if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_head, self.head_dim)

        # Standard scaled dot-product attention (use F.scaled_dot_product_attention for flash)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² from nanogpt speedrun
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, x0: Tensor, cos: Tensor, sin: Tensor,
                resid_lambda: Tensor, x0_lambda: Tensor) -> Tensor:
        # Per-layer residual scaling (from autoresearch)
        x = resid_lambda * x + x0_lambda * x0
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    """GPT-style transformer backbone.

    Does NOT include state interface layers — those are added in MemoriaModel.
    This is the pure language model component.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer residual scalars (from autoresearch)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Precompute rotary embeddings
        head_dim = config.n_embd // config.n_head
        self._init_rotary(config.sequence_len * 2, head_dim)

        # Softcap for logits
        self.softcap = 15.0

    def _init_rotary(self, seq_len: int, head_dim: int, base: int = 10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Custom weight initialization (from autoresearch)."""
        s = 3**0.5 * self.config.n_embd**-0.5
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

    def forward_blocks(self, idx: Tensor) -> list[Tensor]:
        """Run through transformer blocks, returning hidden states after each block.

        Used by MemoriaModel to insert state interface layers between blocks.

        Args:
            idx: [B, T] token indices

        Returns:
            List of [B, T, n_embd] hidden states, one per block
        """
        B, T = idx.size()
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x

        hiddens = []
        for i, block in enumerate(self.blocks):
            x = block(x, x0, cos, sin, self.resid_lambdas[i], self.x0_lambdas[i])
            hiddens.append(x)

        return hiddens

    def head(self, x: Tensor) -> Tensor:
        """Apply LM head to final hidden state.

        Args:
            x: [B, T, n_embd] final hidden state

        Returns:
            [B, T, vocab_size] logits
        """
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:
        """Standard forward pass (no state interface, for baseline comparison).

        Args:
            idx: [B, T] token indices
            targets: [B, T] target token indices (optional)

        Returns:
            logits or loss
        """
        hiddens = self.forward_blocks(idx)
        logits = self.head(hiddens[-1])

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits
