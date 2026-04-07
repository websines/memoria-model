"""DFlash draft head: block diffusion for speculative decoding.

Generates block_size tokens in parallel via cross-attention to the target
model's intermediate hidden states + active belief embeddings. The full
model then verifies the block in one pass (including refinement loops).

Why this helps MemoriaModel specifically:
- Refinement loops make per-token generation ~4x more expensive
- Speculative decoding amortizes that cost across accepted blocks
- Cognitive state as additional cross-attention context improves drafts

Architecture (adapted from DFlash, arXiv:2602.06036):
- Mask token embeddings + positional encoding for draft positions
- N cross-attention layers: Q=draft, KV=concat(target_features, beliefs, draft)
- Shared LM head with main model (no extra 117M params)
- Single-shot parallel prediction (no iterative diffusion at inference)

Reference: DFlash (Chen, Liang, Liu — arXiv:2602.06036)
Reference: Speculative Decoding (Leviathan et al., ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DFlashCrossAttention(nn.Module):
    """Cross+self attention for draft tokens.

    Queries come from draft hidden states. Keys and values come from the
    concatenation of target features, belief embeddings, AND draft tokens
    themselves. The self-attention component lets draft tokens coordinate
    with each other for block coherence.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # QK norm for stability (matches main transformer)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, draft: Tensor, context: Tensor) -> Tensor:
        """Cross+self attention.

        Args:
            draft: [B, block_size, D] draft hidden states
            context: [B, T_ctx, D] target features (+ beliefs)

        Returns:
            [B, block_size, D] updated draft hidden states
        """
        B, N, D = draft.shape

        # Q from draft only
        q = self.q_proj(draft)

        # K, V from concat(context, draft) — enables both cross and self attention
        kv_input = torch.cat([context, draft], dim=1)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head attention
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention (no causal mask — draft tokens see all context + each other)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class DFlashDraftLayer(nn.Module):
    """Single draft layer: cross+self attention + MLP."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm_ctx = nn.RMSNorm(hidden_dim)
        self.attn = DFlashCrossAttention(hidden_dim, n_heads)
        self.norm2 = nn.RMSNorm(hidden_dim)
        # Smaller MLP than main model (2x expansion instead of 4x)
        self.fc = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

    def forward(self, draft: Tensor, context: Tensor) -> Tensor:
        # Cross+self attention
        draft = draft + self.attn(self.norm1(draft), self.norm_ctx(context))
        # MLP with ReLU²
        h = self.fc(self.norm2(draft))
        h = F.relu(h).square()
        draft = draft + self.proj(h)
        return draft


class DFlashDraftHead(nn.Module):
    """Block diffusion draft head for MemoriaModel.

    Predicts block_size tokens in parallel from target model hidden states
    and active belief embeddings. Used for speculative decoding at inference
    and trained jointly with the main model via auxiliary loss.

    The LM head weights are shared with the main model to avoid duplicating
    the vocab projection (~117M params). A small adapter projects from draft
    space to the main model's output space.
    """

    def __init__(self, hidden_dim: int, n_heads: int, n_layers: int,
                 block_size: int, n_target_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.n_layers = n_layers

        # Compute which target layers to tap (evenly spaced)
        self.tap_indices = self._build_tap_indices(n_target_layers, n_layers)

        # Mask token embedding (draft positions start as this)
        self.mask_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional embedding for draft positions within the block
        self.pos_embed = nn.Embedding(block_size, hidden_dim)

        # Feature projection: concat of tapped target hidden states → hidden_dim
        # Each tap contributes hidden_dim features; we project down
        self.feature_proj = nn.Linear(hidden_dim * n_layers, hidden_dim, bias=False)

        # Draft layers
        self.layers = nn.ModuleList([
            DFlashDraftLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Output norm (before shared LM head)
        self.out_norm = nn.RMSNorm(hidden_dim)

        # Initialize feature_proj and output to near-zero so draft starts as no-op
        nn.init.normal_(self.feature_proj.weight, std=0.01)

    @staticmethod
    def _build_tap_indices(n_target_layers: int, n_taps: int) -> list[int]:
        """Select evenly-spaced target layers to extract features from.

        Matches DFlash's build_target_layer_ids strategy.
        """
        if n_taps <= 1:
            return [n_target_layers - 1]
        return [int(i * (n_target_layers - 1) / (n_taps - 1))
                for i in range(n_taps)]

    def extract_features(
        self,
        layer_hiddens: dict[int, Tensor],
        belief_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Extract and project target features for the draft head.

        Args:
            layer_hiddens: {layer_idx: [B, T, D]} hidden states at tapped layers
            belief_embeddings: [N_active, D] active belief vectors (optional)

        Returns:
            [B, T_ctx, D] context features for cross-attention
        """
        # Collect tapped features — take last position only (autoregressive context)
        tapped = []
        for idx in self.tap_indices:
            if idx in layer_hiddens:
                tapped.append(layer_hiddens[idx])

        if not tapped:
            raise ValueError("No target hidden states available at tap indices")

        # Concat along feature dim and project: [B, T, D*n_taps] → [B, T, D]
        combined = torch.cat(tapped, dim=-1)  # [B, T, D * n_taps]
        context = self.feature_proj(combined)  # [B, T, D]

        # Append belief embeddings as additional context tokens
        if belief_embeddings is not None and belief_embeddings.shape[0] > 0:
            B = context.shape[0]
            # beliefs are [N, D] → expand to [B, N, D]
            beliefs = belief_embeddings.unsqueeze(0).expand(B, -1, -1)
            context = torch.cat([context, beliefs], dim=1)

        return context

    def forward(
        self,
        context: Tensor,
        lm_head_fn=None,
    ) -> Tensor:
        """Generate draft logits for block_size positions.

        Args:
            context: [B, T_ctx, D] target features + beliefs
            lm_head_fn: callable that maps [B, N, D] → [B, N, vocab] (shared LM head)

        Returns:
            [B, block_size, vocab_size] draft logits
        """
        B = context.shape[0]
        device = context.device

        # Initialize draft as mask embeddings + position
        pos_ids = torch.arange(self.block_size, device=device)
        draft = self.mask_embed.expand(B, self.block_size, -1) + self.pos_embed(pos_ids)

        # Run through draft layers
        for layer in self.layers:
            draft = layer(draft, context)

        # Apply output norm
        draft = self.out_norm(draft)

        # Project to vocab via shared LM head
        if lm_head_fn is not None:
            return lm_head_fn(draft)
        return draft

    def compute_draft_loss(
        self,
        context: Tensor,
        target_tokens: Tensor,
        lm_head_fn=None,
    ) -> Tensor:
        """Compute training loss for the draft head.

        Predicts target_tokens given context features. Uses cross-entropy
        with the shared LM head.

        Args:
            context: [B, T_ctx, D] target features + beliefs
            target_tokens: [B, block_size] ground truth tokens to predict
            lm_head_fn: shared LM head function

        Returns:
            scalar loss
        """
        logits = self.forward(context, lm_head_fn)  # [B, block_size, vocab]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=-1,
        )
