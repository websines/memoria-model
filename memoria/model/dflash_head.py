"""DFlash draft head: block diffusion for speculative decoding.

Generates block_size tokens in parallel via cross-attention to the target
model's intermediate hidden states + active belief embeddings. The full
model then verifies the block in one pass (including refinement loops).

Why this helps MemoriaModel specifically:
- Refinement loops make per-token generation ~4x more expensive
- Speculative decoding amortizes that cost across accepted blocks
- Cognitive state as additional cross-attention context improves drafts

Three improvements over baseline block diffusion:
1. KV injection (DFlash arXiv:2602.06036): per-layer K/V projections for
   tapped target features → tighter alignment with verifier than concat
2. Streak distillation (SpecDiff-2 arXiv:2511.00606): position-weighted CE
   + expected streak bonus → optimizes consecutive acceptance, not per-token
3. Adaptive block size (FailFast arXiv:2512.20573): entropy-based cutoff
   → draft max_block_size, verify only confident prefix

Architecture:
- Mask token embeddings + positional encoding for draft positions
- N cross-attention layers: Q=draft, KV=concat(beliefs, draft) + KV-injected target features
- KV injection: per-layer K/V projections for tapped hidden states
- Injection projections marked for 3-bit RotorQuant (draft accuracy < verifier)
- Shared LM head with main model (no extra 117M params)
- Single-shot parallel prediction (no iterative diffusion at inference)

Reference: DFlash (Chen, Liang, Liu — arXiv:2602.06036) — KV injection
Reference: SpecDiff-2 (Sandler et al. — arXiv:2511.00606) — streak distillation
Reference: FailFast (Pan, Chen, Netravali — arXiv:2512.20573) — adaptive block size
Reference: DEER (Cheng et al. — arXiv:2512.15176) — single-step diffusion drafting
Reference: Speculative Decoding (Leviathan et al., ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DFlashCrossAttention(nn.Module):
    """Cross+self attention with KV injection for draft tokens.

    Queries come from draft hidden states. Keys and values come from two sources:
    1. Context tokens (beliefs + draft self-attention) via standard Q/K/V projections
    2. KV-injected target features via separate per-layer K/V projections

    KV injection (DFlash arXiv:2602.06036): instead of concatenating tapped target
    features as context tokens, each layer has its own k_inject/v_inject projections.
    Each layer gets independent interpretation of target representations and K/V
    spaces are decoupled (K specializes in query matching, V in information).
    Injection projections marked for 3-bit RotorQuant (draft accuracy < verifier).
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # KV injection projections for target features
        # Zero-initialized: starts as no-op, learns target conditioning gradually
        self.k_inject = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_inject = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.k_inject.weight)
        nn.init.zeros_(self.v_inject.weight)
        # Mark for aggressive quantization (3-bit eligible via RotorQuant)
        self.k_inject._qat_bits = 3
        self.v_inject._qat_bits = 3

        # QK norm for stability (matches main transformer)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, draft: Tensor, context: Tensor,
                injection: Tensor | None = None) -> Tensor:
        """Cross+self attention with KV injection.

        Args:
            draft: [B, block_size, D] draft hidden states
            context: [B, T_ctx, D] belief embeddings (may be [B, 0, D])
            injection: [B, T_inject, D] projected tapped target features

        Returns:
            [B, block_size, D] updated draft hidden states
        """
        B, N, D = draft.shape

        # Q from draft only
        q = self.q_proj(draft)

        # K, V from concat(context, draft) — cross-attention to beliefs + self-attention
        kv_input = torch.cat([context, draft], dim=1)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # KV injection: prepend target features as additional KV pairs
        if injection is not None:
            k_inj = self.k_inject(injection)  # [B, T_inject, D]
            v_inj = self.v_inject(injection)  # [B, T_inject, D]
            k = torch.cat([k_inj, k], dim=1)
            v = torch.cat([v_inj, v], dim=1)

        # Reshape for multi-head attention
        T_kv = k.shape[1]
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention (no causal mask — draft tokens see all context + each other)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class DFlashDraftLayer(nn.Module):
    """Single draft layer: cross+self attention with KV injection + MLP."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm_ctx = nn.RMSNorm(hidden_dim)
        self.norm_inject = nn.RMSNorm(hidden_dim)
        self.attn = DFlashCrossAttention(hidden_dim, n_heads)
        self.norm2 = nn.RMSNorm(hidden_dim)
        # Smaller MLP than main model (2x expansion instead of 4x)
        self.fc = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

    def forward(self, draft: Tensor, context: Tensor,
                injection: Tensor | None = None) -> Tensor:
        # Normalize injection per-layer for stable KV projection
        normed_inject = self.norm_inject(injection) if injection is not None else None
        # Cross+self attention with KV injection
        draft = draft + self.attn(self.norm1(draft), self.norm_ctx(context),
                                  injection=normed_inject)
        # MLP with ReLU²
        h = self.fc(self.norm2(draft))
        h = F.relu(h).square()
        draft = draft + self.proj(h)
        return draft


class DFlashDraftHead(nn.Module):
    """Block diffusion draft head with KV injection, streak distillation,
    and adaptive block size.

    Predicts up to max_block_size tokens in parallel from target model hidden
    states and active belief embeddings. Target features are injected directly
    into K/V projections of each draft layer (not concatenated as context tokens).

    Three improvements over baseline block diffusion:
    1. KV injection: per-layer K/V projections for tapped features give tighter
       verifier alignment (+21% acceptance rate)
    2. Streak distillation: position-weighted CE + expected streak bonus optimizes
       consecutive acceptance, not per-token accuracy (+38% speedup)
    3. Adaptive block size: entropy-based cutoff drafts many tokens but verifies
       only the confident prefix (+90% speedup)

    Combined theoretical improvement: ~8.5x over vanilla AR (vs ~3.4x baseline).
    """

    def __init__(self, hidden_dim: int, n_heads: int, n_layers: int,
                 block_size: int, max_block_size: int, n_target_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size          # training window
        self.max_block_size = max_block_size  # inference max (adaptive cutoff)
        self.n_layers = n_layers

        # Compute which target layers to tap (evenly spaced)
        self.tap_indices = self._build_tap_indices(n_target_layers, n_layers)

        # Mask token embedding (draft positions start as this)
        self.mask_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional embedding supports up to max_block_size for adaptive inference
        self.pos_embed = nn.Embedding(max_block_size, hidden_dim)

        # Feature projection: concat of tapped target hidden states → hidden_dim
        # Output is the KV injection signal (not concatenated as context tokens)
        self.feature_proj = nn.Linear(hidden_dim * n_layers, hidden_dim, bias=False)

        # Draft layers (with per-layer KV injection)
        self.layers = nn.ModuleList([
            DFlashDraftLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Output norm (before shared LM head)
        self.out_norm = nn.RMSNorm(hidden_dim)

        # Initialize feature_proj to near-zero so draft starts as no-op
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
    ) -> tuple[Tensor, Tensor]:
        """Extract KV injection signal and belief context.

        Returns two tensors:
        - context: [B, T_ctx, D] belief embeddings for standard cross-attention
          (empty [B, 0, D] if no beliefs — draft uses only self-attention + injection)
        - injection: [B, T_tap, D] projected tapped features for KV injection

        Args:
            layer_hiddens: {layer_idx: [B, T, D]} hidden states at tapped layers
            belief_embeddings: [N_active, D] active belief vectors (optional)

        Returns:
            (context, injection) tuple
        """
        # Collect tapped features
        tapped = []
        for idx in self.tap_indices:
            if idx in layer_hiddens:
                tapped.append(layer_hiddens[idx])

        if not tapped:
            raise ValueError("No target hidden states available at tap indices")

        # Concat along feature dim and project: [B, T, D*n_taps] → [B, T, D]
        combined = torch.cat(tapped, dim=-1)
        injection = self.feature_proj(combined)  # KV injection signal

        # Belief embeddings as standard context tokens (beliefs have variable
        # count and semantic content — better as attention tokens than injection)
        B = injection.shape[0]
        if belief_embeddings is not None and belief_embeddings.shape[0] > 0:
            context = belief_embeddings.unsqueeze(0).expand(B, -1, -1)
        else:
            context = injection.new_zeros(B, 0, self.hidden_dim)

        return context, injection

    def forward(
        self,
        context: Tensor,
        lm_head_fn=None,
        injection: Tensor | None = None,
        draft_length: int | None = None,
    ) -> Tensor:
        """Generate draft logits for variable number of positions.

        Args:
            context: [B, T_ctx, D] belief embeddings (may be [B, 0, D])
            lm_head_fn: callable that maps [B, N, D] → [B, N, vocab]
            injection: [B, T_inject, D] KV injection signal from target features
            draft_length: positions to draft (default: block_size for training)

        Returns:
            [B, draft_length, vocab_size] draft logits
        """
        B = context.shape[0]
        device = context.device
        length = draft_length or self.block_size

        # Initialize draft as mask embeddings + position
        pos_ids = torch.arange(length, device=device)
        draft = self.mask_embed.expand(B, length, -1) + self.pos_embed(pos_ids)

        # Run through draft layers (with KV injection from target features)
        for layer in self.layers:
            draft = layer(draft, context, injection=injection)

        # Apply output norm
        draft = self.out_norm(draft)

        # Project to vocab via shared LM head
        if lm_head_fn is not None:
            return lm_head_fn(draft)
        return draft

    def compute_draft_loss(
        self,
        context: Tensor,
        injection: Tensor,
        target_tokens: Tensor,
        lm_head_fn=None,
        streak_decay: Tensor | None = None,
        streak_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute streak-distilled training loss.

        Two components beyond standard CE:
        1. Position-weighted CE: w_i = decay^i emphasizes early positions.
           Earlier positions matter more because speculative decoding stops
           at the first mismatch — high accuracy at position 5 is worthless
           if position 2 is wrong.
        2. Expected streak bonus: directly maximizes P(consecutive correct).
           Complementary to weighted CE — CE shapes per-position logits,
           streak bonus shapes the joint probability of consecutive matches.

        Reference: SpecDiff-2 (arXiv:2511.00606) — streak distillation

        Args:
            context: [B, T_ctx, D] belief context
            injection: [B, T_inject, D] KV injection signal
            target_tokens: [B, block_size] ground truth tokens
            lm_head_fn: shared LM head function
            streak_decay: per-position weight decay λ ∈ (0,1) — learned MetaParam
            streak_weight: weight on expected streak bonus — learned MetaParam

        Returns:
            scalar loss (differentiable w.r.t. streak_decay and streak_weight)
        """
        logits = self.forward(context, lm_head_fn, injection=injection)
        B, S, V = logits.shape

        if streak_decay is not None:
            # Position-weighted CE (streak distillation core)
            positions = torch.arange(S, device=logits.device, dtype=logits.dtype)
            weights = streak_decay ** positions  # [S] — early positions weighted higher

            per_pos_loss = F.cross_entropy(
                logits.reshape(-1, V), target_tokens.reshape(-1),
                ignore_index=-1, reduction='none',
            ).view(B, S)

            loss = (per_pos_loss * weights).sum(dim=-1).mean() / weights.sum()
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, V), target_tokens.reshape(-1),
                ignore_index=-1,
            )

        # Expected streak length bonus
        if streak_weight is not None:
            sw_val = streak_weight.item() if isinstance(streak_weight, Tensor) else streak_weight
            if sw_val > 0:
                probs = F.softmax(logits, dim=-1)  # [B, S, V]
                p_correct = probs.gather(
                    -1, target_tokens.clamp(min=0).unsqueeze(-1),
                ).squeeze(-1)  # [B, S]
                # Treat ignore_index positions as P=1 (don't break streak)
                valid = (target_tokens != -1).float()
                p_correct = p_correct * valid + (1.0 - valid)
                p_correct = p_correct.clamp(min=1e-8)

                # Cumulative product = P(streak ≥ i+1)
                cum_correct = p_correct.cumprod(dim=-1)  # [B, S]
                # Expected streak = Σ P(streak ≥ i)
                expected_streak = (cum_correct * valid).sum(dim=-1).mean()

                # Negative: maximize streak (minimize loss)
                loss = loss - streak_weight * expected_streak / S

        return loss

    def compute_entropy(self, logits: Tensor) -> Tensor:
        """Per-position entropy for adaptive block sizing.

        High entropy = uncertain draft = likely rejection by verifier.
        Used by spec_generate to decide verification cutoff.

        Args:
            logits: [B, draft_length, vocab] draft logits

        Returns:
            [B, draft_length] entropy in nats
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)
