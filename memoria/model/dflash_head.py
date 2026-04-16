"""DFlash draft head: block diffusion for speculative decoding.

Generates block_size tokens in parallel via cross-attention to the target
model's intermediate hidden states + active belief embeddings. The full
model then verifies the block in one pass (including refinement loops).

Why this helps MemoriaModel specifically:
- Refinement loops make per-token generation ~4x more expensive
- Speculative decoding amortizes that cost across accepted blocks
- Cognitive state as additional cross-attention context improves drafts

Five improvements over baseline block diffusion:
1. KV injection (DFlash arXiv:2602.06036): per-layer K/V projections for
   tapped target features → tighter alignment with verifier than concat
2. Streak distillation (SpecDiff-2 arXiv:2511.00606): position-weighted CE
   + expected streak bonus → optimizes consecutive acceptance, not per-token
3. Adaptive block size (FailFast arXiv:2512.20573): entropy-based cutoff
   → draft max_block_size, verify only confident prefix
4. OPUT on-policy training (DMax arXiv:2604.08302): train draft head on its
   own (possibly wrong) predictions via SPD hybrid embeddings. Teaches
   self-correction within a block → higher streak length at zero inference cost.
5. DDTree-aware training (DDTree, Ringel & Romano 2026): tree position blend
   (decay↔uniform), prefix mass bonus (DDTree surrogate), top-K recall penalty.
   Optimizes draft quality for multi-branch tree verification, not just single path.

Architecture:
- Mask token embeddings + positional encoding for draft positions
- N cross-attention layers: Q=draft, KV=concat(beliefs, draft) + KV-injected target features
- KV injection: per-layer K/V projections for tapped hidden states
- Injection projections marked for 3-bit RotorQuant (draft accuracy < verifier)
- Draft layer weights marked for 4-bit (attention) / 3-bit (MLP) RotorQuant
- Shared LM head with main model (no extra 117M params)
- Single-shot parallel prediction (no iterative diffusion at inference)

Reference: DFlash (Chen, Liang, Liu — arXiv:2602.06036) — KV injection
Reference: SpecDiff-2 (Sandler et al. — arXiv:2511.00606) — streak distillation
Reference: FailFast (Pan, Chen, Netravali — arXiv:2512.20573) — adaptive block size
Reference: DEER (Cheng et al. — arXiv:2512.15176) — single-step diffusion drafting
Reference: Speculative Decoding (Leviathan et al., ICML 2023)
Reference: DMax (Chen et al. — arXiv:2604.08302) — OPUT + SPD self-correction
Reference: DDTree (Ringel & Romano — liranringel.github.io/ddtree/) — draft tree
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
        # Attention projections: 4-bit RotorQuant (same as backbone attention)
        # OPUT noise-robustness training makes draft head safe to quantize
        self.q_proj._qat_bits = 4
        self.k_proj._qat_bits = 4
        self.v_proj._qat_bits = 4
        self.out_proj._qat_bits = 4

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
        # MLP: 3-bit RotorQuant (same as backbone MLP — MLPs tolerate aggressive quantization)
        self.fc._qat_bits = 3
        self.proj._qat_bits = 3

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
                 block_size: int, max_block_size: int, n_target_layers: int,
                 belief_dim: int | None = None):
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

        # Belief → hidden projection. Active belief vectors live in
        # `belief_dim` space (typically 256), but the draft layers'
        # cross-attention context norm (`DFlashDraftLayer.norm_ctx`) expects
        # `hidden_dim` (typically 768). This Linear brings beliefs into the
        # draft head's residual space before they're used as attention
        # context tokens. Created only when belief_dim differs from
        # hidden_dim — if they happen to match, we use beliefs directly.
        # Near-zero init (std=0.01) means on a fresh model the belief
        # context starts as ~zero vectors, so the draft head trains
        # belief-free first and gradually learns to use them — matches
        # the pattern of `feature_proj` init above.
        if belief_dim is not None and belief_dim != hidden_dim:
            self.belief_to_hidden = nn.Linear(belief_dim, hidden_dim, bias=False)
            nn.init.normal_(self.belief_to_hidden.weight, std=0.01)
            # 4-bit RotorQuant to match the rest of the draft head
            self.belief_to_hidden._qat_bits = 4
        else:
            self.belief_to_hidden = None

        # Draft layers (with per-layer KV injection)
        self.layers = nn.ModuleList([
            DFlashDraftLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Output norm (before shared LM head)
        self.out_norm = nn.RMSNorm(hidden_dim)

        # Initialize feature_proj to near-zero so draft starts as no-op
        nn.init.normal_(self.feature_proj.weight, std=0.01)
        # 4-bit RotorQuant (projection, same as backbone attention)
        self.feature_proj._qat_bits = 4

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
        # Collect tapped features — all tap indices must be present
        tapped = []
        missing = [idx for idx in self.tap_indices if idx not in layer_hiddens]
        if missing:
            raise ValueError(
                f"Missing tapped layers {missing}. "
                f"Available: {sorted(layer_hiddens.keys())}, expected: {self.tap_indices}"
            )
        for idx in self.tap_indices:
            tapped.append(layer_hiddens[idx])

        # Concat along feature dim and project: [B, T, D*n_taps] → [B, T, D]
        combined = torch.cat(tapped, dim=-1)
        injection = self.feature_proj(combined)  # KV injection signal

        # Belief embeddings as standard context tokens (beliefs have variable
        # count and semantic content — better as attention tokens than
        # injection). Beliefs live in `belief_dim` space; if that differs from
        # the draft head's `hidden_dim`, project them through
        # `self.belief_to_hidden` first. Without this projection we'd feed
        # [N_active, belief_dim] into a norm expecting [*, hidden_dim] and
        # crash — which is exactly what happened when the previous
        # `.active_beliefs` typo was fixed without also introducing this
        # projection, because the typo had silently disabled this code path
        # by turning the attribute access into an AttributeError.
        B = injection.shape[0]
        if belief_embeddings is not None and belief_embeddings.shape[0] > 0:
            if self.belief_to_hidden is not None:
                belief_embeddings = self.belief_to_hidden(belief_embeddings)
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

    def _streak_loss(
        self,
        logits: Tensor,
        target_tokens: Tensor,
        streak_decay: Tensor | None,
        streak_weight: Tensor | None,
        ddtree_position_blend: Tensor | None = None,
    ) -> Tensor:
        """Position-weighted CE + expected streak bonus.

        Shared between teacher pass and OPUT self-correction pass.

        When ddtree_position_blend is provided, the position weights are
        interpolated between exponential decay (DFlash-optimal: early
        positions matter most for single-trajectory acceptance) and uniform
        (DDTree-optimal: all positions matter because tree branches recover
        from early mismatches). The model learns the optimal blend.
        """
        B, S, V = logits.shape

        if streak_decay is not None:
            positions = torch.arange(S, device=logits.device, dtype=logits.dtype)
            decay_weights = streak_decay ** positions

            # DDTree position blend: interpolate decay ↔ uniform
            # At blend=0: pure exponential decay (DFlash single-path optimal)
            # At blend=1: uniform weights (DDTree tree-path optimal)
            if ddtree_position_blend is not None:
                uniform_weights = torch.ones_like(decay_weights)
                weights = (
                    ddtree_position_blend * uniform_weights
                    + (1.0 - ddtree_position_blend) * decay_weights
                )
            else:
                weights = decay_weights

            per_pos_loss = F.cross_entropy(
                logits.reshape(-1, V), target_tokens.reshape(-1),
                ignore_index=-1, reduction='none',
            ).view(B, S)

            valid_mask = (target_tokens != -1).float()
            loss = (per_pos_loss * weights).sum(dim=-1) / (weights * valid_mask).sum(dim=-1).clamp(min=1e-8)
            loss = loss.mean()
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, V), target_tokens.reshape(-1),
                ignore_index=-1,
            )

        if streak_weight is not None:
            sw_val = streak_weight.item() if isinstance(streak_weight, Tensor) else streak_weight
            if sw_val > 0:
                probs = F.softmax(logits.float(), dim=-1)
                p_correct = probs.gather(
                    -1, target_tokens.clamp(min=0).unsqueeze(-1),
                ).squeeze(-1)
                valid = (target_tokens != -1).float()
                p_correct = p_correct * valid + (1.0 - valid)
                p_correct = p_correct.clamp(min=1e-8)

                cum_correct = p_correct.cumprod(dim=-1)
                expected_streak = (cum_correct * valid).sum(dim=-1).mean()

                loss = loss - streak_weight * expected_streak / S

        return loss

    def _tree_prefix_mass_bonus(
        self,
        logits: Tensor,
        target_tokens: Tensor,
    ) -> Tensor:
        """Tree prefix mass bonus: uniform-weighted cumulative product of P(correct).

        Directly optimizes the DDTree surrogate objective: the probability that
        the target continuation appears in the draft tree. Unlike the decayed
        streak bonus (which emphasizes early positions for single-path acceptance),
        this uses UNIFORM weights — all depths matter equally because the tree
        verifies multiple branches independently.

        The DDTree surrogate is (Proposition 1):
            E_Q[α_T(Y)] = ∑_{u ∈ T} q(u|c,b) = ∑_{u ∈ T} ∏_i q_i(u_i|c,b)
        Training maximizes q(target|c,b) = ∏_i q_i(target_i|c,b), the prefix
        probability of the ground truth under the factorized draft distribution.
        This ensures the target path has high probability and thus survives
        the budget cutoff in Algorithm 1.

        Args:
            logits: [B, S, V] draft logits (before softmax).
            target_tokens: [B, S] target token IDs (-1 = padding).

        Returns:
            scalar: negative mean prefix mass (to be minimized as a bonus).
        """
        B, S, _ = logits.shape
        probs = F.softmax(logits.float(), dim=-1)
        p_correct = probs.gather(
            -1, target_tokens.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)  # [B, S]

        valid = (target_tokens != -1).float()
        p_correct = p_correct * valid + (1.0 - valid)
        p_correct = p_correct.clamp(min=1e-8)

        # Cumulative product: prefix probability at each depth
        # No decay — uniform weighting across all positions (tree-optimal)
        prefix_probs = p_correct.cumprod(dim=-1)  # [B, S]

        # Bonus = mean prefix probability across positions and batch
        # Negated because this is subtracted from the total loss
        prefix_mass = (prefix_probs * valid).sum(dim=-1).mean() / max(S, 1)
        return -prefix_mass

    def compute_draft_loss(
        self,
        context: Tensor,
        injection: Tensor,
        target_tokens: Tensor,
        lm_head_fn=None,
        embed_fn=None,
        streak_decay: Tensor | None = None,
        streak_weight: Tensor | None = None,
        oput_weight: Tensor | None = None,
        ddtree_position_blend: Tensor | None = None,
        ddtree_prefix_weight: Tensor | None = None,
        ddtree_recall_weight: Tensor | None = None,
        ddtree_train_budget: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Compute streak-distilled training loss with OPUT self-correction
        and DDTree-aware training terms.

        Six components:
        1. Position-weighted CE: w_i = blend(decay^i, 1.0) emphasizes positions
           according to learned decay↔uniform blend. Pure decay = DFlash-optimal,
           uniform = DDTree-optimal where tree branches recover from early mismatches.
        2. Expected streak bonus: directly maximizes P(consecutive correct).
           Complementary to weighted CE — CE shapes per-position logits,
           streak bonus shapes the joint probability of consecutive matches.
        3. OPUT self-correction (DMax arXiv:2604.08302): sample from the draft
           head's own predictions, construct SPD hybrid embeddings (interpolation
           between predicted token embedding and mask embedding weighted by
           confidence), run a second forward pass, and train to still produce
           the correct output. Teaches the draft head to recover from its own
           errors within a block — directly addresses error accumulation in
           parallel decoding. Training-only, zero inference cost.
        4. Tree prefix mass bonus (DDTree): uniform-weighted cumprod of P(correct)
           directly optimizes the DDTree surrogate ∑ q(u|c,b), ensuring the
           target path has high prefix probability and survives budget cutoff.
        5. Top-K recall penalty (DDTree): penalizes positions where the target
           token falls outside the draft's top-K. These are "tree misses" where
           DDTree cannot accept the correct token regardless of budget.

        Reference: SpecDiff-2 (arXiv:2511.00606) — streak distillation
        Reference: DMax (Chen et al. — arXiv:2604.08302) — OPUT + SPD
        Reference: DDTree (Ringel & Romano) — tree surrogate, top-K recall

        Args:
            context: [B, T_ctx, D] belief context
            injection: [B, T_inject, D] KV injection signal
            target_tokens: [B, block_size] ground truth tokens
            lm_head_fn: shared LM head function (hidden → logits)
            embed_fn: embedding function (token_ids → embeddings) for OPUT
            streak_decay: per-position weight decay λ ∈ (0,1) — learned MetaParam
            streak_weight: weight on expected streak bonus — learned MetaParam
            oput_weight: weight on self-correction loss — learned MetaParam
            ddtree_position_blend: blend between decay/uniform weights — learned MetaParam
            ddtree_prefix_weight: weight on tree prefix mass bonus — learned MetaParam
            ddtree_recall_weight: weight on top-K recall penalty — learned MetaParam
            ddtree_train_budget: tree budget for top-K recall computation (0 = disabled)

        Returns:
            (total_loss, loss_self_correct) — both scalar tensors
        """
        # ── Pass 1: teacher-forced (streak distillation + tree-aware blend) ──
        logits = self.forward(context, lm_head_fn, injection=injection)
        loss_teacher = self._streak_loss(
            logits, target_tokens, streak_decay, streak_weight,
            ddtree_position_blend=ddtree_position_blend,
        )

        # ── DDTree term 1: tree prefix mass bonus ──
        if ddtree_prefix_weight is not None:
            pw_val = ddtree_prefix_weight.item() if isinstance(ddtree_prefix_weight, Tensor) else ddtree_prefix_weight
            if pw_val > 0:
                tree_prefix_bonus = self._tree_prefix_mass_bonus(logits, target_tokens)
                loss_teacher = loss_teacher + ddtree_prefix_weight * tree_prefix_bonus

        # ── DDTree term 2: top-K recall penalty ──
        if ddtree_recall_weight is not None and ddtree_train_budget > 0:
            rw_val = ddtree_recall_weight.item() if isinstance(ddtree_recall_weight, Tensor) else ddtree_recall_weight
            if rw_val > 0:
                from .ddtree import compute_tree_top_k_for_training
                # K = min(budget, vocab): Lemma 1 guarantees optimal tree uses only top-K
                effective_k = min(ddtree_train_budget, logits.shape[-1])
                recall_loss = compute_tree_top_k_for_training(
                    logits, target_tokens, effective_k,
                )
                loss_teacher = loss_teacher + ddtree_recall_weight * recall_loss

        # ── Pass 2: OPUT on-policy self-correction (DMax) ──
        loss_self_correct = torch.tensor(0.0, device=logits.device)
        oput_w = oput_weight.item() if isinstance(oput_weight, Tensor) else (oput_weight or 0.0)
        if oput_w > 0 and embed_fn is not None:
            with torch.no_grad():
                # Sample from draft's own predictions (on-policy rollout)
                sampled_ids = logits.argmax(dim=-1)  # [B, S] — greedy
                sampled_embeds = embed_fn(sampled_ids)  # [B, S, hidden_dim]

                # Per-position confidence from first pass
                confidence = F.softmax(logits, dim=-1).max(dim=-1).values  # [B, S]

                # SPD hybrid embedding (DMax Eq. 9-10): interpolate between
                # predicted token embedding and mask embedding weighted by
                # confidence. Low confidence → mostly mask → explicit uncertainty.
                pi = confidence.unsqueeze(-1)  # [B, S, 1]
                mask_expanded = self.mask_embed.expand_as(sampled_embeds)

                # Unnormalized hybrid (DMax Eq. 9)
                hybrid_raw = pi * sampled_embeds + (1.0 - pi) * mask_expanded

                # Norm-preserving renormalization (DMax Eq. 10):
                # scale to match weighted sum of component norms
                target_norm = (
                    pi * sampled_embeds.norm(dim=-1, keepdim=True)
                    + (1.0 - pi) * mask_expanded.norm(dim=-1, keepdim=True)
                )
                hybrid_norm = hybrid_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hybrid = hybrid_raw * (target_norm / hybrid_norm)

            # Positional encoding OUTSIDE no_grad so pos_embed gets OPUT gradients
            pos_ids = torch.arange(hybrid.shape[1], device=hybrid.device)
            hybrid = hybrid + self.pos_embed(pos_ids)

            # Second forward pass through draft layers with hybrid input
            # (gradients flow through draft layers + pos_embed, not through hybrid construction)
            draft = hybrid
            for layer in self.layers:
                draft = layer(draft, context, injection=injection)
            draft = self.out_norm(draft)
            if lm_head_fn is not None:
                refined_logits = lm_head_fn(draft)
            else:
                refined_logits = draft

            # OPUT loss also gets tree-aware position blend
            loss_self_correct = self._streak_loss(
                refined_logits, target_tokens, streak_decay, streak_weight,
                ddtree_position_blend=ddtree_position_blend,
            )

        if oput_weight is not None and oput_w > 0:
            # Detach oput_weight: the gradient d(L)/d(w) = loss_self_correct
            # is structurally always positive (CE > 0 × softplus' > 0), so
            # gradient descent monotonically decreases the weight to zero.
            # Detaching breaks this degenerate gradient while preserving the
            # forward-pass scaling. The weight still adapts via the MetaParam
            # optimizer group (cognitive_meta_lr) responding to the overall
            # loss landscape, just not through the direct d(w*L)/dw path.
            total_loss = loss_teacher + oput_weight.detach() * loss_self_correct
        else:
            total_loss = loss_teacher
        return total_loss, loss_self_correct

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
