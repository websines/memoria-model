"""In-Place Test-Time Training: live gradient updates at inference time.

This is the core self-improvement mechanism. During BOTH training and inference:
1. Process a chunk of tokens through the model
2. Compute next-token prediction loss on that chunk
3. Take a gradient step on designated fast-weight parameters
4. Those updated weights PERSIST across chunks and sessions

The model literally gets better at predicting what this user says next,
every time they interact with it. This is what makes 500M+experience > 10B.

What gets updated at inference time:
- MLP.c_proj fast-weight deltas (backbone adaptation)
- Belief vectors (world model refinement)
- Edge weights and relations (structural learning)

What stays frozen at inference time:
- Attention weights, embeddings, LM head (core language capability)
- Interface layer weights (read/write projections)
- EdgeProposer, TelosModule (structural policies)

Reference: In-Place TTT (ByteDance/PKU, ICLR 2026)
Reference: TTT-E2E (NVIDIA/Stanford, arXiv:2512.23675) — meta-learned delta init, 25% of MLP layers as mutable
Reference: Titans (Google, arXiv:2501.00663) — momentum-based surprise smoothing
Reference: LaCT (ICLR 2026, arXiv:2505.23884) — large-chunk TTT batching for GPU utilization
Reference: DeltaProduct (NeurIPS 2025, arXiv:2502.10297) — multi-step inner loop for expressiveness
Reference: RWKV-7 — per-belief adaptive learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InPlaceTTT(nn.Module):
    """Live test-time training on MLP projection layers.

    For each designated layer, maintains a persistent low-rank delta:
        W_effective = W_frozen + delta_A @ delta_B

    At each forward pass (training AND inference), after computing the
    output with current deltas, takes a gradient step on the deltas using
    next-token prediction loss. The deltas persist and accumulate changes
    across interactions.

    This is NOT an adapter that's fixed after training. The deltas update
    EVERY TIME the model processes tokens. The model improves with use.

    Improvements over baseline:
    - Meta-learned initialization (TTT-E2E, arXiv:2512.23675): deltas start from
      a learned warm point instead of zero.
    - Momentum-based surprise smoothing (Titans, arXiv:2501.00663): adaptive
      gating thresholds derived from running variance instead of fixed ratios.
    - Large-chunk gradient accumulation (LaCT, arXiv:2505.23884): accumulate
      gradients over multiple chunks before stepping for better GPU utilization.
    - Multi-step inner loop (DeltaProduct, arXiv:2502.10297): N smaller steps
      per chunk for more expressive state tracking.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        ttt_layers: list[int] | None = None,
        rank: int = 32,
        ttt_lr: float = 0.001,
        ttt_accum_steps: int = 1,
        ttt_inner_steps: int = 1,
    ):
        """
        Args:
            hidden_dim: model hidden dimension
            n_layers: total number of transformer layers
            ttt_layers: which layer indices get TTT deltas (default: top 25%)
            rank: low-rank dimension for delta matrices
            ttt_lr: base learning rate (overridden per-layer by log_step_size)
            ttt_accum_steps: accumulate gradients over this many chunks before
                stepping. 1 = no accumulation (original behaviour).
                (LaCT, arXiv:2505.23884)
            ttt_inner_steps: gradient steps per chunk. 1 = single step (original).
                Step size is divided by ttt_inner_steps so total update magnitude
                is comparable. (DeltaProduct, arXiv:2502.10297)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.ttt_lr = ttt_lr
        self.ttt_accum_steps: int = ttt_accum_steps
        self.ttt_inner_steps: int = ttt_inner_steps

        # Default: top 25% of layers (most abstract representations)
        if ttt_layers is None:
            n_ttt = max(1, n_layers // 4)
            ttt_layers = list(range(n_layers - n_ttt, n_layers))
        self.ttt_layers = set(ttt_layers)
        self._ttt_layer_list = sorted(ttt_layers)

        # ── Per-layer persistent low-rank deltas ──
        # These are NOT nn.Parameters in the sense that the main optimizer
        # trains them. They ARE registered as parameters so they're saved/
        # loaded with the model. The TTT inner loop updates them via .data.
        self.delta_A = nn.ParameterDict()
        self.delta_B = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            self.delta_A[key] = nn.Parameter(
                torch.zeros(hidden_dim, rank), requires_grad=False,
            )
            self.delta_B[key] = nn.Parameter(
                torch.zeros(rank, hidden_dim), requires_grad=False,
            )

        # ── Meta-learned initialization (TTT-E2E, arXiv:2512.23675) ──
        # Instead of always cold-starting deltas at zero, learn an initialization
        # point that is optimal as a warm start. These ARE trained by the main
        # optimizer (requires_grad=True), unlike the deltas themselves.
        self.init_A = nn.ParameterDict()
        self.init_B = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            self.init_A[key] = nn.Parameter(torch.zeros(hidden_dim, rank))
            self.init_B[key] = nn.Parameter(torch.zeros(rank, hidden_dim))

        # Learnable step-size per layer (trained during training, fixed at inference)
        # Controls how much each layer adapts per chunk
        self.log_step_size = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            # Init: log(0.001) ≈ -6.9 → step_size ≈ 0.001
            self.log_step_size[key] = nn.Parameter(torch.tensor(-6.9))

        # ── Titans-style learned adaptive decay gate (arXiv:2501.00663) ──
        # Decides how much of the previous delta to retain each step.
        # The decay gate is the single largest contributor to Titans performance.
        # sigmoid(2.0) ≈ 0.88 — default: mostly preserve deltas.
        self.decay_gate = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.decay_gate.bias, 2.0)
        self._last_decay_alpha: float = 1.0

        # ── Surprise tracking (Titans-inspired, arXiv:2501.00663) ──
        # Short-term EMA tracks recent surprise.
        # Long-term momentum tracks the overall surprise level.
        # Variance enables adaptive thresholds instead of fixed 0.1/3.0 ratios.
        # Cold-start at 0.0: all updates accepted for first ~100 steps
        # (no OOD protection). Intentional for scratch training where all data is novel.
        self._surprise_ema: float = 0.0         # short-term EMA (fast)
        self._surprise_momentum: float = 0.0    # long-term momentum (slow)
        self._surprise_variance: float = 1.0    # variance of surprise deviation
        self._momentum_decay: float = 0.999     # slower decay → long-term view
        self._variance_decay: float = 0.99

        self._last_ttt_accepted: bool = True

        # ── Large-chunk gradient accumulation buffers (LaCT) ──
        self._grad_accum_A: dict[str, Tensor] = {}
        self._grad_accum_B: dict[str, Tensor] = {}
        self._accum_count: dict[str, int] = {}  # per-layer accumulation counters

    # ── Meta-learned init ──────────────────────────────────────────────────────

    def reset_deltas(self):
        """Reset deltas to the meta-learned initialization (not zero).

        Call at the start of a new session or document so the model begins
        from its learned warm-start point rather than cold zero.

        Reference: TTT-E2E (NVIDIA/Stanford, arXiv:2512.23675).
        """
        with torch.no_grad():
            for key in self.delta_A:
                self.delta_A[key].data.copy_(self.init_A[key].data)
                self.delta_B[key].data.copy_(self.init_B[key].data)
                # LaCT-style L2 normalization on reset (arXiv:2505.23884)
                self.delta_A[key].data = F.normalize(self.delta_A[key].data, dim=0)
                self.delta_B[key].data = F.normalize(self.delta_B[key].data, dim=0)

    def apply_decay(self, hidden: Tensor, layer_key: str | None = None):
        """Apply learned Titans-style decay to deltas before TTT step.

        The decay gate learns when to forget vs retain delta adaptations.
        High alpha (near 1) = preserve deltas. Low alpha (near 0) = forget.

        Args:
            hidden: hidden states for computing decay gate
            layer_key: if provided, only decay this layer's deltas.
                If None, decay all layers (for backward compat).

        Reference: Titans (Google, arXiv:2501.00663) — learned decay is the
        single largest contributor to Titans performance.
        """
        with torch.no_grad():
            pooled = hidden.mean(dim=(0, 1))  # [D]
            alpha = torch.sigmoid(self.decay_gate(pooled))  # scalar in (0, 1)
            if layer_key is not None:
                if layer_key in self.delta_A:
                    self.delta_A[layer_key].data.mul_(alpha)
                    self.delta_B[layer_key].data.mul_(alpha)
            else:
                for key in self.delta_A:
                    self.delta_A[key].data.mul_(alpha)
                    self.delta_B[key].data.mul_(alpha)
            self._last_decay_alpha = alpha.item()
        return alpha.item()

    # ── Surprise gating ───────────────────────────────────────────────────────

    def update_surprise_ema(self, surprise: float):
        """Track running mean with momentum-based smoothing (Titans-inspired).

        Short-term EMA tracks recent surprise.
        Long-term momentum tracks overall surprise level.
        Variance enables adaptive thresholds instead of fixed 0.1/3.0.

        Reference: Titans (Google, arXiv:2501.00663).

        Args:
            surprise: RND surprise value for the current input
        """
        # Short-term EMA (fast, α = 0.01)
        self._surprise_ema = 0.99 * self._surprise_ema + 0.01 * surprise
        # Long-term momentum (slow, 1 - α = 0.001)
        self._surprise_momentum = (
            self._momentum_decay * self._surprise_momentum
            + (1.0 - self._momentum_decay) * surprise
        )
        # Variance tracking: deviation from short-term EMA
        deviation = surprise - self._surprise_ema
        self._surprise_variance = (
            self._variance_decay * self._surprise_variance
            + (1.0 - self._variance_decay) * deviation * deviation
        )

    def should_update(self, surprise: float) -> bool:
        """Gate TTT updates using momentum-smoothed surprise (Titans-inspired).

        Instead of fixed 0.1/3.0 ratios, use adaptive thresholds based on
        the running variance of surprise. Updates proceed when surprise is
        within 2 standard deviations of the short-term mean.

        Too surprising (OOD) → skip to protect deltas from corruption.
        Too boring (fully predicted) → skip to save compute.
        Sweet spot: moderate surprise means the input is learnable.

        Reference: Titans (Google, arXiv:2501.00663).

        Args:
            surprise: mean RND surprise for the current input's beliefs

        Returns:
            True if the TTT step should proceed
        """
        mean = self._surprise_ema
        if mean < 1e-8:
            return True  # no history yet, allow all updates

        std = max(self._surprise_variance ** 0.5, 1e-8)
        # Adaptive thresholds: mean ± 2*std, floored at 5% of mean
        lower = max(mean - 2.0 * std, mean * 0.05)
        upper = mean + 2.0 * std

        return lower < surprise < upper

    # ── Delta application ─────────────────────────────────────────────────────

    def is_ttt_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.ttt_layers

    def apply_delta(self, layer_idx: int, hidden: Tensor) -> Tensor:
        """Apply the current persistent delta to hidden states.

        This runs during the normal forward pass.
        W_effective @ x = W_frozen @ x + (A @ B) @ x

        Args:
            hidden: [B, T, hidden_dim] post-MLP hidden states

        Returns:
            hidden + delta contribution
        """
        key = str(layer_idx)
        if key not in self.delta_A:
            return hidden

        A = self.delta_A[key]  # [hidden_dim, rank]
        B = self.delta_B[key]  # [rank, hidden_dim]

        # Efficient: hidden @ B^T @ A^T = hidden @ (A @ B)^T
        delta_out = F.linear(F.linear(hidden, B), A)
        return hidden + delta_out

    # ── TTT gradient step ─────────────────────────────────────────────────────

    @torch.no_grad()
    def ttt_step(
        self,
        layer_idx: int,
        hidden_pre_delta: Tensor,
        targets: Tensor,
        lm_head_weight: Tensor,
        vocab_size: int,
    ):
        """Take a TTT gradient step: update the persistent delta.

        This is the self-improvement step. It runs at BOTH training and inference.
        Uses next-token prediction loss on the current chunk as the objective.
        Includes quality gating: snapshots deltas before update, rolls back if
        the update increased loss (bad input protection).

        The gradient is computed analytically for the low-rank delta,
        avoiding full backward() through the transformer.

        Enhancements over baseline:
        - Large-chunk accumulation (LaCT, arXiv:2505.23884): accumulate
          gradients over ttt_accum_steps chunks before applying. Set
          ttt_accum_steps=1 to restore original single-step behaviour.
        - Multi-step inner loop (DeltaProduct, arXiv:2502.10297): take
          ttt_inner_steps gradient steps of size step_size/ttt_inner_steps.
          Set ttt_inner_steps=1 to restore original single-step behaviour.

        Args:
            layer_idx: which layer's delta to update
            hidden_pre_delta: [B, T, D] hidden states BEFORE delta was applied
            targets: [B, T] target token ids
            lm_head_weight: [vocab_size, D] LM head weight matrix
            vocab_size: vocabulary size
        """
        key = str(layer_idx)
        if key not in self.delta_A:
            return

        A = self.delta_A[key]  # [D, R]
        B = self.delta_B[key]  # [R, D]
        step_size = self.log_step_size[key].exp().item()

        D = self.hidden_dim
        h = hidden_pre_delta  # [B, T, D]

        # ── Snapshot deltas for rollback (BEFORE decay) ──
        A_snapshot = A.data.clone()
        B_snapshot = B.data.clone()

        # Apply learned decay to THIS layer only (Titans-style, arXiv:2501.00663)
        self.apply_decay(hidden_pre_delta, layer_key=key)

        # Current output with delta: h' = h + h @ B^T @ A^T
        h_delta = h + F.linear(F.linear(h, B), A)  # [B, T, D]

        # Sample token positions to keep cost bounded
        T_size = h.shape[1]
        max_positions = min(128, T_size)
        if T_size > max_positions:
            indices = torch.randperm(T_size, device=h.device)[:max_positions]
            h_sample = h_delta[:, indices]
            t_sample = targets[:, indices]
            h_for_grad = h[:, indices]
        else:
            h_sample = h_delta
            t_sample = targets
            h_for_grad = h

        # ── Compute loss BEFORE update (for rollback check) ──
        logits = F.linear(h_sample, lm_head_weight)
        valid = t_sample >= 0
        if not valid.any():
            return
        loss_before = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            t_sample.reshape(-1),
            ignore_index=-1,
            reduction='mean',
        ).item()

        # ── Shared setup for gradient computation ──
        probs = F.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(2, t_sample.clamp(min=0).unsqueeze(-1), 1.0)
        mask = valid.unsqueeze(-1).float()

        # ── Multi-step inner loop (DeltaProduct, arXiv:2502.10297) ──
        # Take ttt_inner_steps smaller gradient steps instead of one large step.
        # Dividing step_size keeps total update magnitude comparable to single step.
        # Set ttt_inner_steps=1 to reproduce original single-step behaviour.
        inner_step_size = step_size / max(self.ttt_inner_steps, 1)

        if self.ttt_inner_steps <= 1:
            # Original single-step path — compute gradient once
            grad_logits = (probs - one_hot) * mask / max(valid.sum().item(), 1)
            grad_h = grad_logits @ lm_head_weight  # [B, max_pos, D]

            h_flat = h_for_grad.reshape(-1, D)
            grad_flat = grad_h.reshape(-1, D)

            hB = h_flat @ B.T  # [N, R]
            grad_A = grad_flat.T @ hB
            grad_B = A.T @ (grad_flat.T @ h_flat)

            # ── Large-chunk accumulation (LaCT, arXiv:2505.23884) ──
            # Accumulate gradients over ttt_accum_steps chunks; apply once.
            # Set ttt_accum_steps=1 to reproduce original immediate-step behaviour.
            if self.ttt_accum_steps > 1:
                if key not in self._grad_accum_A:
                    self._grad_accum_A[key] = torch.zeros_like(grad_A)
                    self._grad_accum_B[key] = torch.zeros_like(grad_B)
                self._grad_accum_A[key] += grad_A
                self._grad_accum_B[key] += grad_B
                self._accum_count[key] = self._accum_count.get(key, 0) + 1

                if self._accum_count[key] < self.ttt_accum_steps:
                    # Not enough chunks accumulated for this layer — defer
                    return

                # Average accumulated gradients and apply
                grad_A = self._grad_accum_A[key] / self.ttt_accum_steps
                grad_B = self._grad_accum_B[key] / self.ttt_accum_steps
                self._grad_accum_A[key].zero_()
                self._grad_accum_B[key].zero_()
                self._accum_count[key] = 0

            A.data -= inner_step_size * grad_A
            B.data -= inner_step_size * grad_B
            # Clamp delta magnitude to prevent explosion (replaces LaCT L2 normalization
            # which made log_step_size ineffective for magnitude control)
            A.data.clamp_(-2.0, 2.0)
            B.data.clamp_(-2.0, 2.0)

        else:
            # Multi-step path: recompute gradient at each updated position
            for _inner in range(self.ttt_inner_steps):
                h_delta_curr = h_for_grad + F.linear(F.linear(h_for_grad, B), A)
                logits_curr = F.linear(h_delta_curr, lm_head_weight)
                probs_curr = F.softmax(logits_curr, dim=-1)
                grad_logits_curr = (probs_curr - one_hot) * mask / max(valid.sum().item(), 1)
                grad_h_curr = grad_logits_curr @ lm_head_weight

                h_flat_curr = h_for_grad.reshape(-1, D)
                grad_flat_curr = grad_h_curr.reshape(-1, D)

                hB_curr = h_flat_curr @ B.T           # [N, R]
                grad_A_inner = grad_flat_curr.T @ hB_curr
                grad_B_inner = A.T @ (grad_flat_curr.T @ h_flat_curr)

                A.data -= inner_step_size * grad_A_inner
                B.data -= inner_step_size * grad_B_inner
                # LaCT-style L2 normalization (arXiv:2505.23884)
                A.data = F.normalize(A.data, dim=0)
                B.data = F.normalize(B.data, dim=0)

        # ── Rollback check: did the update help? ──
        h_delta_new = h_for_grad + F.linear(F.linear(h_for_grad, B), A)
        logits_new = F.linear(h_delta_new, lm_head_weight)
        loss_after = F.cross_entropy(
            logits_new.reshape(-1, vocab_size),
            t_sample.reshape(-1),
            ignore_index=-1,
            reduction='mean',
        ).item()

        # If loss increased, this input made things worse — roll back
        if loss_after > loss_before:
            A.data.copy_(A_snapshot)
            B.data.copy_(B_snapshot)
            self._last_ttt_accepted = False
        else:
            self._last_ttt_accepted = True

    # ── Belief updates ────────────────────────────────────────────────────────

    def ttt_step_beliefs(
        self,
        beliefs: Tensor,
        candidates_from_write: list,
        active_mask: Tensor,
        belief_lr: float = 0.0001,
        belief_lr_scale: Tensor | None = None,
    ):
        """Update belief vectors using prediction error signal.

        At inference time, beliefs should drift toward being more useful.
        We use the write path's observation vectors as targets: beliefs
        that are far from current observations get pulled toward them.

        This replaces what the optimizer did during training, but now runs
        at inference time.

        Args:
            beliefs: [max_beliefs, D] belief parameter (modified in-place)
            candidates_from_write: write candidates from the current pass
            active_mask: [max_beliefs] boolean mask of active beliefs
            belief_lr: base step size for belief updates
            belief_lr_scale: [max_beliefs] per-belief adaptive LR scale
                (RWKV-7 inspired). If None, uniform scale of 1.0 is used.
                Allows different beliefs to adapt at different rates based
                on their uncertainty or recency.
        """
        if not candidates_from_write or not active_mask.any():
            return

        with torch.no_grad():
            for candidate in candidates_from_write:
                slot = candidate.matched_slot
                if slot < 0 or not active_mask[slot]:
                    continue

                obs = candidate.belief_vector
                current = beliefs.data[slot]

                # Pull the belief toward the observation, weighted by observation precision
                obs_radius = obs.norm().clamp(min=1e-10)
                current_radius = current.norm().clamp(min=1e-10)

                # Kalman-like gain: how much to trust the observation vs existing belief
                gain = obs_radius / (obs_radius + current_radius)

                # Per-belief adaptive LR scale (RWKV-7 inspired)
                # Allows each belief to adapt at its own rate
                lr_scale = 1.0
                if belief_lr_scale is not None:
                    lr_scale = belief_lr_scale[slot].item()

                # Update: move belief toward observation
                beliefs.data[slot] = current + belief_lr * lr_scale * gain * (obs - current)


class TTTContext:
    """Tracks TTT state within a single forward pass.

    Collects utility signals from interface layers and hidden states
    needed for the TTT gradient step.
    """

    def __init__(self):
        self.utility_signals: dict[int, float] = {}
        self.fe_value: float = 0.0
        # Store pre-delta hidden states for TTT gradient computation
        self.pre_delta_hiddens: dict[int, Tensor] = {}

    def record_utility(self, interface_idx: int, layer_idx: int, utility_logits: Tensor):
        with torch.no_grad():
            self.utility_signals[layer_idx] = utility_logits.norm(dim=-1).mean().item()

    def get_utility(self, layer_idx: int) -> float:
        best_utility = 0.0
        for iface_layer, utility in self.utility_signals.items():
            if iface_layer < layer_idx:
                best_utility = max(best_utility, utility)
        return best_utility

    def save_pre_delta(self, layer_idx: int, hidden: Tensor):
        """Save hidden states before delta application for gradient computation."""
        self.pre_delta_hiddens[layer_idx] = hidden.detach()

    def get_pre_delta(self, layer_idx: int) -> Tensor | None:
        return self.pre_delta_hiddens.get(layer_idx)
