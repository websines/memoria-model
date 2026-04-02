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
Reference: TTT-E2E (NVIDIA/Stanford, Dec 2025) — 25% of MLP layers as mutable
Reference: Titans (Google, Jan 2025) — surprise-driven memory updates
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
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        ttt_layers: list[int] | None = None,
        rank: int = 32,
        ttt_lr: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.ttt_lr = ttt_lr

        # Default: top 25% of layers (most abstract representations)
        if ttt_layers is None:
            n_ttt = max(1, n_layers // 4)
            ttt_layers = list(range(n_layers - n_ttt, n_layers))
        self.ttt_layers = set(ttt_layers)
        self._ttt_layer_list = sorted(ttt_layers)

        # Per-layer persistent low-rank deltas
        # These are NOT nn.Parameters (not updated by the main optimizer)
        # They are updated by the TTT inner loop gradient step
        self.delta_A = nn.ParameterDict()
        self.delta_B = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            # Register as parameters so they're saved/loaded with the model,
            # but we'll exclude them from the main optimizer
            self.delta_A[key] = nn.Parameter(
                torch.zeros(hidden_dim, rank), requires_grad=False,
            )
            self.delta_B[key] = nn.Parameter(
                torch.zeros(rank, hidden_dim), requires_grad=False,
            )

        # Learnable step-size per layer (trained during training, fixed at inference)
        # Controls how much each layer adapts per chunk
        self.log_step_size = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            # Init: log(0.001) ≈ -6.9 → step_size ≈ 0.001
            self.log_step_size[key] = nn.Parameter(torch.tensor(-6.9))

        # RND surprise EMA for quality gating (not a parameter, just tracking state)
        self._surprise_ema: float = 0.0
        self._last_ttt_accepted: bool = True

    def is_ttt_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.ttt_layers

    def should_update(self, surprise: float) -> bool:
        """Gate TTT updates using RND surprise from the Telos module.

        If surprise is extremely high (> 3× the running mean), the input
        is too far out of distribution — updating on it would corrupt the
        deltas. Skip the TTT step entirely.

        If surprise is extremely low (< 0.1× the running mean), the input
        is fully predicted — updating on it wastes compute. Skip.

        The sweet spot: moderate surprise means the input is learnable
        (within distribution but not fully predicted).

        Args:
            surprise: mean RND surprise for the current input's beliefs

        Returns:
            True if the TTT step should proceed
        """
        mean = self._surprise_ema
        if mean < 1e-8:
            return True  # no history yet, allow all updates

        ratio = surprise / mean
        # Skip if: too surprising (OOD, ratio > 3) or too boring (ratio < 0.1)
        return 0.1 < ratio < 3.0

    def update_surprise_ema(self, surprise: float):
        """Track running mean of RND surprise for gating."""
        self._surprise_ema = 0.99 * self._surprise_ema + 0.01 * surprise

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

        # delta @ hidden^T → [B, T, hidden_dim]
        # Efficient: hidden @ B^T @ A^T = hidden @ (A @ B)^T
        delta_out = F.linear(F.linear(hidden, B), A)
        return hidden + delta_out

    @torch.no_grad()
    def ttt_step(
        self,
        layer_idx: int,
        hidden_pre_delta: Tensor,
        targets: Tensor,
        lm_head_weight: Tensor,
        vocab_size: int,
    ):
        """Take one TTT gradient step: update the persistent delta.

        This is the self-improvement step. It runs at BOTH training and inference.
        Uses next-token prediction loss on the current chunk as the objective.
        Includes quality gating: snapshots deltas before update, rolls back if
        the update increased loss (bad input protection).

        The gradient is computed analytically for the low-rank delta,
        avoiding full backward() through the transformer.

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

        # ── Snapshot deltas for rollback ──
        A_snapshot = A.data.clone()
        B_snapshot = B.data.clone()

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

        # ── Compute gradient ──
        probs = F.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(2, t_sample.clamp(min=0).unsqueeze(-1), 1.0)
        # Zero out gradient for invalid positions
        mask = valid.unsqueeze(-1).float()
        grad_logits = (probs - one_hot) * mask / max(valid.sum().item(), 1)
        grad_h = grad_logits @ lm_head_weight  # [B, max_pos, D]

        h_flat = h_for_grad.reshape(-1, D)
        grad_flat = grad_h.reshape(-1, D)
        n = max(h_flat.shape[0], 1)

        hB = h_flat @ B.T  # [N, R]
        grad_A = grad_flat.T @ hB / n
        grad_B = A.T @ (grad_flat.T @ h_flat) / n

        # ── Apply update ──
        A.data -= step_size * grad_A
        B.data -= step_size * grad_B

        # ── Rollback check: did the update help? ──
        h_delta_new = h + F.linear(F.linear(h_for_grad, B), A)
        if T_size > max_positions:
            logits_new = F.linear(h_delta_new, lm_head_weight)
        else:
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

    def ttt_step_beliefs(
        self,
        beliefs: Tensor,
        candidates_from_write: list,
        active_mask: Tensor,
        belief_lr: float = 0.0001,
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
            belief_lr: step size for belief updates
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

                # Update: move belief toward observation
                beliefs.data[slot] = current + belief_lr * gain * (obs - current)


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
