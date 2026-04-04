"""C4: Learned Recursion Depth — Adaptive Computation Time for beliefs.

Replaces fixed 2-pass processing with learned N-pass: each belief decides
its own recursion depth based on uncertainty and surprise. Beliefs that
need more processing get more iterations; converged beliefs halt early.

Uses ACT (Adaptive Computation Time, Graves 2016) with a learned ponder
cost. The halting probability is produced by a small network that takes
belief features as input:

    P(halt | belief, uncertainty, iteration) = sigmoid(halt_net(...))

The ponder cost (from MetaParams.recursion_depth_penalty) is added to the
training loss to prevent unbounded computation:

    L_ponder = depth_penalty * Σ_beliefs remaining_probability

Mixture-of-Recursions (MoR) style: different beliefs in the same batch
can have different recursion depths. The system learns which beliefs need
deep processing vs which are "easy" (already converged).

Reference: Mixture-of-Recursions (arXiv:2507.10524)
Reference: Adaptive Computation Time (Graves, 2016)
Reference: Universal Transformers (Dehghani et al., 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.polar import EPSILON


class AdaptiveDepth(nn.Module):
    """Per-belief halting decision network for adaptive recursion depth.

    Each belief gets its own halting probability computed from:
    - belief vector content
    - current uncertainty (precision variance)
    - iteration index (positional encoding)
    - accumulated update magnitude (how much has changed)

    Args:
        belief_dim: dimension of belief vectors
        max_depth: hard cap on recursion depth (safety limit)
    """

    def __init__(self, belief_dim: int, max_depth: int = 8):
        super().__init__()
        self.belief_dim = belief_dim
        self.max_depth = max_depth

        # Halting network: belief features → P(halt)
        # Input: belief_dim + 3 (uncertainty, iteration_encoding, accumulated_change)
        self.halt_net = nn.Sequential(
            nn.Linear(belief_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Initialize with slight bias toward halting (conservative)
        nn.init.zeros_(self.halt_net[-1].weight)
        nn.init.constant_(self.halt_net[-1].bias, 0.0)  # sigmoid(0) = 0.5

        # Iteration positional encoding (sinusoidal, fixed)
        self.register_buffer(
            '_iter_encoding',
            self._make_iter_encoding(max_depth),
        )

    @staticmethod
    def _make_iter_encoding(max_depth: int) -> torch.Tensor:
        """Sinusoidal encoding of iteration index, normalized to [0, 1]."""
        pos = torch.arange(max_depth, dtype=torch.float32)
        # Monotonically increasing encoding: later iterations → higher values
        return (pos / max(max_depth - 1, 1)).unsqueeze(-1)  # [max_depth, 1]

    def compute_halt_probs(
        self,
        beliefs: torch.Tensor,
        uncertainties: torch.Tensor,
        iteration: int,
        accumulated_change: torch.Tensor,
        halt_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Compute halting probability for each belief at this iteration.

        Args:
            beliefs: [N, D] belief vectors
            uncertainties: [N] precision variance per belief
            iteration: current iteration index (0-based)
            accumulated_change: [N] total update magnitude so far
            halt_bias: scalar bias from MetaParams.recursion_halt_bias

        Returns:
            [N] halting probabilities ∈ (0, 1)
        """
        N = beliefs.shape[0]
        device = beliefs.device

        # Iteration encoding (same for all beliefs)
        iter_enc = self._iter_encoding[min(iteration, self.max_depth - 1)]
        iter_enc = iter_enc.expand(N, -1)  # [N, 1]

        features = torch.cat([
            beliefs,                                      # [N, D]
            uncertainties.unsqueeze(-1),                   # [N, 1]
            iter_enc,                                      # [N, 1]
            accumulated_change.unsqueeze(-1),              # [N, 1]
        ], dim=-1)  # [N, D+3]

        logits = self.halt_net(features).squeeze(-1)  # [N]
        # Add learned bias
        logits = logits + halt_bias
        return torch.sigmoid(logits)


class ACTController:
    """Adaptive Computation Time controller for belief processing.

    Manages the halt-or-continue decision for each belief across iterations.
    Tracks cumulative halting probability (Graves' "remainder" trick) and
    computes the weighted output and ponder cost.

    Usage:
        controller = ACTController(adaptive_depth, beliefs, uncertainties, halt_bias)
        for iteration in range(max_depth):
            updates = compute_updates(active_beliefs)  # your update logic
            controller.step(updates, iteration)
            if controller.all_halted():
                break
        final_beliefs, ponder_cost = controller.finalize(depth_penalty)
    """

    def __init__(
        self,
        module: AdaptiveDepth,
        beliefs: torch.Tensor,
        uncertainties: torch.Tensor,
        halt_bias: torch.Tensor,
    ):
        self.module = module
        self.N = beliefs.shape[0]
        self.device = beliefs.device

        self.beliefs = beliefs.clone()
        self.uncertainties = uncertainties
        self.halt_bias = halt_bias

        # ACT state
        self.cumulative_halt = torch.zeros(self.N, device=self.device)
        self.remainders = torch.ones(self.N, device=self.device)
        self.accumulated_output = torch.zeros_like(beliefs)
        self.accumulated_change = torch.zeros(self.N, device=self.device)
        self.halted = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.n_steps = torch.zeros(self.N, device=self.device)

    def step(self, updated_beliefs: torch.Tensor, iteration: int):
        """Process one iteration for all non-halted beliefs.

        Args:
            updated_beliefs: [N, D] beliefs after this iteration's update
            iteration: current iteration index
        """
        # Compute halt probabilities for active beliefs
        halt_probs = self.module.compute_halt_probs(
            updated_beliefs, self.uncertainties, iteration,
            self.accumulated_change, self.halt_bias,
        )

        # Update change tracking
        change = (updated_beliefs - self.beliefs).norm(dim=-1)
        self.accumulated_change += change

        # ACT: determine who halts this step
        # A belief halts if cumulative_halt + halt_prob >= 1
        still_active = ~self.halted
        new_halt = still_active & ((self.cumulative_halt + halt_probs) >= 1.0)

        # For newly halted: use remainder as weight
        if new_halt.any():
            self.accumulated_output[new_halt] += (
                self.remainders[new_halt].unsqueeze(-1) * updated_beliefs[new_halt]
            )
            self.halted[new_halt] = True
            self.n_steps[new_halt] = iteration + 1

        # For still active: accumulate with halt_prob weight
        continuing = still_active & ~new_halt
        if continuing.any():
            self.accumulated_output[continuing] += (
                halt_probs[continuing].unsqueeze(-1) * updated_beliefs[continuing]
            )
            self.cumulative_halt[continuing] += halt_probs[continuing]
            self.remainders[continuing] = 1.0 - self.cumulative_halt[continuing]
            self.n_steps[continuing] = iteration + 1

        self.beliefs = updated_beliefs.clone()

    def all_halted(self) -> bool:
        """Check if all beliefs have halted."""
        return self.halted.all().item()

    def finalize(self, depth_penalty: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Finalize: compute weighted output and ponder cost.

        Beliefs that didn't halt get their remainder applied to the last state.

        Args:
            depth_penalty: scalar ponder cost per remaining probability
                (from MetaParams.recursion_depth_penalty)

        Returns:
            final_beliefs: [N, D] ACT-weighted belief vectors
            ponder_cost: scalar loss term to add to training loss
        """
        # Apply remainder to non-halted beliefs
        still_active = ~self.halted
        if still_active.any():
            self.accumulated_output[still_active] += (
                self.remainders[still_active].unsqueeze(-1) * self.beliefs[still_active]
            )

        # Ponder cost: sum of remainders (penalizes using many steps)
        ponder_cost = depth_penalty * self.remainders.sum()

        return self.accumulated_output, ponder_cost

    def get_stats(self) -> dict:
        """Get statistics about the computation depth."""
        return {
            'mean_depth': self.n_steps.mean().item(),
            'max_depth': self.n_steps.max().item(),
            'min_depth': self.n_steps.min().item(),
            'frac_halted_early': (self.n_steps < self.module.max_depth).float().mean().item(),
        }


def run_adaptive_depth_update(
    state,
    adaptive_depth: AdaptiveDepth,
    update_fn,
    max_depth: int | None = None,
) -> dict:
    """Run adaptive-depth belief processing.

    Each belief gets as many update iterations as it needs. The halting
    network learns when to stop. A ponder cost regularizes depth.

    Args:
        state: CognitiveState
        adaptive_depth: AdaptiveDepth module
        update_fn: callable(beliefs, iteration) → updated_beliefs
        max_depth: override for max iterations (default: module.max_depth)

    Returns:
        dict with final_beliefs, ponder_cost, and depth statistics
    """
    if max_depth is None:
        max_depth = adaptive_depth.max_depth

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return {
            'ponder_cost': torch.tensor(0.0, device=state.beliefs.device),
            'stats': {'mean_depth': 0, 'max_depth': 0, 'min_depth': 0, 'frac_halted_early': 1.0},
        }

    active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
    beliefs = state.beliefs.data[active_indices]
    uncertainties = state.belief_precision_var[active_indices]

    halt_bias = state.meta_params.recursion_halt_bias
    depth_penalty = state.meta_params.recursion_depth_penalty

    controller = ACTController(adaptive_depth, beliefs, uncertainties, halt_bias)

    for iteration in range(max_depth):
        updated = update_fn(controller.beliefs, iteration)
        controller.step(updated, iteration)
        if controller.all_halted():
            break

    final_beliefs, ponder_cost = controller.finalize(depth_penalty)
    stats = controller.get_stats()

    # Write back final beliefs
    with torch.no_grad():
        state.beliefs.data[active_indices] = final_beliefs

    return {
        'ponder_cost': ponder_cost,
        'stats': stats,
    }
