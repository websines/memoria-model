"""C2: Meta-learned belief update function.

Parameterizes the Pass 2 belief update as a small neural network.
Instead of the hand-coded Kalman-gain update rule, a learned function maps
(belief, observation, precision, prediction_error, edge_context) →
(delta_belief, new_precision, merge_signal).

The learned update is gated by MetaParams.update_fn_gate ∈ (0, 1):
    final = (1 - gate) * handcoded_update + gate * learned_update

This allows the system to gradually learn its own belief update algorithm
while maintaining stability. The gate starts near 0 (hand-coded dominates)
and learns to open as the NN becomes trustworthy.

Before deploying a learned update, it should be certified by the SGM safety
gate (A4) — the system proves statistically that the learned update produces
lower free energy than the hand-coded one.

Reference: ACL — Metalearning Continual Learning Algorithms (arXiv:2312.00276)
Reference: OPEN: Learned Optimization (arXiv:2407.07082)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.polar import EPSILON


class LearnedUpdateFunction(nn.Module):
    """Learned belief update function (C2).

    A small MLP that takes the current belief state and incoming observation,
    along with contextual features, and produces an updated belief vector.

    The network learns:
    - How to blend old belief with new observation (generalizes Kalman gain)
    - When a belief's precision should increase or decrease
    - When two beliefs should be merged (merge signal)

    Input features:
        belief: [D] current belief vector (cartesian)
        observation: [D] new observation vector (from write candidates)
        precision: [1] current belief radius (precision)
        prediction_error: [1] magnitude of surprise
        edge_context: [D] mean of connected belief vectors (relational context)

    Output:
        delta_belief: [D] additive update to belief vector
        precision_scale: [1] multiplicative factor for radius (softplus, ≥0)
        merge_signal: [1] probability that this belief should be merged (sigmoid)

    Args:
        belief_dim: dimension of belief vectors
        hidden_dim: hidden layer size (derived: belief_dim // 2)
    """

    def __init__(self, belief_dim: int):
        super().__init__()
        self.belief_dim = belief_dim
        hidden_dim = belief_dim // 2

        # Input: belief + obs + precision + error + edge_context = 2D + 2 + D = 3D + 2
        input_dim = belief_dim * 3 + 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Separate heads for each output (cleaner gradients)
        self.delta_head = nn.Linear(hidden_dim, belief_dim)
        self.precision_head = nn.Linear(hidden_dim, 1)
        self.merge_head = nn.Linear(hidden_dim, 1)

        # Initialize delta_head near zero (learned update starts as no-op)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        # Initialize precision_head to produce ~1.0 (no change)
        # softplus(0.541) ≈ 1.0
        nn.init.zeros_(self.precision_head.weight)
        nn.init.constant_(self.precision_head.bias, 0.541)

        # Initialize merge_head to produce low probability
        nn.init.zeros_(self.merge_head.weight)
        nn.init.constant_(self.merge_head.bias, -2.0)

    def forward(
        self,
        belief: torch.Tensor,
        observation: torch.Tensor,
        precision: torch.Tensor,
        prediction_error: torch.Tensor,
        edge_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute learned belief update.

        Args:
            belief: [N, D] or [D] current belief vectors
            observation: [N, D] or [D] incoming observation vectors
            precision: [N] or [1] current belief radii
            prediction_error: [N] or [1] surprise magnitudes
            edge_context: [N, D] or [D] mean of connected beliefs

        Returns:
            delta_belief: [N, D] additive update to apply
            precision_scale: [N] multiplicative precision factor (>0)
            merge_signal: [N] probability of merge (0-1)
        """
        # Handle both batched and single inputs
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
            observation = observation.unsqueeze(0)
            edge_context = edge_context.unsqueeze(0)
        if precision.dim() == 0:
            precision = precision.unsqueeze(0)
        if prediction_error.dim() == 0:
            prediction_error = prediction_error.unsqueeze(0)

        # Pack input features
        x = torch.cat([
            belief,                                    # [N, D]
            observation,                               # [N, D]
            precision.unsqueeze(-1),                   # [N, 1]
            prediction_error.unsqueeze(-1),            # [N, 1]
            edge_context,                              # [N, D]
        ], dim=-1)  # [N, 3D+2]

        hidden = self.net(x)  # [N, hidden_dim]

        delta_belief = self.delta_head(hidden)         # [N, D]
        precision_scale = F.softplus(self.precision_head(hidden)).squeeze(-1)  # [N]
        merge_signal = torch.sigmoid(self.merge_head(hidden)).squeeze(-1)      # [N]

        return delta_belief, precision_scale, merge_signal


def apply_learned_update(
    beliefs: torch.Tensor,
    observations: torch.Tensor,
    precisions: torch.Tensor,
    errors: torch.Tensor,
    edge_contexts: torch.Tensor,
    update_fn: LearnedUpdateFunction,
    gate: torch.Tensor,
    handcoded_deltas: torch.Tensor,
    handcoded_precision_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply gated blend of learned and hand-coded updates.

    final_delta = (1 - gate) * handcoded_delta + gate * learned_delta
    final_precision = (1 - gate) * handcoded_scale + gate * learned_scale

    The gate (from MetaParams.update_fn_gate) starts near 0.1 so the system
    is initially hand-coded. As training progresses, the gate can open to
    trust the learned update more.

    Args:
        beliefs: [N, D] current beliefs
        observations: [N, D] new observations
        precisions: [N] current radii
        errors: [N] surprise values
        edge_contexts: [N, D] relational context
        update_fn: the learned update network
        gate: scalar ∈ (0, 1) blend factor
        handcoded_deltas: [N, D] updates from standard Kalman-gain rule
        handcoded_precision_scales: [N] precision scales from standard rule

    Returns:
        final_deltas: [N, D] blended belief updates
        final_precision_scales: [N] blended precision scales
        merge_signals: [N] merge probabilities (from learned fn only)
    """
    learned_delta, learned_scale, merge_signal = update_fn(
        beliefs, observations, precisions, errors, edge_contexts,
    )

    # Gated blend
    g = gate.clamp(0.0, 1.0)
    final_delta = (1.0 - g) * handcoded_deltas + g * learned_delta
    final_scale = (1.0 - g) * handcoded_precision_scales + g * learned_scale

    return final_delta, final_scale, merge_signal


def get_edge_context(state, belief_indices: torch.Tensor) -> torch.Tensor:
    """Compute mean of connected belief vectors for each belief.

    For beliefs with no edges, returns zero vector. This provides relational
    context to the learned update function — beliefs connected to many others
    update differently than isolated beliefs.

    Args:
        state: CognitiveState
        belief_indices: [N] indices of beliefs being updated

    Returns:
        [N, D] mean connected belief vectors
    """
    device = state.beliefs.device
    D = state.config.belief_dim
    N = len(belief_indices)
    context = torch.zeros(N, D, device=device)

    if state.num_active_edges() == 0:
        return context

    src_idx, tgt_idx, _, weights = state.get_active_edges()

    for i, bidx in enumerate(belief_indices.tolist()):
        # Find edges where this belief is source or target
        as_src = (src_idx == bidx)
        as_tgt = (tgt_idx == bidx)

        neighbors = []
        if as_src.any():
            neighbor_idx = tgt_idx[as_src]
            neighbor_w = weights[as_src].abs()
            neighbors.append((neighbor_idx, neighbor_w))
        if as_tgt.any():
            neighbor_idx = src_idx[as_tgt]
            neighbor_w = weights[as_tgt].abs()
            neighbors.append((neighbor_idx, neighbor_w))

        if neighbors:
            all_idx = torch.cat([n[0] for n in neighbors])
            all_w = torch.cat([n[1] for n in neighbors])
            # Weighted mean of neighbor beliefs
            neighbor_beliefs = state.beliefs.data[all_idx]  # [K, D]
            w_sum = all_w.sum().clamp(min=EPSILON)
            context[i] = (neighbor_beliefs * all_w.unsqueeze(-1)).sum(0) / w_sum

    return context
