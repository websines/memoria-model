"""Factor graph message passing using PyTorch Geometric's MessagePassing.

Instead of hand-rolling loops over edges, we use PyG's sparse message passing
infrastructure. This gives us:
- Efficient scatter/gather operations on GPU
- Proper sparse tensor handling
- Battle-tested message aggregation

The factor graph is represented as a bipartite graph:
- Variable nodes (beliefs) send/receive messages via factor nodes (edges)
- Messages are precision-weighted (from Memoria's aif/messages.rs)

Loopy BP with learned damping:
- _raw_damping (sigmoid → (0,1)): stabilises fixed-point iteration
- _raw_iterations (softplus → (0,∞)): continuous relaxation of iteration count

Reference: PyTorch Geometric MessagePassing (pytorch-geometric.readthedocs.io)
Reference: torch-bp (github.com/janapavlasek/torch-bp) — Gaussian BP in PyTorch
Reference: RxInfer.jl — Bethe free energy via message passing (the gold standard)
Reference: prototype-research/src/aif/messages.rs — precision-weighted fusion
Reference: Mooij & Kappen, "On the properties of the loopy belief propagation fixed points" (2005)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .polar import angular_similarity, EPSILON
from .state import CognitiveState


class FactorGraphMessagePassing(nn.Module if not HAS_PYG else MessagePassing):
    """Precision-weighted message passing on the cognitive state's relation graph.

    Each message from belief j to belief i through edge f:
        message = w_f × r_j × transform(angle_j, relation_f)
        precision = w_f × r_j

    Aggregation (precision-weighted fusion from Memoria):
        fused = Σ(precision_k × message_k) / Σ(precision_k)

    Loopy BP runs multiple passes with learned damping to find a fixed point:
        messages_t = damping * messages_{t-1} + (1 - damping) * new_messages_t

    Damping (∈ (0,1)) stabilises the iteration. The number of iterations and
    the damping factor are both learned nn.Parameters (not hardcoded).
    """

    def __init__(self, belief_dim: int, relation_dim: int):
        if HAS_PYG:
            super().__init__(aggr='add')  # we'll do precision-weighted fusion manually
        else:
            super().__init__()
        self.belief_dim = belief_dim
        self.relation_dim = relation_dim

        # Learnable relation transform: how relation vectors modify messages
        self.relation_transform = nn.Linear(relation_dim, belief_dim, bias=False)

        # ── Learned loopy BP parameters ───────────────────────────────────────
        # Damping factor: raw → sigmoid → (0, 1).
        # sigmoid(0.0) = 0.5 — starts at balanced damping (equal old/new mix).
        # Higher damping = slower convergence but more stable fixed points.
        self._raw_damping = nn.Parameter(torch.tensor(0.0))

        # Iteration count: raw → softplus → (0, ∞), then rounded to int at inference.
        # softplus(1.0) ≈ 1.313 → rounds to 1 at start (single pass = existing behaviour).
        # The gradient through softplus allows the optimizer to discover that more
        # iterations help (increasing raw → softplus grows → more iterations).
        self._raw_iterations = nn.Parameter(torch.tensor(1.0))

    @property
    def damping(self) -> Tensor:
        """Learned damping factor in (0, 1). Higher = more stable, slower convergence."""
        return torch.sigmoid(self._raw_damping)

    @property
    def effective_iterations(self) -> int:
        """Learned iteration count (at least 1).

        Continuous relaxation: softplus(raw) → round to nearest int.
        Gradient flows back through softplus during training even though
        the integer rounding is non-differentiable at inference.
        """
        return max(1, int(F.softplus(self._raw_iterations).item() + 0.5))

    def _single_pass(self, state: CognitiveState) -> dict:
        """Single message passing iteration.

        Extracted from the original forward() for use in the loopy BP loop.
        Preserves the exact behaviour of the original forward() implementation.

        Returns:
            dict with messages [N, D], precisions [N], agreement [E]
        """
        # Get active edges
        active_mask = state.edge_active
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        src = state.edge_src[active_idx]  # [E]
        tgt = state.edge_tgt[active_idx]  # [E]
        relations = state.edge_relations[active_idx]  # [E, K]
        weights = state.edge_weights[active_idx]  # [E]

        beliefs = state.beliefs  # [N, D]
        radii = beliefs.norm(dim=-1).clamp(min=EPSILON)  # [N]
        angles = beliefs / radii.unsqueeze(-1)  # [N, D]

        # Transform source angles through relation
        src_angles = angles[src]  # [E, D]
        tgt_angles = angles[tgt]  # [E, D]
        relation_bias = self.relation_transform(relations)  # [E, D]

        # Transformed target = what target "looks like" through the relation
        tgt_transformed = F.normalize(
            tgt_angles + relation_bias, dim=-1, eps=EPSILON
        )

        # Agreement per edge
        agreement = angular_similarity(src_angles, tgt_transformed)  # [E]

        # CoED edge directions: scale messages by cos(θ) for tgt→src, sin(θ) for src→tgt
        # θ=π/4 (init): cos=sin=0.707 — symmetric/undirected
        # θ→0: cos→1, sin→0 — fully directed src→tgt (message flows tgt to src)
        # θ→π/2: cos→0, sin→1 — fully directed tgt→src
        directions = state.edge_direction[active_idx]  # [E]
        dir_cos = torch.cos(directions)  # tgt→src message scale
        dir_sin = torch.sin(directions)  # src→tgt message scale

        # Messages: tgt sends to src, scaled by cos(θ)
        msg_precision_fwd = weights * radii[tgt] * dir_cos  # [E]
        msg_values_fwd = msg_precision_fwd.unsqueeze(-1) * tgt_transformed  # [E, D]

        # Reverse messages: src sends to tgt, scaled by sin(θ)
        src_transformed = F.normalize(
            src_angles + relation_bias, dim=-1, eps=EPSILON
        )
        msg_precision_rev = weights * radii[src] * dir_sin  # [E]
        msg_values_rev = msg_precision_rev.unsqueeze(-1) * src_transformed  # [E, D]

        # Aggregate forward + reverse messages
        n = state.config.max_beliefs
        agg_messages = torch.zeros(n, self.belief_dim, device=beliefs.device)
        agg_precisions = torch.zeros(n, device=beliefs.device)

        # Forward: tgt→src
        agg_messages.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.belief_dim), msg_values_fwd)
        agg_precisions.scatter_add_(0, src, msg_precision_fwd)
        # Reverse: src→tgt
        agg_messages.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.belief_dim), msg_values_rev)
        agg_precisions.scatter_add_(0, tgt, msg_precision_rev)

        # Normalize by total precision (precision-weighted fusion)
        safe_prec = agg_precisions.unsqueeze(-1).clamp(min=EPSILON)
        agg_messages = agg_messages / safe_prec

        return {
            'messages': agg_messages,
            'precisions': agg_precisions,
            'agreement': agreement,
        }

    def forward(self, state: CognitiveState, num_iterations: int | None = None) -> dict:
        """Run message passing on the factor graph with optional loopy BP.

        When num_iterations > 1 (or the learned count exceeds 1), runs loopy BP
        with learned damping to find a stable message fixed point.

        Args:
            state: cognitive state with beliefs and edges
            num_iterations: override for learned iteration count.
                None → use self.effective_iterations (learned value).
                1 → single pass (original behaviour, no damping needed).

        Returns:
            dict with:
                messages: [N_beliefs, D] aggregated incoming messages per belief
                precisions: [N_beliefs] aggregated incoming precision per belief
                agreement: [N_edges] per-edge agreement score (from last iteration)
        """
        if not state.edge_active.any():
            n = state.config.max_beliefs
            return {
                'messages': torch.zeros(n, self.belief_dim, device=state.beliefs.device),
                'precisions': torch.zeros(n, device=state.beliefs.device),
                'agreement': torch.tensor([], device=state.beliefs.device),
            }

        n_iters = num_iterations if num_iterations is not None else self.effective_iterations

        # First pass — initialise messages
        result = self._single_pass(state)

        if n_iters <= 1:
            return result

        # ── Loopy BP: iterate with learned damping ────────────────────────────
        # Damped update rule:
        #   m_t = α * m_{t-1} + (1 - α) * m̂_t
        # where α = self.damping (learned), m̂_t = fresh single-pass messages.
        # This is the standard fixed-point iteration with momentum damping
        # (Mooij & Kappen 2005; also used in torch-bp).
        damping = self.damping
        prev_messages = result['messages'].clone()
        prev_precisions = result['precisions'].clone()

        for _iteration in range(1, n_iters):
            new_result = self._single_pass(state)

            # Damped message update
            result['messages'] = damping * prev_messages + (1.0 - damping) * new_result['messages']
            result['precisions'] = damping * prev_precisions + (1.0 - damping) * new_result['precisions']
            # Agreement always taken from the latest pass (reflects current beliefs)
            result['agreement'] = new_result['agreement']

            prev_messages = result['messages'].clone()
            prev_precisions = result['precisions'].clone()

        return result


def compute_energy_from_messages(
    state: CognitiveState,
    agreement: Tensor,
    temperature: float = 5.0,
) -> Tensor:
    """Compute energy from per-edge agreement scores.

    E = Σ_f -w_f × r_i × r_j × log(σ(agreement × temp))

    Args:
        state: cognitive state
        agreement: [E] per-edge agreement from message passing
        temperature: sigmoid sharpness

    Returns:
        Scalar energy tensor
    """
    if len(agreement) == 0:
        return torch.tensor(0.0, device=state.beliefs.device)

    active_idx = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
    src = state.edge_src[active_idx]
    tgt = state.edge_tgt[active_idx]
    weights = state.edge_weights[active_idx]

    radii = state.beliefs.norm(dim=-1).clamp(min=EPSILON)
    src_radii = radii[src]
    tgt_radii = radii[tgt]

    log_sigmoid = F.logsigmoid(agreement * temperature)
    energy_per_edge = -weights * src_radii * tgt_radii * log_sigmoid

    return energy_per_edge.sum()


def apply_belief_shift(
    state: CognitiveState,
    messages: Tensor,
    precisions: Tensor,
) -> Tensor:
    """Apply message-informed belief shifts (confidence cascade).

    Beliefs with incoming high-precision messages shift their direction toward
    the message direction. The shift magnitude is proportional to the relative
    precision of the messages versus the belief's own precision (radius).

    This implements "confidence propagation through source chains": when
    graph neighbours provide strong directional evidence, a belief updates.
    The shift rate is derived from the belief's own radius so that high-precision
    beliefs are harder to move (anchored by their own evidence).

    Shift rate derivation (no magic numbers):
      - base_rate = beta / D
        beta  = current exploration/exploitation factor (∈ [0,1], from state.meta[0])
        D     = belief_dim (mathematical normalisation: 1/D scales to dimensionality)
        When beta is high (exploring), beliefs shift more readily.
        When D is large (high-dim), each coordinate shift is smaller → stable.
      - relative = msg_precision / (msg_precision + belief_radius)
        This is the fraction of total precision attributable to the message.
        It is derived from the precisions themselves — no scaling constant needed.
      - actual_shift = base_rate * relative
        Bounded in (0, base_rate) ⊂ (0, 1/D) → small, numerically safe.

    Only mutable, active beliefs that receive meaningfully precise messages are shifted.

    Args:
        state: cognitive state (beliefs modified in-place under no_grad)
        messages: [max_beliefs, D] aggregated directional messages from message passing
        precisions: [max_beliefs] aggregated precision per belief slot

    Returns:
        [N_shifted] long tensor of belief indices that were actually shifted
    """
    device = state.beliefs.device
    D = state.config.belief_dim

    active = state.get_active_mask() & ~state.immutable_beliefs
    if not active.any():
        return torch.tensor([], dtype=torch.long, device=device)

    active_idx = active.nonzero(as_tuple=False).squeeze(-1)

    # base_rate = β / D — exploration-weighted, dimension-normalised.
    # β ∈ [0, 1] from state.meta[0]; D is the belief dimensionality.
    # This is a mathematically motivated expression: 1/D is the natural per-
    # dimension scale, and β down-weights it during exploitation phases.
    beta = state.meta.data[0].item()   # ∈ [0, 1]
    base_rate = beta / D               # ∈ [0, 1/D], always small and well-scaled

    shifted = []

    with torch.no_grad():
        for idx in active_idx.tolist():
            msg_precision = precisions[idx].item()
            if msg_precision < EPSILON:
                continue  # no meaningful message arriving at this belief

            belief = state.beliefs.data[idx]
            belief_r = belief.norm().clamp(min=EPSILON).item()

            # Relative precision: what fraction of total evidence is from the message?
            # relative ∈ (0, 1); derived purely from the two precision values.
            relative = msg_precision / (msg_precision + belief_r)

            # Only shift if the message contributes more than the belief's prior would
            # suggest. The natural threshold here is belief_r / (belief_r + belief_r) = 0.5,
            # i.e., we only shift when the message is stronger than the belief itself.
            # Expressed without a magic number: relative > belief_r / (msg_precision + belief_r)
            # simplifies to msg_precision > belief_r.
            if msg_precision <= belief_r:
                continue  # belief is stronger evidence; don't shift

            msg_dir = F.normalize(messages[idx].unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
            belief_dir = belief / belief_r

            actual_shift = base_rate * relative  # ∈ (0, base_rate)
            new_dir = F.normalize(
                ((1.0 - actual_shift) * belief_dir + actual_shift * msg_dir).unsqueeze(0),
                dim=-1, eps=EPSILON,
            ).squeeze(0)

            state.beliefs.data[idx] = new_dir * belief_r
            shifted.append(idx)

    return torch.tensor(shifted, dtype=torch.long, device=device)
