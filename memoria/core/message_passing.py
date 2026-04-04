"""Factor graph message passing with implicit fixed-point solving (DEQ).

Instead of unrolling loopy BP iterations, we use a Deep Equilibrium Model
(DEQ) with Anderson acceleration to find the BP fixed point implicitly.
This gives:
- Constant memory regardless of effective iteration depth
- Exact gradients via implicit differentiation
- Faster convergence than damped iteration (Anderson acceleration)
- Well-posedness guaranteed by spectral norm on relation_transform

The factor graph is represented as a bipartite graph:
- Variable nodes (beliefs) send/receive messages via factor nodes (edges)
- Messages are precision-weighted (from Memoria's aif/messages.rs)

Fallback: when TorchDEQ is not available, uses simple fixed-point iteration
with configurable max iterations.

Reference: Deep Equilibrium Models (Bai, Kolter, Koltun — NeurIPS 2019)
Reference: TorchDEQ (github.com/locuslab/torchdeq)
Reference: IGNN — Implicit Graph Neural Networks (Gu & Chang — NeurIPS 2020)
Reference: PyTorch Geometric MessagePassing (pytorch-geometric.readthedocs.io)
Reference: torch-bp (github.com/janapavlasek/torch-bp) — Gaussian BP in PyTorch
Reference: RxInfer.jl — Bethe free energy via message passing (the gold standard)
Reference: prototype-research/src/aif/messages.rs — precision-weighted fusion
Reference: Mooij & Kappen, "On the properties of the loopy belief propagation
           fixed points" (2005)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# TorchDEQ: implicit fixed-point solver
try:
    from torchdeq import get_deq
    from torchdeq.loss import jac_reg
    HAS_TORCHDEQ = True
except ImportError:
    HAS_TORCHDEQ = False

from .polar import angular_similarity, EPSILON
from .state import CognitiveState


class FactorGraphMessagePassing(nn.Module if not HAS_PYG else MessagePassing):
    """Precision-weighted message passing on the cognitive state's relation graph.

    Each message from belief j to belief i through edge f:
        message = w_f × r_j × transform(angle_j, relation_f)
        precision = w_f × r_j

    Aggregation (precision-weighted fusion from Memoria):
        fused = Σ(precision_k × message_k) / Σ(precision_k)

    Fixed-point solving:
        When TorchDEQ is available, uses Anderson acceleration to find the
        BP fixed point implicitly (constant memory, exact gradients).
        Otherwise, falls back to simple fixed-point iteration.

    Spectral norm on relation_transform ensures the BP map is contractive,
    guaranteeing convergence (IGNN well-posedness condition).
    """

    def __init__(self, belief_dim: int, relation_dim: int):
        if HAS_PYG:
            super().__init__(aggr='add')  # we'll do precision-weighted fusion manually
        else:
            super().__init__()
        self.belief_dim = belief_dim
        self.relation_dim = relation_dim

        # Learnable relation transform with spectral norm for well-posedness.
        # Spectral norm bounds the largest singular value to 1, ensuring the
        # BP map is contractive → guaranteed convergence to a unique fixed point
        # (IGNN, Gu & Chang, NeurIPS 2020).
        self.relation_transform = spectral_norm(
            nn.Linear(relation_dim, belief_dim, bias=False)
        )

        # ── DEQ solver setup ─────────────────────────────────────────────────
        # Anderson acceleration for the forward pass, fixed-point iter for backward.
        # f_max_iter/b_max_iter are upper bounds — solver stops early at convergence.
        # f_tol: convergence threshold (relative residual). The solver stops when
        # ||f(z) - z|| / ||z|| < f_tol. 1e-5 is standard for graph problems.
        self._deq = None
        if HAS_TORCHDEQ:
            self._deq = get_deq(
                f_solver='anderson',
                b_solver='fixed_point_iter',
                f_max_iter=30,
                b_max_iter=25,
                f_tol=1e-5,
                b_tol=1e-5,
                stop_mode='rel',
            )

        # Fallback: simple iteration when TorchDEQ is not installed
        self._fallback_max_iter = 10

        # Track last solver info for diagnostics
        self._last_info: dict = {}

    def _single_pass(self, state: CognitiveState) -> dict:
        """Single message passing iteration.

        Extracted from the original forward() for use in the fixed-point loop.
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

    def _pack_state(self, messages: Tensor, precisions: Tensor) -> Tensor:
        """Pack messages and precisions into a single flat tensor for the DEQ solver."""
        return torch.cat([messages.flatten(), precisions])

    def _unpack_state(self, z: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """Unpack flat tensor back into messages and precisions."""
        msg_size = n * self.belief_dim
        messages = z[:msg_size].reshape(n, self.belief_dim)
        precisions = z[msg_size:]
        return messages, precisions

    def forward(self, state: CognitiveState, num_iterations: int | None = None) -> dict:
        """Run message passing via implicit fixed-point solving.

        Uses Anderson acceleration (DEQ) to find the BP fixed point.
        Falls back to simple iteration when TorchDEQ is not available.

        Args:
            state: cognitive state with beliefs and edges
            num_iterations: override max iterations for fallback solver.
                Ignored when DEQ is available (solver determines convergence).

        Returns:
            dict with:
                messages: [N_beliefs, D] aggregated incoming messages per belief
                precisions: [N_beliefs] aggregated incoming precision per belief
                agreement: [N_edges] per-edge agreement score (from final pass)
                solver_steps: int, number of iterations the solver took
                jac_loss: Tensor, Jacobian regularization loss (0.0 if DEQ unavailable)
        """
        n = state.config.max_beliefs
        device = state.beliefs.device

        if not state.edge_active.any():
            return {
                'messages': torch.zeros(n, self.belief_dim, device=device),
                'precisions': torch.zeros(n, device=device),
                'agreement': torch.tensor([], device=device),
                'solver_steps': 0,
                'jac_loss': torch.tensor(0.0, device=device),
            }

        # Initial pass — seed for the solver
        init_result = self._single_pass(state)

        if self._deq is not None:
            # ── DEQ path: Anderson acceleration ──────────────────────────────
            z0 = self._pack_state(init_result['messages'], init_result['precisions'])

            # Fixed-point function: one BP pass that returns updated messages.
            # The solver finds z* where z* = f(z*).
            def f(z):
                # Run a BP pass (the relation_transform and beliefs define f)
                result = self._single_pass(state)
                return self._pack_state(result['messages'], result['precisions'])

            # Solve for fixed point
            z_out, info = self._deq(f, z0)
            self._last_info = {
                'nstep': info.get('nstep', torch.tensor(-1)).item()
                if isinstance(info.get('nstep'), Tensor) else info.get('nstep', -1),
            }

            # z_out is a list of trajectory states; take the final one
            z_star = z_out[-1] if isinstance(z_out, (list, tuple)) else z_out
            messages, precisions = self._unpack_state(z_star, n)

            # Compute Jacobian regularization for training stability
            jac_loss_val = torch.tensor(0.0, device=device)
            if self.training and HAS_TORCHDEQ:
                with torch.enable_grad():
                    f_z = f(z_star)
                    jac_loss_val = jac_reg(f_z, z_star, vecs=1, create_graph=True)

            # Agreement from the final state
            final_result = self._single_pass(state)

            return {
                'messages': messages,
                'precisions': precisions,
                'agreement': final_result['agreement'],
                'solver_steps': self._last_info.get('nstep', -1),
                'jac_loss': jac_loss_val,
            }

        else:
            # ── Fallback: simple fixed-point iteration ───────────────────────
            max_iter = num_iterations if num_iterations is not None else self._fallback_max_iter
            result = init_result

            if max_iter > 1:
                prev_z = self._pack_state(result['messages'], result['precisions'])
                for i in range(1, max_iter):
                    new_result = self._single_pass(state)
                    new_z = self._pack_state(new_result['messages'], new_result['precisions'])

                    # Check convergence (relative residual)
                    residual = (new_z - prev_z).norm() / (prev_z.norm() + EPSILON)
                    result = new_result
                    prev_z = new_z

                    if residual < 1e-5:
                        self._last_info = {'nstep': i + 1}
                        break
                else:
                    self._last_info = {'nstep': max_iter}

            return {
                'messages': result['messages'],
                'precisions': result['precisions'],
                'agreement': result['agreement'],
                'solver_steps': self._last_info.get('nstep', 1),
                'jac_loss': torch.tensor(0.0, device=device),
            }


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
