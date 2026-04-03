"""Sleep-inspired memory consolidation and dream phase.

SleepGate: learned module that decides which beliefs to consolidate or forget
during periodic sleep cycles. Addresses proactive interference where stale
beliefs disrupt retrieval of current information.

Dream phase: periodic internal simulation where beliefs propagate through the
causal graph without external input, allowing the world model to converge
and form abstractions.

Reference: "Learning to Forget: Sleep-Inspired Memory Consolidation" (arXiv:2603.14517)
Reference: NeuroDream (SSRN 5377250) — sleep-inspired consolidation framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.state import CognitiveState
from ..core.polar import angular_similarity, precision_weighted_average, belief_is_active, EPSILON
from ..core.message_passing import FactorGraphMessagePassing


class SleepGate(nn.Module):
    """Learned gate that decides which beliefs to consolidate, strengthen, or forget.

    During periodic "sleep" cycles (every N steps), the gate scores each active
    belief and produces three actions:
    - STRENGTHEN: increase precision (the belief is useful)
    - MAINTAIN: no change
    - FORGET: decrease precision (the belief causes interference)

    The gate is trained by the main optimizer through the free energy loss:
    forgetting interfering beliefs reduces free energy (less conflict in the graph).

    Reference: SleepGate (arXiv:2603.14517) — 99.5% retrieval at PI depth 5
    """

    def __init__(self, belief_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input: belief vector (D) + metadata features (6)
        # metadata: radius, access_count, level, age, n_edges, mean_edge_weight
        self.gate = nn.Sequential(
            nn.Linear(belief_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # [strengthen, maintain, forget]
        )
        # Init bias toward maintain (safe default)
        # softmax([0, 2, 0]) ≈ [0.11, 0.78, 0.11] — starts 78% maintain
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([0.0, 2.0, 0.0]))

    def forward(self, beliefs: Tensor, metadata: Tensor) -> Tensor:
        """Score beliefs for sleep cycle actions.

        Args:
            beliefs: [N, D] active belief vectors
            metadata: [N, 6] per-belief metadata features

        Returns:
            [N, 3] softmax probabilities over [strengthen, maintain, forget]
        """
        features = torch.cat([beliefs, metadata], dim=-1)
        logits = self.gate(features)
        return F.softmax(logits, dim=-1)


def run_sleep_cycle(
    state: CognitiveState,
    sleep_gate: SleepGate,
    current_step: int,
    strengthen_factor: float | None = None,
    forget_factor: float | None = None,
    forget_threshold: float | None = None,
) -> dict:
    """Run a learned sleep consolidation cycle.

    Periodically called (e.g., every 50-100 steps) to:
    1. Score each belief with the SleepGate
    2. Strengthen useful beliefs (increase radius)
    3. Forget interfering beliefs (decrease radius)
    4. Deallocate beliefs that decay below threshold

    Factors are derived from state if not provided:
    - strengthen_factor: 1 + (1 - fill_ratio) * 0.2 — strengthen more when there's room
    - forget_factor: fill_ratio — forget more aggressively when near capacity
    - forget_threshold: running_stats.hard_cleanup_precision_threshold

    Args:
        state: cognitive state
        sleep_gate: learned gate module
        current_step: for age computation
        strengthen_factor: multiplicative factor for strengthened beliefs
        forget_factor: multiplicative factor for forgotten beliefs
        forget_threshold: beliefs forgotten below this radius get deallocated

    Returns:
        dict with statistics
    """
    stats = {'strengthened': 0, 'forgotten': 0, 'deallocated': 0}

    # Derive factors from state if not explicitly provided
    fill_ratio = state.running_stats.belief_fill_ratio.item()
    if strengthen_factor is None:
        # Strengthen more when there's room to grow (low fill = aggressive strengthen)
        strengthen_factor = 1.0 + (1.0 - fill_ratio) / 5.0  # 1.0-1.2 range
    if forget_factor is None:
        # Forget more when near capacity (high fill = aggressive forget)
        forget_factor = 1.0 - fill_ratio / 3.0  # 0.67-1.0 range
        forget_factor = max(forget_factor, 0.5)  # floor: never forget more than half
    if forget_threshold is None:
        # Use the same threshold as hard cleanup (already derived from stats)
        forget_threshold = state.running_stats.hard_cleanup_precision_threshold

    active_mask = state.get_active_mask()
    if not active_mask.any():
        return stats

    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    active_beliefs = state.beliefs.data[active_idx]
    n_active = len(active_idx)

    # Build metadata features [N, 6]
    radii = active_beliefs.norm(dim=-1)
    access_counts = state.belief_access_count[active_idx]
    levels = (
        state.belief_level[active_idx].float()
        if hasattr(state, 'belief_level')
        else torch.zeros(n_active, device=active_beliefs.device)
    )
    ages = current_step - (
        state.belief_created_step[active_idx].float()
        if hasattr(state, 'belief_created_step')
        else torch.zeros(n_active, device=active_beliefs.device)
    )

    # Edge connectivity per belief
    n_edges = torch.zeros(n_active, device=active_beliefs.device)
    mean_edge_weight = torch.zeros(n_active, device=active_beliefs.device)
    if state.edge_active.any():
        active_edges = state.edge_active.nonzero(as_tuple=False).squeeze(-1)
        edge_srcs = state.edge_src[active_edges]
        edge_tgts = state.edge_tgt[active_edges]
        edge_w = state.edge_weights.data[active_edges]

        # Map global belief indices to local [0, n_active) indices
        idx_map = torch.full(
            (state.config.max_beliefs,), -1, dtype=torch.long, device=active_beliefs.device
        )
        idx_map[active_idx] = torch.arange(n_active, device=active_beliefs.device)

        for i in range(len(active_edges)):
            src_local = idx_map[edge_srcs[i]].item()
            tgt_local = idx_map[edge_tgts[i]].item()
            if src_local >= 0:
                n_edges[src_local] += 1
                mean_edge_weight[src_local] += edge_w[i]
            if tgt_local >= 0:
                n_edges[tgt_local] += 1
                mean_edge_weight[tgt_local] += edge_w[i]

        safe_n = n_edges.clamp(min=1)
        mean_edge_weight = mean_edge_weight / safe_n

    # Normalize metadata features to [0, 1]
    metadata = torch.stack([
        radii / (radii.max() + EPSILON),
        access_counts / (access_counts.max() + 1),
        levels / 3.0,
        ages / (ages.max() + 1),
        n_edges / (n_edges.max() + 1),
        mean_edge_weight,
    ], dim=-1)  # [N, 6]

    # Score beliefs with SleepGate
    with torch.no_grad():
        actions = sleep_gate(active_beliefs, metadata)  # [N, 3]
        best_action = actions.argmax(dim=-1)  # [N]

    with torch.no_grad():
        for i in range(n_active):
            idx = active_idx[i].item()
            if state.immutable_beliefs[idx]:
                continue

            action = best_action[i].item()

            if action == 0:  # strengthen
                state.beliefs.data[idx] *= strengthen_factor
                stats['strengthened'] += 1
            elif action == 2:  # forget
                state.beliefs.data[idx] *= forget_factor
                # Deallocate if decayed below threshold
                if state.beliefs.data[idx].norm().item() < forget_threshold:
                    state.deallocate_belief(idx)
                    stats['deallocated'] += 1
                else:
                    stats['forgotten'] += 1

    return stats


def run_dream_phase(
    state: CognitiveState,
    message_passing: FactorGraphMessagePassing,
    n_iterations: int | None = None,
    dream_lr: float | None = None,
    damping: float | None = None,
) -> dict:
    """Run a dream phase: internal belief propagation without external input.

    Disconnects from the input stream and runs N iterations of message passing
    on the belief graph, allowing beliefs to converge toward internal consistency.
    This is analogous to offline consolidation in complementary learning systems.

    During dreaming:
    1. Run message passing to compute per-belief messages
    2. Shift beliefs toward their incoming messages (weighted by precision)
    3. Repeat for N iterations
    4. The belief graph becomes more internally consistent

    Reference: NeuroDream (SSRN 5377250) — 38% reduction in forgetting,
               17.6% increase in zero-shot transfer

    Args:
        state: cognitive state (modified in-place)
        message_passing: the factor graph message passing module
        n_iterations: number of dream iterations
        dream_lr: step size for belief updates during dreaming
        damping: momentum factor for stability (0 = no damping, 1 = full damping)

    Returns:
        dict with statistics
    """
    stats = {'iterations': 0, 'total_shift': 0.0, 'converged': False}

    # Derive parameters from state if not provided
    if n_iterations is None:
        # More iterations when graph is denser (more edges to propagate)
        edge_density = state.num_active_edges() / max(state.num_active_beliefs(), 1)
        n_iterations = max(1, min(10, int(edge_density + 1)))
    if dream_lr is None:
        # Scale with beta: high exploration → larger dream steps
        dream_lr = state.meta.data[0].item() * 0.02 + 0.001  # beta-scaled
    if damping is None:
        # Use the learned damping from message passing if available
        if hasattr(message_passing, 'damping'):
            damping = message_passing.damping.item()
        else:
            damping = state.meta_params.precision_decay_factor.item()  # reuse a learned param

    active_mask = state.get_active_mask()
    if not active_mask.any() or not state.edge_active.any():
        return stats

    prev_shift = float('inf')

    with torch.no_grad():
        for iteration in range(n_iterations):
            # Run message passing (no gradient — dreaming is non-differentiable)
            mp_result = message_passing(state)
            messages = mp_result['messages']      # [max_beliefs, D]
            precisions = mp_result['precisions']  # [max_beliefs]

            # Only update active, non-immutable beliefs
            updatable = active_mask & ~state.immutable_beliefs
            if not updatable.any():
                break

            upd_idx = updatable.nonzero(as_tuple=False).squeeze(-1)

            total_shift = 0.0
            for idx in upd_idx.tolist():
                msg = messages[idx]
                msg_precision = precisions[idx].item()

                if msg_precision < EPSILON:
                    continue  # no meaningful messages for this belief

                current = state.beliefs.data[idx]
                current_r = current.norm().clamp(min=EPSILON)

                # Message-informed target direction
                msg_dir = F.normalize(msg.unsqueeze(0), dim=-1, eps=EPSILON).squeeze(0)
                current_dir = current / current_r

                # Move direction toward message, weighted by relative precision
                relative_precision = msg_precision / (msg_precision + current_r.item())
                shift = dream_lr * relative_precision

                # Damped directional update
                new_dir = F.normalize(
                    ((1.0 - shift) * current_dir + shift * msg_dir).unsqueeze(0),
                    dim=-1,
                    eps=EPSILON,
                ).squeeze(0)

                # Keep original radius — dreaming refines direction, not confidence
                state.beliefs.data[idx] = new_dir * current_r

                total_shift += shift

            stats['iterations'] = iteration + 1
            stats['total_shift'] += total_shift

            # Convergence check: stop if shift change is below noise floor
            if abs(total_shift - prev_shift) < EPSILON * len(upd_idx):
                stats['converged'] = True
                break
            prev_shift = total_shift

    return stats
