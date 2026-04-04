"""C3: Structural Plasticity — split, prune, and grow the belief store.

Monitors per-belief activation statistics to detect:
1. **Polysemantic beliefs**: high co-activation entropy → belief encodes multiple
   unrelated concepts → split into children (FPE: Expand Neurons Not Parameters)
2. **Dead beliefs**: low activation, low precision → prune and reclaim slot
3. **Capacity pressure**: sustained high surprise + low belief density → grow

All thresholds are learned via MetaParams (plasticity_split_threshold,
plasticity_prune_threshold, plasticity_growth_rate).

Reference: SMGrNN — Self-Motivated Growing Neural Network (arXiv:2512.12713)
Reference: FPE — Expand Neurons Not Parameters (arXiv:2510.04500)
Reference: DynMoE — Dynamic Mixture of Experts (arXiv:2405.14297)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..core.polar import EPSILON


class StructuralPlasticity(nn.Module):
    """Monitors activation statistics and proposes structural changes.

    Maintains per-belief running statistics:
    - co_activation_entropy: how diverse are the contexts in which this belief fires
    - activation_frequency: how often the belief is accessed (normalized)
    - context_diversity: number of distinct goal contexts the belief participates in

    Uses learned networks to decide split/prune/grow actions.

    Args:
        belief_dim: dimension of belief vectors
        max_beliefs: maximum number of belief slots
    """

    def __init__(self, belief_dim: int, max_beliefs: int):
        super().__init__()
        self.belief_dim = belief_dim
        self.max_beliefs = max_beliefs

        # Per-belief activation statistics (running EMA)
        self.register_buffer('activation_count', torch.zeros(max_beliefs))
        self.register_buffer('activation_entropy', torch.zeros(max_beliefs))
        self.register_buffer('context_signatures', torch.zeros(max_beliefs, 8))
        self.register_buffer('_total_steps', torch.tensor(0, dtype=torch.long))

        # Split decision network: belief features → P(should split)
        # Input: belief_dim + 3 (entropy, frequency, variance)
        self.split_net = nn.Sequential(
            nn.Linear(belief_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Initialize conservatively (don't split by default)
        nn.init.zeros_(self.split_net[-1].weight)
        nn.init.constant_(self.split_net[-1].bias, -3.0)

        # Prune decision network: belief features → P(should prune)
        self.prune_net = nn.Sequential(
            nn.Linear(belief_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.prune_net[-1].weight)
        nn.init.constant_(self.prune_net[-1].bias, -3.0)

        # Split direction network: belief → two child directions
        self.split_dir = nn.Linear(belief_dim, belief_dim * 2)
        nn.init.xavier_normal_(self.split_dir.weight, gain=0.1)

    def record_activation(self, indices: torch.Tensor, context: torch.Tensor | None = None):
        """Record that beliefs at given indices were activated.

        Updates running activation counts and context diversity.

        Args:
            indices: [N] belief indices that were accessed
            context: [N, D] optional context vectors (e.g., query that retrieved them)
        """
        if len(indices) == 0:
            return
        with torch.no_grad():
            self._total_steps += 1
            self.activation_count[indices] += 1

            if context is not None and context.shape[0] == len(indices):
                # Update context signature via EMA of first 8 PCA-like features
                sig = context[:, :min(8, context.shape[-1])]
                if sig.shape[-1] < 8:
                    sig = F.pad(sig, (0, 8 - sig.shape[-1]))
                decay = 0.95
                self.context_signatures[indices] = (
                    decay * self.context_signatures[indices]
                    + (1 - decay) * sig
                )

    def compute_activation_entropy(self, state) -> torch.Tensor:
        """Compute per-belief activation entropy from context signatures.

        High entropy = belief fires in diverse, dissimilar contexts = polysemantic.
        Low entropy = belief fires in similar contexts = specialized.

        Returns:
            [max_beliefs] entropy values, higher = more polysemantic
        """
        # Entropy from context signature variance
        sig = self.context_signatures  # [max_beliefs, 8]
        # Variance across signature dimensions → entropy proxy
        sig_var = sig.var(dim=-1).clamp(min=EPSILON)
        # Normalize to [0, 1] range
        entropy = (sig_var / (sig_var.max() + EPSILON)).clamp(0, 1)
        self.activation_entropy.copy_(entropy)
        return entropy

    def evaluate(
        self,
        state,
        split_threshold: torch.Tensor,
        prune_threshold: torch.Tensor,
    ) -> dict:
        """Evaluate which beliefs should be split, pruned, or if growth is needed.

        Args:
            state: CognitiveState
            split_threshold: from MetaParams.plasticity_split_threshold
            prune_threshold: from MetaParams.plasticity_prune_threshold

        Returns:
            dict with:
                split_candidates: list of belief indices to split
                prune_candidates: list of belief indices to prune
                should_grow: bool, whether capacity should increase
                growth_pressure: float, how urgently growth is needed
        """
        active_mask = state.get_active_mask()
        n_active = active_mask.sum().item()
        if n_active == 0:
            return {
                'split_candidates': [],
                'prune_candidates': [],
                'should_grow': False,
                'growth_pressure': 0.0,
            }

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        beliefs = state.beliefs.data[active_indices]  # [N, D]
        radii = beliefs.norm(dim=-1)

        # Compute features for each active belief
        entropy = self.activation_entropy[active_indices]
        total_steps = max(self._total_steps.item(), 1)
        frequency = self.activation_count[active_indices] / total_steps
        variance = state.belief_precision_var[active_indices]

        # Feature vectors for split/prune networks
        features = torch.cat([
            beliefs,                           # [N, D]
            entropy.unsqueeze(-1),             # [N, 1]
            frequency.unsqueeze(-1),           # [N, 1]
            variance.unsqueeze(-1),            # [N, 1]
        ], dim=-1)  # [N, D+3]

        # Split scores
        split_logits = self.split_net(features).squeeze(-1)  # [N]
        split_probs = torch.sigmoid(split_logits)
        split_mask = (split_probs > split_threshold) & (entropy > split_threshold)

        # Prune scores
        prune_logits = self.prune_net(features).squeeze(-1)  # [N]
        prune_probs = torch.sigmoid(prune_logits)
        # Only prune low-frequency, low-precision beliefs
        freq_low = frequency < prune_threshold
        radius_low = radii < radii.median() * prune_threshold
        prune_mask = (prune_probs > 0.5) & freq_low & radius_low

        # Don't split and prune the same belief
        prune_mask = prune_mask & ~split_mask

        # Don't touch immutable beliefs
        immutable = state.immutable_beliefs[active_indices]
        split_mask = split_mask & ~immutable
        prune_mask = prune_mask & ~immutable

        # Growth pressure: how full is the store, and is surprise high?
        fill_ratio = n_active / max(state.config.max_beliefs, 1)
        mean_surprise = state.meta.data[1].item()
        growth_pressure = fill_ratio * (1.0 + mean_surprise)
        should_grow = fill_ratio > 0.9 and mean_surprise > 0.5

        split_candidates = active_indices[split_mask].tolist()
        prune_candidates = active_indices[prune_mask].tolist()

        return {
            'split_candidates': split_candidates,
            'prune_candidates': prune_candidates,
            'should_grow': should_grow,
            'growth_pressure': growth_pressure,
        }

    def split_belief(self, state, belief_idx: int) -> tuple[int, int]:
        """Split a polysemantic belief into two children (FPE-style).

        The parent belief is replaced by two children:
        - Child A: parent + learned perturbation direction
        - Child B: parent - learned perturbation direction

        Each child gets half the parent's radius (precision splits).
        The parent slot is reused for child A; child B gets a new slot.

        Source chain tracking: both children cite the parent as their source.

        Args:
            state: CognitiveState
            belief_idx: index of belief to split

        Returns:
            (child_a_idx, child_b_idx), or (-1, -1) if no slot available
        """
        parent = state.beliefs.data[belief_idx]  # [D]
        parent_radius = parent.norm().item()

        if parent_radius < EPSILON:
            return (-1, -1)

        # Compute split directions from learned network
        split_out = self.split_dir(parent.unsqueeze(0)).squeeze(0)  # [2D]
        dir_a = split_out[:self.belief_dim]
        dir_b = split_out[self.belief_dim:]

        # Normalize and scale
        parent_angle = parent / max(parent_radius, EPSILON)
        # Children diverge from parent by the learned perturbation
        child_a_angle = F.normalize(parent_angle + 0.1 * dir_a, dim=-1, eps=EPSILON)
        child_b_angle = F.normalize(parent_angle + 0.1 * dir_b, dim=-1, eps=EPSILON)

        # Split precision: each child gets radius/sqrt(2) (energy conservation)
        child_radius = parent_radius / math.sqrt(2)

        child_a_vec = child_a_angle * child_radius
        child_b_vec = child_b_angle * child_radius

        # Reuse parent slot for child A
        with torch.no_grad():
            state.beliefs.data[belief_idx] = child_a_vec
            state.belief_level[belief_idx] = 0  # reset to raw
            state.belief_precision_var[belief_idx] = 1.0  # reset uncertainty
            state.belief_reinforcement_count[belief_idx] = 0
            state.belief_source_type[belief_idx] = 1  # source type: merge/split
            state.belief_sources[belief_idx, 0] = belief_idx  # self-referential

        # Allocate new slot for child B
        child_b_idx = state.allocate_belief(
            child_b_vec,
            source_type=1,  # merge/split
            source_ids=[belief_idx],
        )

        return (belief_idx, child_b_idx)

    def prune_belief(self, state, belief_idx: int):
        """Prune a dead belief by deallocating its slot.

        Also removes edges connected to this belief.

        Args:
            state: CognitiveState
            belief_idx: index of belief to prune
        """
        state.deallocate_belief(belief_idx)

        # Clean up connected edges
        with torch.no_grad():
            edge_mask = state.edge_active & (
                (state.edge_src == belief_idx) | (state.edge_tgt == belief_idx)
            )
            if edge_mask.any():
                edge_indices = edge_mask.nonzero(as_tuple=False).squeeze(-1)
                for eidx in edge_indices.tolist():
                    state.deallocate_edge(eidx)

    def reset_stats(self, indices: torch.Tensor | None = None):
        """Reset activation statistics for given beliefs (or all)."""
        with torch.no_grad():
            if indices is None:
                self.activation_count.zero_()
                self.activation_entropy.zero_()
                self.context_signatures.zero_()
            else:
                self.activation_count[indices] = 0
                self.activation_entropy[indices] = 0
                self.context_signatures[indices] = 0


def run_structural_plasticity(
    state,
    plasticity: StructuralPlasticity,
    max_splits: int = 5,
    max_prunes: int = 10,
) -> dict:
    """Run one step of structural plasticity evaluation and action.

    Called periodically from pass2 (not every step — only when fill ratio
    is moderate and there's enough activation data).

    Args:
        state: CognitiveState
        plasticity: StructuralPlasticity module
        max_splits: cap on splits per step (prevents explosive growth)
        max_prunes: cap on prunes per step

    Returns:
        dict with statistics
    """
    split_threshold = state.meta_params.plasticity_split_threshold
    prune_threshold = state.meta_params.plasticity_prune_threshold

    # Update entropy estimates
    plasticity.compute_activation_entropy(state)

    # Evaluate
    result = plasticity.evaluate(state, split_threshold, prune_threshold)

    # Execute splits (capped)
    splits_done = 0
    for bidx in result['split_candidates'][:max_splits]:
        a, b = plasticity.split_belief(state, bidx)
        if b >= 0:
            splits_done += 1
            plasticity.reset_stats(torch.tensor([bidx, b], device=state.beliefs.device))

    # Execute prunes (capped)
    prunes_done = 0
    for bidx in result['prune_candidates'][:max_prunes]:
        plasticity.prune_belief(state, bidx)
        prunes_done += 1

    return {
        'split_candidates': len(result['split_candidates']),
        'splits_executed': splits_done,
        'prune_candidates': len(result['prune_candidates']),
        'prunes_executed': prunes_done,
        'should_grow': result['should_grow'],
        'growth_pressure': result['growth_pressure'],
    }
