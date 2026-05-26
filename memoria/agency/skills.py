"""D4: Skill Crystallization + differentiable procedural memory.

When the daemon loop detects recurring successful action patterns:
1. **Detect**: 3+ similar episodes with positive outcomes (threshold from MetaParams)
2. **Extract**: identify the common belief pattern
3. **Disentangle**: ensure each skill component affects one state factor (DUSDi)
4. **Store**: crystallize as tensor pattern in skill bank
5. **Compose**: skills combine via vector operations (not sequence concatenation)
6. **Evolve**: free energy gradient refines skill parameters
7. **Transfer**: canonical tensor format enables sharing between instances

Skills are stored as [max_skills, D] embeddings in a persistent bank.  Routing
is soft/sparse and differentiable: active skills are selected by similarity to
the current belief context, blended into a skill bias, and trained by learned
outcome/transition prediction losses. External verifiers are labels for those
losses; they are not embedded as hand-scored reward tables.

Reference: Option-Critic — learned initiation/termination for temporally
    extended skills.
Reference: entmax / DSelect-k — differentiable sparse routing.
Reference: successor features / contrastive skill discovery — credit assignment
    through predicted state transitions instead of fixed scalar rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.polar import EPSILON


def _entmax_or_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparse differentiable routing with a numerically safe fallback."""
    logits = logits.clamp(-30.0, 30.0)
    has_bad = bool((torch.isnan(logits).any() | torch.isinf(logits).any()).item())
    span = logits.max(dim=dim).values - logits.min(dim=dim).values
    degenerate = bool(torch.all(span.abs() < EPSILON).item())
    if not has_bad and not degenerate:
        try:
            from entmax import entmax15
            weights = entmax15(logits, dim=dim)
            if torch.isfinite(weights).all() and weights.sum(dim=dim).min().item() > 0:
                return weights
        except (ImportError, RuntimeError):
            pass
    return F.softmax(logits.nan_to_num(0.0), dim=dim)


class SkillBank(nn.Module):
    """Persistent bank of crystallized skills.

    Each skill is a tensor pattern in belief space that can be composed
    with other skills via vector addition. The bank maintains utility
    tracking and automatic pruning of low-utility skills.

    Args:
        belief_dim: dimension of belief/skill vectors
        max_skills: maximum number of stored skills
    """

    def __init__(self, belief_dim: int, max_skills: int = 128):
        super().__init__()
        self.belief_dim = belief_dim
        self.max_skills = max_skills

        # Skill embeddings (learnable, updated by gradient + crystallization)
        self.skill_embeddings = nn.Parameter(
            torch.zeros(max_skills, belief_dim),
            requires_grad=True,
        )

        # Per-skill metadata (not gradient-trained)
        self.register_buffer('skill_active', torch.zeros(max_skills, dtype=torch.bool))
        self.register_buffer('skill_utility', torch.zeros(max_skills))
        self.register_buffer('skill_use_count', torch.zeros(max_skills))
        self.register_buffer('skill_created_step', torch.zeros(max_skills))
        self.register_buffer('skill_last_used', torch.zeros(max_skills))

        outcome_in = belief_dim * 2 + 4
        hidden = max(32, belief_dim // 2)
        self.outcome_head = nn.Sequential(
            nn.Linear(outcome_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(outcome_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, belief_dim),
        )

        # Side-loss buffers. Predictions stay attached to autograd; targets are
        # observed continuous outcomes supplied by pass2/tool-result encoders.
        self._outcome_pairs: list[tuple[Tensor, float]] = []
        self._transition_pairs: list[tuple[Tensor, Tensor]] = []
        self._router_entropy_terms: list[Tensor] = []
        self._pending_transition: tuple[Tensor, Tensor] | None = None
        self._max_side_buffer = 64

    def num_active_skills(self) -> int:
        return self.skill_active.sum().item()

    def get_active_skills(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get active skill indices and embeddings.

        Returns:
            indices: [K] active skill indices
            embeddings: [K, D] skill vectors
        """
        indices = self.skill_active.nonzero(as_tuple=False).squeeze(-1)
        return indices, self.skill_embeddings[indices]

    def allocate_skill(self, embedding: torch.Tensor, step: int = 0) -> int:
        """Allocate a new skill slot.

        Args:
            embedding: [D] skill vector
            step: current step for creation tracking

        Returns:
            slot index, or -1 if full
        """
        empty = (~self.skill_active).nonzero(as_tuple=False)
        if len(empty) == 0:
            return -1
        slot = empty[0].item()
        with torch.no_grad():
            self.skill_embeddings.data[slot] = embedding
            self.skill_active[slot] = True
            self.skill_utility[slot] = 0.5  # neutral initial utility
            self.skill_use_count[slot] = 0
            self.skill_created_step[slot] = float(step)
            self.skill_last_used[slot] = float(step)
        return slot

    def deallocate_skill(self, index: int):
        """Free a skill slot."""
        with torch.no_grad():
            self.skill_active[index] = False
            self.skill_embeddings.data[index].zero_()
            self.skill_utility[index] = 0.0
            self.skill_use_count[index] = 0

    def update_utility(self, skill_idx: int, fe_improvement: float, decay: float = 0.9):
        """Update skill utility via EMA of free energy improvement.

        Args:
            skill_idx: skill index
            fe_improvement: positive = skill helped reduce FE
            decay: EMA decay factor
        """
        with torch.no_grad():
            old = self.skill_utility[skill_idx].item()
            self.skill_utility[skill_idx] = decay * old + (1 - decay) * fe_improvement
            self.skill_use_count[skill_idx] += 1

    def update_routed_utility(
        self,
        skill_indices: Tensor,
        skill_weights: Tensor,
        outcome: float,
        decay: float = 0.9,
        step: int | None = None,
    ) -> None:
        """Attribute a continuous observed outcome to softly routed skills."""
        if skill_indices.numel() == 0:
            return
        weights = skill_weights.detach()
        with torch.no_grad():
            for idx, weight in zip(skill_indices.tolist(), weights.tolist()):
                self.update_utility(idx, outcome * weight, decay=decay)
                if step is not None:
                    self.skill_last_used[idx] = float(step)

    def route_skills(
        self,
        context: Tensor,
        temperature: Tensor,
    ) -> dict[str, Tensor]:
        """Soft-route active skills against the current belief context.

        Returns active skill indices, differentiable weights, and the blended
        skill bias. No hard top-k is used, so outcome losses can assign credit
        through the routing weights and skill embeddings.
        """
        device = self.skill_embeddings.device
        context = context.to(device=device, dtype=self.skill_embeddings.dtype)
        indices, active_skills = self.get_active_skills()
        if indices.numel() == 0:
            zero_bias = self.skill_embeddings.sum(dim=0) * 0.0
            zero = zero_bias.sum() * 0.0
            return {
                'indices': indices,
                'weights': torch.zeros(0, device=device, dtype=zero_bias.dtype),
                'skill_bias': zero_bias,
                'entropy': zero,
                'max_weight': zero,
                'logits': torch.zeros(0, device=device, dtype=zero_bias.dtype),
            }

        context_norm = F.normalize(context, dim=-1, eps=EPSILON)
        skill_norm = F.normalize(active_skills, dim=-1, eps=EPSILON)
        temp = temperature.to(device=device, dtype=active_skills.dtype).clamp(min=EPSILON)
        logits = (skill_norm @ context_norm) / temp
        weights = _entmax_or_softmax(logits, dim=-1)
        weights = weights / weights.sum().clamp(min=EPSILON)
        skill_bias = weights @ active_skills
        probs = weights.clamp(min=EPSILON)
        entropy = -(probs * probs.log()).sum()

        return {
            'indices': indices,
            'weights': weights,
            'skill_bias': skill_bias,
            'entropy': entropy,
            'max_weight': weights.max(),
            'logits': logits,
        }

    def _outcome_features(self, context: Tensor, skill_bias: Tensor,
                          weights: Tensor) -> Tensor:
        if weights.numel() > 0:
            probs = weights.clamp(min=EPSILON)
            entropy = -(probs * probs.log()).sum()
            max_weight = weights.max()
            active_frac = torch.tensor(
                weights.numel() / max(self.max_skills, 1),
                device=context.device,
                dtype=context.dtype,
            )
        else:
            entropy = context.sum() * 0.0
            max_weight = context.sum() * 0.0
            active_frac = context.sum() * 0.0
        bias_norm = skill_bias.norm().unsqueeze(0)
        return torch.cat([
            context,
            skill_bias,
            entropy.unsqueeze(0),
            max_weight.unsqueeze(0),
            active_frac.unsqueeze(0),
            bias_norm,
        ])

    def record_outcome_prediction(
        self,
        context: Tensor,
        skill_bias: Tensor,
        weights: Tensor,
        observed_outcome: float,
    ) -> None:
        """Record differentiable predictions against a continuous outcome.

        `observed_outcome` can come from belief advantage during pretraining or
        from an encoded tool/test result later. The bank learns to predict it;
        no verifier-specific score table is baked into the architecture.
        """
        context = context.detach().to(
            device=self.skill_embeddings.device,
            dtype=self.skill_embeddings.dtype,
        )
        skill_bias = skill_bias.to(device=context.device, dtype=context.dtype)
        weights = weights.to(device=context.device, dtype=context.dtype)
        features = self._outcome_features(context, skill_bias, weights)

        pred_outcome = self.outcome_head(features.unsqueeze(0)).squeeze()
        self._outcome_pairs.append((pred_outcome, float(observed_outcome)))

        pred_delta = self.transition_head(features.unsqueeze(0)).squeeze(0)
        if self._pending_transition is not None:
            prev_pred_delta, prev_context = self._pending_transition
            target_delta = context - prev_context
            self._transition_pairs.append((prev_pred_delta, target_delta.detach()))
        self._pending_transition = (pred_delta, context.detach())

        if weights.numel() > 0:
            self._router_entropy_terms.append(
                -(weights.clamp(min=EPSILON) * weights.clamp(min=EPSILON).log()).sum()
            )

        if len(self._outcome_pairs) > self._max_side_buffer:
            self._outcome_pairs = self._outcome_pairs[-self._max_side_buffer:]
        if len(self._transition_pairs) > self._max_side_buffer:
            self._transition_pairs = self._transition_pairs[-self._max_side_buffer:]
        if len(self._router_entropy_terms) > self._max_side_buffer:
            self._router_entropy_terms = self._router_entropy_terms[-self._max_side_buffer:]

    def compute_side_loss(
        self,
        outcome_weight: Tensor,
        transition_weight: Tensor,
        router_entropy_weight: Tensor,
    ) -> Tensor:
        """Train routing credit with continuous outcome/transition losses."""
        zero = self.skill_embeddings.sum() * 0.0
        zero = zero + sum(p.sum() * 0.0 for p in self.outcome_head.parameters())
        zero = zero + sum(p.sum() * 0.0 for p in self.transition_head.parameters())

        total = zero
        if self._outcome_pairs:
            preds = torch.stack([p for p, _ in self._outcome_pairs])
            targets = torch.tensor(
                [t for _, t in self._outcome_pairs],
                device=preds.device,
                dtype=preds.dtype,
            )
            total = total + outcome_weight * F.mse_loss(preds, targets)
            self._outcome_pairs.clear()

        if self._transition_pairs:
            pred_delta = torch.stack([p for p, _ in self._transition_pairs])
            target_delta = torch.stack([t for _, t in self._transition_pairs]).to(
                device=pred_delta.device,
                dtype=pred_delta.dtype,
            )
            total = total + transition_weight * F.mse_loss(pred_delta, target_delta)
            self._transition_pairs.clear()

        if self._router_entropy_terms:
            entropy = torch.stack(self._router_entropy_terms).mean()
            total = total + router_entropy_weight * entropy
            self._router_entropy_terms.clear()

        return total


class SkillDetector(nn.Module):
    """Detects recurring patterns that should be crystallized as skills.

    Maintains a buffer of recent successful action-belief patterns and
    clusters them to find recurrences. When a cluster reaches the
    detection threshold (from MetaParams), the centroid is crystallized.

    Args:
        belief_dim: dimension of belief vectors
        buffer_size: how many recent patterns to remember
    """

    def __init__(self, belief_dim: int, buffer_size: int = 256):
        super().__init__()
        self.belief_dim = belief_dim
        self.buffer_size = buffer_size

        # Circular buffer of successful patterns
        self.register_buffer('pattern_buffer', torch.zeros(buffer_size, belief_dim))
        self.register_buffer('pattern_rewards', torch.zeros(buffer_size))
        self.register_buffer('_buffer_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_buffer_count', torch.tensor(0, dtype=torch.long))

        # Pattern encoder: compress action-belief pattern to skill-space
        self.encoder = nn.Sequential(
            nn.Linear(belief_dim, belief_dim // 2),
            nn.GELU(),
            nn.Linear(belief_dim // 2, belief_dim),
        )

    def record_pattern(self, pattern: torch.Tensor, reward: float):
        """Record a successful action-belief pattern.

        Args:
            pattern: [D] belief state during successful action
            reward: how successful (positive = good)
        """
        if reward <= 0:
            return  # only record successes

        with torch.no_grad():
            ptr = self._buffer_ptr.item()
            self.pattern_buffer[ptr] = pattern.detach()
            self.pattern_rewards[ptr] = reward
            self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
            self._buffer_count = min(self._buffer_count + 1, self.buffer_size)

    def detect_skills(
        self,
        similarity_threshold: torch.Tensor,
        detection_threshold: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Detect recurring patterns in the buffer.

        Uses greedy clustering: find the densest point, collect its
        neighborhood, check if the cluster is large enough.

        Args:
            similarity_threshold: cosine sim for "same pattern" (MetaParams)
            detection_threshold: min cluster size for crystallization (MetaParams)

        Returns:
            list of [D] centroid vectors for detected skills
        """
        n = self._buffer_count.item()
        if n < 3:  # need at least 3 patterns
            return []

        patterns = self.pattern_buffer[:n]  # [n, D]
        rewards = self.pattern_rewards[:n]  # [n]

        # Encode patterns
        encoded = self.encoder(patterns)  # [n, D]
        encoded_norm = F.normalize(encoded, dim=-1, eps=EPSILON)

        # Similarity matrix
        sim = encoded_norm @ encoded_norm.T  # [n, n]

        # Greedy clustering
        min_cluster = max(3, int(detection_threshold.item() * 3))
        used = torch.zeros(n, dtype=torch.bool, device=patterns.device)
        detected = []

        for _ in range(10):  # max 10 skills per detection round
            if used.all():
                break

            # Find densest point (most neighbors above threshold)
            neighbor_count = ((sim > similarity_threshold) & ~used.unsqueeze(0)).sum(dim=1)
            neighbor_count[used] = 0

            if neighbor_count.max().item() < min_cluster:
                break

            center = neighbor_count.argmax().item()
            cluster_mask = (sim[center] > similarity_threshold) & ~used
            cluster_indices = cluster_mask.nonzero(as_tuple=False).squeeze(-1)

            if len(cluster_indices) >= min_cluster:
                # Weighted centroid (weight by reward)
                cluster_patterns = encoded[cluster_indices]
                cluster_rewards = rewards[cluster_indices].clamp(min=EPSILON)
                weights = cluster_rewards / cluster_rewards.sum()
                centroid = (cluster_patterns * weights.unsqueeze(-1)).sum(dim=0)
                detected.append(centroid)
                used[cluster_indices] = True
            else:
                used[center] = True

        return detected


class SkillComposer(nn.Module):
    """Composes multiple skills via learned vector operations.

    Simple composition: vector addition in skill space.
    Learned composition: gated addition with compatibility scoring.

    Reference: DUSDi (arXiv:2410.11251) — each skill affects one factor.

    Args:
        belief_dim: dimension of skill vectors
    """

    def __init__(self, belief_dim: int):
        super().__init__()
        self.belief_dim = belief_dim

        # Compatibility gate: do these skills work together?
        self.compatibility = nn.Sequential(
            nn.Linear(belief_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def compose(
        self,
        skill_a: torch.Tensor,
        skill_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose two skills with learned compatibility gating.

        Args:
            skill_a: [D] first skill embedding
            skill_b: [D] second skill embedding

        Returns:
            composed: [D] composed skill vector
            compatibility: scalar ∈ (0, 1)
        """
        pair = torch.cat([skill_a, skill_b]).unsqueeze(0)
        compat = self.compatibility(pair).squeeze()

        # Gated addition: high compatibility → full blend, low → keep A
        composed = skill_a + compat * skill_b
        return composed, compat

    def compose_multiple(self, skills: torch.Tensor) -> torch.Tensor:
        """Compose multiple skills via sequential pairwise composition.

        Args:
            skills: [K, D] skill embeddings to compose

        Returns:
            [D] composed skill vector
        """
        if skills.shape[0] == 0:
            return torch.zeros(self.belief_dim, device=skills.device)
        if skills.shape[0] == 1:
            return skills[0]

        result = skills[0]
        for i in range(1, skills.shape[0]):
            result, _ = self.compose(result, skills[i])
        return result


def run_skill_step(
    state,
    skill_bank: SkillBank,
    detector: SkillDetector,
    composer: SkillComposer,
    current_step: int,
) -> dict:
    """Run one skill crystallization and management step.

    Called periodically from pass2 or daemon loop.

    1. Detect new skills from pattern buffer
    2. Crystallize detected skills into bank
    3. Prune low-utility skills
    4. Match current goals to available skills

    Args:
        state: CognitiveState
        skill_bank: SkillBank instance
        detector: SkillDetector instance
        composer: SkillComposer instance
        current_step: current training step

    Returns:
        dict with skill management statistics
    """
    stats = {'skills_detected': 0, 'skills_crystallized': 0, 'skills_pruned': 0}

    sim_threshold = state.meta_params.skill_similarity_threshold
    det_threshold = state.meta_params.skill_detection_threshold

    # 1. Detect new skills
    new_skills = detector.detect_skills(sim_threshold, det_threshold)
    stats['skills_detected'] = len(new_skills)

    # 2. Crystallize: check they're not duplicates of existing skills
    if new_skills:
        active_idx, active_skills = skill_bank.get_active_skills()
        for skill_vec in new_skills:
            is_duplicate = False
            if len(active_idx) > 0:
                # Check similarity to existing skills
                sim = F.cosine_similarity(
                    skill_vec.unsqueeze(0),
                    active_skills,
                    dim=-1,
                )
                if sim.max().item() > sim_threshold.item():
                    is_duplicate = True

            if not is_duplicate:
                slot = skill_bank.allocate_skill(skill_vec.detach(), step=current_step)
                if slot >= 0:
                    stats['skills_crystallized'] += 1

    # 3. Prune low-utility skills (below threshold, not recently used)
    prune_threshold = state.meta_params.plasticity_prune_threshold.item()
    active_idx, _ = skill_bank.get_active_skills()
    for idx in active_idx.tolist():
        utility = skill_bank.skill_utility[idx].item()
        use_count = skill_bank.skill_use_count[idx].item()
        age = current_step - skill_bank.skill_created_step[idx].item()
        # Prune if: low utility AND at least 10 uses AND older than 100 steps
        if utility < prune_threshold and use_count > 10 and age > 100:
            skill_bank.deallocate_skill(idx)
            stats['skills_pruned'] += 1

    stats['total_active_skills'] = skill_bank.num_active_skills()
    return stats
