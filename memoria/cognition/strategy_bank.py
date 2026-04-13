"""LSR Strategy Bank: learned reasoning-mode perturbations for refinement loops.

Implements the core insight from Latent Space Reasoning (dl1683/Latent-Space-
Reasoning): diverse perturbations in hidden space cause the model to explore
different reasoning trajectories, and the union covers more solution space than
any single trajectory. Instead of random noise (LSR's approach), Memoria learns
which perturbation modes help for which goals and persists them in the cognitive
state.

Three tiers, each building on the previous:

Tier 1 — Orthogonal perturbation bank: fixed-size bank of orthonormal vectors
    in R^n_embd. Each refinement iteration uses a different vector, guaranteeing
    maximal directional diversity. Initialized via QR decomposition (Haar-random
    orthonormal basis on the Stiefel manifold).

Tier 2 — Goal-conditioned selection (StrategySelector): instead of cycling
    strategies by iteration index, a learned selector picks strategies based on
    (hidden_state, goal_embedding). Uses entmax15 for sparse selection — each
    iteration commits to 1-2 strategies rather than averaging all of them.

Tier 3 — Strategy evolution in Pass 2 (StrategyEvolver): strategies evolve
    through fitness tracking, re-orthogonalization, and autoresearch-style
    hypothesis generation. Failed strategies feed a per-goal ring buffer that
    conditions future generation (same pattern as HypothesisTracker).

Reference: Latent Space Reasoning (github.com/dl1683/Latent-Space-Reasoning)
    — trajectory perturbation, orthogonal W projection, evolutionary search
Reference: Johnson-Lindenstrauss (1984) — random projection distance preservation
Reference: Meta-Harness (arXiv:2603.28052) — failed-hypothesis conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.polar import EPSILON


class StrategySelector(nn.Module):
    """Goal-conditioned strategy selection for refinement loops.

    Takes the current hidden state summary and active goal embedding, produces
    sparse weights over the strategy bank via entmax15. The weighted combination
    of strategies becomes the perturbation injected into the refinement loop.

    Zero-init: the projection starts at zero weights, so entmax15 sees uniform
    logits → uniform weights → perturbation is mean of orthogonal bank → zero.
    Backward compatible at t=0; learns to specialize over training.
    """

    def __init__(self, hidden_dim: int, belief_dim: int, max_strategies: int):
        """
        Args:
            hidden_dim: transformer hidden dimension (n_embd)
            belief_dim: belief/goal embedding dimension
            max_strategies: number of strategies in the bank
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.max_strategies = max_strategies

        # Input: [hidden_pooled, goal_embed, loop_fraction]
        # The loop_fraction lets the selector learn iteration-dependent
        # strategy preferences (e.g., broad exploration early, focused late).
        in_dim = hidden_dim + belief_dim + 1

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, max_strategies),
        )
        # Zero-init final layer → uniform logits at t=0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        hidden_pooled: Tensor,
        goal_embed: Tensor,
        loop_fraction: float,
        strategy_bank: Tensor,
    ) -> Tensor:
        """Select and blend strategies from the bank.

        Args:
            hidden_pooled: [B, hidden_dim] mean-pooled hidden state
            goal_embed: [belief_dim] active goal embedding (mean of active goals)
            loop_fraction: scalar in [0, 1], current loop index / max loops
            strategy_bank: [max_strategies, hidden_dim] the strategy vectors

        Returns:
            [B, hidden_dim] perturbation vector to inject into residual stream
        """
        try:
            from entmax import entmax15
        except ImportError:
            entmax15 = None

        B = hidden_pooled.shape[0]
        device = hidden_pooled.device

        # Expand goal_embed to batch
        goal_expanded = goal_embed.unsqueeze(0).expand(B, -1)  # [B, belief_dim]
        loop_t = torch.full((B, 1), loop_fraction, device=device)

        features = torch.cat([hidden_pooled, goal_expanded, loop_t], dim=-1)
        logits = self.proj(features)  # [B, max_strategies]

        # Sparse selection: entmax15 zeros out low-relevance strategies,
        # forcing each iteration to commit to a specific reasoning mode.
        if entmax15 is not None:
            weights = entmax15(logits, dim=-1)  # [B, max_strategies]
        else:
            weights = F.softmax(logits, dim=-1)

        # Weighted combination of strategies
        # [B, max_strategies] @ [max_strategies, hidden_dim] → [B, hidden_dim]
        perturbation = weights @ strategy_bank

        return perturbation


class StrategyEvolver(nn.Module):
    """Pass 2 evolution of the strategy bank.

    Handles:
    1. Fitness tracking: EMA of per-strategy FE improvement attribution
    2. Re-orthogonalization: periodic Gram-Schmidt to maintain diversity
    3. Strategy hypothesis generation: synthesize new candidate strategies
       from goal embeddings + failure conditioning
    4. Failed-strategy logging: ring buffer per goal, conditions generation
    """

    def __init__(self, hidden_dim: int, belief_dim: int, max_strategies: int,
                 max_goals: int, failed_buffer_depth: int):
        """
        Args:
            hidden_dim: transformer hidden dimension
            belief_dim: belief/goal embedding dimension
            max_strategies: strategy bank size
            max_goals: max concurrent goals
            failed_buffer_depth: ring buffer depth for failed strategies per goal
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.max_strategies = max_strategies
        self.max_goals = max_goals
        self.failed_buffer_depth = failed_buffer_depth

        # Strategy hypothesis generator: goal → candidate strategy direction
        # Input: [goal_embed (projected to hidden_dim), belief_summary (hidden_dim),
        #         failure_summary (hidden_dim), beta, failure_count_norm]
        gen_in_dim = hidden_dim * 3 + 2
        self.strategy_gen = nn.Sequential(
            nn.Linear(gen_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate: should we generate a strategy for this goal?
        self.gen_gate = nn.Sequential(
            nn.Linear(gen_in_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Starts mostly closed — strategies are expensive to test
        nn.init.constant_(self.gen_gate[-1].bias, -2.0)

        # Project belief_dim → hidden_dim for goal/failure embeddings
        # (strategies live in hidden_dim space, goals live in belief_dim space)
        self.goal_proj = nn.Linear(belief_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.goal_proj.weight)

        # ── Failed-strategy ring buffer (per goal) ──
        # Stores evicted strategy directions so the generator can push away
        # from recently-failed reasoning modes for each goal.
        self.register_buffer(
            'failed_strategies',
            torch.zeros(max_goals, failed_buffer_depth, hidden_dim),
        )
        self.register_buffer(
            'failed_fe_deltas',
            torch.zeros(max_goals, failed_buffer_depth),
        )
        self.register_buffer(
            'failed_write_idx',
            torch.zeros(max_goals, dtype=torch.long),
        )
        self.register_buffer(
            'failed_count',
            torch.zeros(max_goals, dtype=torch.long),
        )

        # Zero-init the failure-conditioning columns of the generator.
        # Input layout:
        #   [0 : H]         goal_embed (projected)
        #   [H : 2H]        belief_summary
        #   [2H : 3H]       failure_summary    ← zero-init
        #   [3H]             beta
        #   [3H+1]           failure_count_norm ← zero-init
        with torch.no_grad():
            for head in (self.strategy_gen, self.gen_gate):
                first_linear = head[0]
                first_linear.weight[:, hidden_dim * 2: hidden_dim * 3].zero_()
                first_linear.weight[:, hidden_dim * 3 + 1].zero_()

    def update_fitness(
        self,
        strategy_bank: Tensor,
        strategy_fitness: Tensor,
        strategy_weights: Tensor,
        fe_delta: float,
        ema_decay: float,
    ):
        """Update per-strategy fitness EMA based on FE improvement.

        Attribution is proportional to the selection weight each strategy
        received during the refinement loop. If the FE decreased (good),
        strategies with high weight get positive fitness; if FE increased
        (bad), they get negative fitness.

        Args:
            strategy_bank: [max_strategies, hidden_dim] (unused, for API consistency)
            strategy_fitness: [max_strategies] current fitness EMA (modified in-place)
            strategy_weights: [max_strategies] accumulated selection weights from
                this step's refinement loop (sum of per-iteration entmax weights)
            fe_delta: FE(after refinement) - FE(before refinement), negative = good
            ema_decay: EMA decay rate (from MetaParams)
        """
        with torch.no_grad():
            # Normalize weights to sum to 1 (attribution shares)
            total_weight = strategy_weights.sum().clamp(min=EPSILON)
            attribution = strategy_weights / total_weight

            # Negate fe_delta so positive = improvement (lower FE)
            reward = -fe_delta
            per_strategy_reward = attribution * reward

            # EMA update
            strategy_fitness.mul_(ema_decay).add_(
                per_strategy_reward, alpha=(1.0 - ema_decay)
            )

    def reorthogonalize(
        self,
        strategy_bank: nn.Parameter,
        strategy_fitness: Tensor,
        min_similarity: float,
    ) -> int:
        """Re-orthogonalize strategy bank via modified Gram-Schmidt.

        Only runs when max pairwise cosine similarity exceeds the threshold.
        Preserves the ordering of strategies by fitness (highest-fitness
        strategy is the first Gram-Schmidt pivot, so it's unchanged).

        Args:
            strategy_bank: [max_strategies, hidden_dim] parameter (modified in-place)
            strategy_fitness: [max_strategies] fitness values for ordering
            min_similarity: threshold — skip if max similarity is below this

        Returns:
            Number of strategies that were modified (0 if skipped)
        """
        with torch.no_grad():
            bank = strategy_bank.data
            N = bank.shape[0]

            # Check if re-orthogonalization is needed
            norms = bank.norm(dim=-1, keepdim=True).clamp(min=EPSILON)
            normalized = bank / norms
            sim_matrix = normalized @ normalized.T
            # Zero diagonal
            sim_matrix.fill_diagonal_(0.0)
            max_sim = sim_matrix.abs().max().item()

            if max_sim < min_similarity:
                return 0

            # Order by fitness (highest first → pivot is the best strategy)
            order = strategy_fitness.argsort(descending=True)
            bank_ordered = bank[order].clone()

            # Modified Gram-Schmidt
            n_modified = 0
            for i in range(N):
                for j in range(i):
                    proj = (bank_ordered[i] @ bank_ordered[j]) / (
                        bank_ordered[j] @ bank_ordered[j]
                    ).clamp(min=EPSILON)
                    bank_ordered[i] -= proj * bank_ordered[j]
                # Preserve original norm (don't normalize — norm carries
                # information about strategy strength)
                new_norm = bank_ordered[i].norm().clamp(min=EPSILON)
                original_norm = norms[order[i]].squeeze()
                bank_ordered[i] *= original_norm / new_norm
                if i > 0:
                    n_modified += 1

            # Un-order back to original indices
            bank[order] = bank_ordered
            return n_modified

    def generate_strategy_hypotheses(
        self,
        goal_embeddings: Tensor,
        goal_indices: Tensor,
        belief_summary: Tensor,
        beta: float,
    ) -> tuple[Tensor, Tensor]:
        """Generate candidate strategy vectors from active goals.

        Similar to HypothesisGenerator but produces hidden_dim vectors
        (strategy space) instead of belief_dim vectors (belief space).

        Args:
            goal_embeddings: [G, belief_dim] active goal embeddings
            goal_indices: [G] global goal slot indices
            belief_summary: [hidden_dim] precision-weighted mean of active beliefs
                projected to hidden space
            beta: exploration/exploitation parameter

        Returns:
            strategies: [K, hidden_dim] candidate strategy vectors (K <= G)
            source_goals: [K] which goal generated each strategy
        """
        G = goal_embeddings.shape[0]
        device = goal_embeddings.device

        if G == 0:
            return (torch.zeros(0, self.hidden_dim, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        # Project goals to hidden space
        goal_h = self.goal_proj(goal_embeddings)  # [G, hidden_dim]

        # Belief summary broadcast
        belief_h = belief_summary.unsqueeze(0).expand(G, -1)  # [G, hidden_dim]

        # Failure conditioning
        failure_summary, failure_count_norm = self.get_failure_summary(
            goal_indices
        )

        beta_t = torch.full((G, 1), beta, device=device)

        features = torch.cat([
            goal_h,             # [G, hidden_dim]
            belief_h,           # [G, hidden_dim]
            failure_summary,    # [G, hidden_dim]
            beta_t,             # [G, 1]
            failure_count_norm, # [G, 1]
        ], dim=-1)  # [G, 3H + 2]

        # Gate
        gate_logits = self.gen_gate(features).squeeze(-1)  # [G]
        gate = torch.sigmoid(gate_logits)
        generate_mask = gate > 0.5

        if not generate_mask.any():
            return (torch.zeros(0, self.hidden_dim, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        selected = features[generate_mask]
        selected_indices = goal_indices[generate_mask]

        # Generate strategy direction and normalize
        raw = self.strategy_gen(selected)  # [K, hidden_dim]
        strategies = F.normalize(raw, dim=-1, eps=EPSILON)

        return strategies, selected_indices

    def push_failure(
        self,
        goal_idx: int,
        strategy_direction: Tensor,
        fe_delta: float,
    ):
        """Record a failed strategy for a goal into the ring buffer."""
        with torch.no_grad():
            if goal_idx < 0 or goal_idx >= self.max_goals:
                return
            write_idx = int(self.failed_write_idx[goal_idx].item())
            self.failed_strategies[goal_idx, write_idx] = strategy_direction.to(
                self.failed_strategies.device
            )
            self.failed_fe_deltas[goal_idx, write_idx] = float(fe_delta)
            self.failed_write_idx[goal_idx] = (
                (write_idx + 1) % self.failed_buffer_depth
            )
            self.failed_count[goal_idx] = min(
                int(self.failed_count[goal_idx].item()) + 1,
                self.failed_buffer_depth,
            )

    def get_failure_summary(
        self, goal_indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Get per-goal failure conditioning for the strategy generator.

        Returns:
            summary: [G, hidden_dim] mean of stored failed strategies per goal
            count_norm: [G, 1] fraction of buffer filled, in [0, 1]
        """
        device = self.failed_strategies.device
        G = len(goal_indices)
        summary = torch.zeros(G, self.hidden_dim, device=device)
        count_norm = torch.zeros(G, 1, device=device)

        for i in range(G):
            g = int(goal_indices[i].item())
            if g < 0 or g >= self.max_goals:
                continue
            n = int(self.failed_count[g].item())
            if n == 0:
                continue
            n_active = min(n, self.failed_buffer_depth)
            summary[i] = self.failed_strategies[g, :n_active].mean(dim=0)
            count_norm[i, 0] = n_active / self.failed_buffer_depth

        return summary, count_norm


def initialize_strategy_bank(
    max_strategies: int,
    hidden_dim: int,
    init_scale: float,
    device: torch.device | None = None,
) -> Tensor:
    """Initialize strategy bank with Haar-random orthonormal rows.

    Uses QR decomposition of a Gaussian random matrix to produce uniformly
    random orthonormal vectors (Haar measure on the Stiefel manifold V(k, n)).
    Scaled by init_scale so the perturbation magnitude is controlled.

    Mathematical guarantee: for max_strategies << hidden_dim (e.g., 8 << 768),
    the rows are exactly orthonormal. Each strategy is maximally different from
    every other — no two share any variance.

    Args:
        max_strategies: number of strategy vectors (rows)
        hidden_dim: dimension of each vector (n_embd)
        init_scale: magnitude scaling (controls perturbation strength at init)
        device: target device

    Returns:
        [max_strategies, hidden_dim] orthonormal matrix scaled by init_scale
    """
    # Draw from Gaussian → QR gives uniform random orthonormal basis
    W = torch.randn(hidden_dim, max_strategies, device=device)
    Q, _ = torch.linalg.qr(W)  # [hidden_dim, max_strategies], columns orthonormal
    # Take first max_strategies columns, transpose to [max_strategies, hidden_dim]
    bank = Q[:, :max_strategies].T.contiguous()
    # Scale: rows are unit-norm after QR, multiply by init_scale
    return bank * init_scale
