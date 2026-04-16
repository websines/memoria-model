"""Running statistics tracker for the cognitive state.

Tracks exponential moving averages of key cognitive signals and derives
adaptive thresholds from them — replacing hardcoded constants with
state-derived values.

All EMA buffers are registered via register_buffer so they are saved and
loaded with the model checkpoint but are not learnable parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from .state import CognitiveState


class RunningStats(nn.Module):
    """Tracks EMAs of cognitive signals and derives adaptive thresholds.

    Update call:
        stats.update(cognitive_state, pass2_stats)

    where pass2_stats is the dict returned by pass2 containing at minimum:
        - "mean_surprise": float / scalar tensor
        - "mean_precision": float / scalar tensor  (mean belief radius)

    All derived values are read as properties.

    Args:
        decay: EMA decay factor. 0.99 keeps history for ~100 steps.
        base_max_candidates: Upper bound on candidate beliefs for matching.
    """

    def __init__(self, decay: float = 0.99, base_max_candidates: int = 1024) -> None:
        super().__init__()

        self.decay = decay
        self.base_max_candidates = base_max_candidates

        # ── EMA buffers ──
        # Initialised to sensible neutral values so properties are stable
        # before the first update().

        # Mean prediction error / free-energy surprise from pass2
        self.register_buffer("mean_surprise", torch.tensor(1.0))

        # Mean belief radius (precision) across active beliefs
        self.register_buffer("mean_precision", torch.tensor(1.0))

        # Variance of surprise signal — small = system is stable
        self.register_buffer("surprise_variance", torch.tensor(0.0))

        # Mean number of steps from goal creation to completion
        self.register_buffer("mean_goal_completion_time", torch.tensor(50.0))

        # Fraction of belief slots that are occupied (active / max_beliefs)
        self.register_buffer("belief_fill_ratio", torch.tensor(0.0))

        # Mean access count across active beliefs (for promotion thresholds)
        self.register_buffer("mean_access_count", torch.tensor(10.0))

        # Track whether we have received at least one real update
        self.register_buffer("_initialised", torch.tensor(False))

    # ── Update ──────────────────────────────────────────────────────────────

    def update(
        self,
        cognitive_state: "CognitiveState",
        pass2_stats: dict[str, Any],
    ) -> None:
        """Apply one EMA step from the current cognitive state and pass2 stats.

        Args:
            cognitive_state: The live CognitiveState module.
            pass2_stats: Dict from pass2 containing signal measurements.
                         Unknown keys are silently ignored.
        """
        decay = self.decay

        def _ema(buf: Tensor, observed: float | Tensor) -> None:
            val = torch.as_tensor(observed, dtype=buf.dtype, device=buf.device)
            buf.copy_(decay * buf + (1.0 - decay) * val)

        # --- surprise ---
        if "mean_surprise" in pass2_stats:
            obs_surprise = float(pass2_stats["mean_surprise"])
            # EMA of variance: Var_new = decay * Var_old + (1-decay) * (x - mean)^2
            deviation = obs_surprise - self.mean_surprise.item()
            _ema(self.surprise_variance, deviation * deviation)
            _ema(self.mean_surprise, obs_surprise)

        # --- precision ---
        if "mean_precision" in pass2_stats:
            _ema(self.mean_precision, float(pass2_stats["mean_precision"]))
        else:
            # Derive from state directly when not provided by pass2
            radii = cognitive_state.get_belief_radii()
            active_mask = cognitive_state.get_active_mask()
            if active_mask.any():
                _ema(self.mean_precision, radii[active_mask].mean().item())

        # --- goal completion time ---
        # goal_metadata columns: [priority, urgency, progress, status, depth,
        #                          surprise_accum, created_step, deadline]
        # We look at recently-completed goals (status ≈ 0.8) and read
        # elapsed = current_step - created_step.  current_step is approximated
        # via the consolidation_timer meta slot; callers may pass it directly.
        current_step = pass2_stats.get("current_step", None)
        if current_step is not None:
            gm = cognitive_state.goal_metadata.data  # [max_goals, G]
            status_col = gm[:, 3]
            completed_mask = (status_col - 0.8).abs() < 0.05
            if completed_mask.any():
                created_steps = gm[completed_mask, 6]
                elapsed = float(current_step) - created_steps
                elapsed = elapsed.clamp(min=1.0)
                _ema(self.mean_goal_completion_time, elapsed.mean().item())

        # --- belief fill ratio ---
        max_beliefs = cognitive_state.config.max_beliefs
        active_mask = cognitive_state.get_active_mask()
        active_count = active_mask.sum().item()
        _ema(self.belief_fill_ratio, active_count / max(max_beliefs, 1))

        # --- mean access count ---
        if active_mask.any():
            _ema(
                self.mean_access_count,
                cognitive_state.belief_access_count[active_mask].mean().item(),
            )

        self._initialised.fill_(True)

    # ── Derived threshold properties ─────────────────────────────────────────

    @property
    def soft_consolidation_interval(self) -> int:
        """Steps between soft consolidation passes.

        Reduces from 10 to 5 when beliefs are near capacity so that the
        state is pruned more aggressively before it fills completely.
        """
        if self.belief_fill_ratio.item() > 0.9:
            return 5
        return 10

    @property
    def hard_consolidation_interval(self) -> int:
        """Steps between full (hard) consolidation passes.

        Scales with fill ratio: less frequent when sparse (state is healthy),
        more frequent near capacity (needs cleanup). Floor of 200 prevents
        the sleep/verification/recalibration/replay operations from stacking
        up every 20 steps with a dense state — at 8K beliefs and 17K edges,
        each operation takes seconds and running ALL of them every 20 steps
        causes 30+ minute pass2 hangs that timeout DDP barriers.

        200-step floor = ~1-2 consolidation passes per training phase
        transition, which is sufficient for state hygiene.
        """
        fill = self.belief_fill_ratio.item()
        # Linear scale: 500 at fill=0 → 200 at fill=1
        interval = int(500 - 300 * fill)
        return max(interval, 200)

    @property
    def max_candidates(self) -> int:
        """Maximum belief candidates considered during matching.

        Base value is configurable (default 1024).  Could be extended to
        scale with batch size if pass2_stats provides it.
        """
        return self.base_max_candidates

    @property
    def goal_cooldown_steps(self) -> int:
        """Minimum steps between new goal proposals.

        Longer cooldown when surprise is high (system is unstable) or when
        many goals are already active (avoid goal proliferation).
        """
        # Use mean_goal_completion_time as the natural timescale, then
        # scale by surprise so volatile periods suppress new goal creation.
        surprise_scale = float(
            self.mean_surprise.clamp(min=0.5, max=2.0).item()
        )
        base = max(int(self.mean_goal_completion_time.item() * 0.5), 20)
        return int(base * surprise_scale)

    @property
    def goal_threshold_scale(self) -> float:
        """Scaling factor replacing the literal `2.0` in `2.0 / (1 + beta)`.

        Raised when surprise is high (be more selective about new goals)
        and lowered when surprise is low (allow exploration).
        """
        return float(2.0 * self.mean_surprise.clamp(min=0.5, max=2.0).item())

    @property
    def max_new_goals_scale(self) -> int:
        """Maximum number of new goals that may be proposed per step.

        Derived from beta (meta[0]).  Low beta = exploitation = fewer new
        goals.  High beta = exploration = more new goals, up to a cap of 5.
        """
        # This property does not have access to the live state; callers
        # should pass beta explicitly or use the default of 3.
        # The value is intentionally kept simple — beta is managed outside.
        return 3

    @property
    def stall_threshold_base(self) -> int:
        """Base stall detection threshold in steps.

        Callers multiply this by an urgency factor (e.g. 1×, 2.5×, 5×, 10×)
        to get urgency-level-specific thresholds, replacing the hardcoded
        sequence [20, 50, 100, 200].

        Calibrated to mean goal completion time so that the system does not
        declare goals stalled before they typically finish.
        """
        base = int(self.mean_goal_completion_time.item() * 0.4)
        return max(base, 10)

    @property
    def merge_similarity_threshold(self) -> float:
        """Cosine similarity above which two beliefs are candidates for merge.

        Lowered when fill ratio is high to allow more aggressive deduplication
        before the state hits capacity.
        """
        fill = self.belief_fill_ratio.item()
        # 0.95 at fill=0 → 0.85 at fill=1
        threshold = 0.95 - 0.10 * fill
        return float(max(threshold, 0.80))

    @property
    def hard_cleanup_precision_threshold(self) -> float:
        """Minimum precision (radius) to keep a belief during hard cleanup.

        Raised when near capacity so that more low-confidence beliefs are
        evicted, freeing space for new, higher-quality beliefs.
        """
        fill = self.belief_fill_ratio.item()
        # 0.1 at fill=0 → 0.3 at fill=1
        threshold = 0.1 + 0.2 * fill
        return float(threshold)

    def promotion_thresholds(self, level: int) -> tuple[float, float]:
        """Return (min_radius, min_access) for promoting from `level` to `level+1`.

        Thresholds are derived from running statistics, not hardcoded:
          - min_radius: relative multipliers [0.3, 0.6, 1.0] × mean_precision
          - min_access: relative multipliers [0.3, 1.0, 3.0] × mean_access_count

        The multipliers express *relative difficulty* of each tier — they are
        pure ratios, not absolute numbers.
        """
        # Relative multipliers per level transition: 0→1, 1→2, 2→3
        radius_multipliers = (0.3, 0.6, 1.0)
        access_multipliers = (0.3, 1.0, 3.0)
        if level < 0 or level >= len(radius_multipliers):
            return (float("inf"), float("inf"))
        mean_r = self.mean_precision.item()
        mean_a = self.mean_access_count.item()
        return (
            radius_multipliers[level] * mean_r,
            access_multipliers[level] * mean_a,
        )

    @property
    def eviction_recency_weight(self) -> float:
        """Weight on recency vs. precision when scoring beliefs for eviction.

        A value of 0.1 means recency contributes 10 % to the eviction score;
        precision contributes the remainder.  Kept stable at the base value
        for now — exposed as a property so it can be made adaptive later.
        """
        return 0.1

    # ── Repr ─────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"decay={self.decay}, "
            f"mean_surprise={self.mean_surprise.item():.3f}, "
            f"belief_fill_ratio={self.belief_fill_ratio.item():.3f}"
        )
