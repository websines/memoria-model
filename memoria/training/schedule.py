"""Learning rate and alpha scheduling.

LR schedule: warmup → constant → warmdown (linear or cosine) — from autoresearch.
Alpha schedule: KL annealing for L_fe — ramps from 0 to alpha_max during phase 2.

Reference: β-VAE literature (KL annealing)
Reference: autoresearch/train.py (LR schedule)
"""

import math
from ..model.config import TrainingConfig


def get_lr_multiplier(step: int, total_steps: int, config: TrainingConfig) -> float:
    """Compute LR multiplier at current step.

    Three-phase schedule:
    1. Warmup: 0 → 1 (linear)
    2. Constant: 1
    3. Warmdown: 1 → final_lr_frac (linear or cosine)

    Args:
        step: current training step
        total_steps: estimated total training steps
        config: training config

    Returns:
        multiplier in [final_lr_frac, 1.0]
    """
    progress = min(step / max(total_steps, 1), 1.0)

    if progress < config.warmup_ratio:
        # Linear warmup
        return progress / config.warmup_ratio if config.warmup_ratio > 0 else 1.0
    elif config.warmdown_ratio <= 0 or progress < 1.0 - config.warmdown_ratio:
        # Constant (also handles warmdown_ratio=0 to avoid division by zero)
        return 1.0
    else:
        # Warmdown: cooldown_progress goes from 1.0 → 0.0
        cooldown_progress = (1.0 - progress) / config.warmdown_ratio
        if getattr(config, 'warmdown_type', None) == 'linear':
            # Linear decay: 1.0 → 0.0
            return cooldown_progress
        # Cosine warmdown: smooth decay from 1.0 → final_lr_frac
        return config.final_lr_frac + 0.5 * (1.0 - config.final_lr_frac) * (
            1.0 + math.cos(math.pi * (1.0 - cooldown_progress))
        )


def get_alpha(step: int, config: TrainingConfig) -> float:
    """Compute α (L_fe weight) at current step.

    KL annealing schedule:
    - Phase 1 (step < phase1_steps): α = 0 (L_token only)
    - Phase 2 (phase1_steps ≤ step < phase1_steps + alpha_warmup_steps):
      α ramps linearly from 0 to alpha_max
    - Phase 3 (step ≥ phase1_steps + alpha_warmup_steps): α = alpha_max

    Args:
        step: current training step
        config: training config

    Returns:
        α ∈ [0, alpha_max]
    """
    if step < config.phase1_steps:
        return 0.0

    ramp_start = config.phase1_steps
    ramp_end = ramp_start + config.alpha_warmup_steps

    if step >= ramp_end:
        return config.alpha_max

    # Linear ramp
    progress = (step - ramp_start) / max(config.alpha_warmup_steps, 1)
    return config.alpha_max * progress
