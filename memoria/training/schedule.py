"""Learning rate, alpha, and context length scheduling.

LR schedule: warmup → constant → warmdown (linear or cosine) — from autoresearch.
Alpha schedule: KL annealing for L_fe — ramps from 0 to alpha_max during phase 2.
Context schedule: SkyLadder progressive context extension — short→long ramp.

Reference: β-VAE literature (KL annealing)
Reference: autoresearch/train.py (LR schedule)
Reference: SkyLadder (NeurIPS 2025, arxiv.org/abs/2503.15450) — context window scheduling
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
            # Linear decay: 1.0 → final_lr_frac (NOT 0.0 — respects the floor,
            # matching the cosine branch below). cooldown_progress goes 1.0 → 0.0,
            # so we interpolate between 1.0 and final_lr_frac.
            return config.final_lr_frac + (1.0 - config.final_lr_frac) * cooldown_progress
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


def get_context_length(step: int, total_steps: int, config: TrainingConfig,
                       sequence_len: int = 2048) -> int:
    """Compute context length at current step via SkyLadder schedule.

    Progressive context extension: ramps from skyladder_start to sequence_len
    over the first skyladder_ratio of training, then holds at sequence_len.

    Short context early = faster training (less attention compute) + better
    representations (model learns local patterns first, then long-range).

    Schedules:
    - "linear": ctx = start + (target - start) × progress
    - "exponential": ctx = start × (target/start)^progress (log-linear ramp)
    - "step": doubles context at equal intervals (2K→4K→8K→...)

    Args:
        step: current training step
        total_steps: estimated total training steps
        config: training config
        sequence_len: target sequence length (from TransformerConfig.sequence_len)

    Returns:
        context length, quantized to POWER-OF-TWO boundaries within [start, target].

    Quantization is to powers of two, not arbitrary multiples of 64, because
    every shape change invalidates torch.compile's trace cache. Over a 60%
    ramp from 512 → 131072 with a 64-boundary that triggered ~2000 recompiles
    per run — each one a multi-second hit. Power-of-two quantization caps
    the number of distinct shapes at log2(target/start) ≈ 8, which matches
    the recompile budget of a reasonable pretraining run.

    The trade-off: steps between shape boundaries train at a slightly larger
    context than the linear/exponential schedule would produce. At the extreme,
    a step that should be at ctx=700 runs at ctx=1024 — a 46% overshoot in the
    worst case, averaging ~30%. This extra compute is dwarfed by the recompile
    cost it avoids and preserves SkyLadder's "shorter first" principle.
    """
    target = sequence_len
    ratio = getattr(config, 'skyladder_ratio', 0.0)
    start = getattr(config, 'skyladder_start', 256)
    schedule = getattr(config, 'skyladder_schedule', 'linear')

    if ratio <= 0 or start >= target:
        return target

    ramp_steps = int(total_steps * ratio)
    if step >= ramp_steps:
        return target

    progress = step / max(ramp_steps, 1)

    if schedule == 'exponential':
        # Log-linear: spends more steps at shorter context (cheaper)
        log_start = math.log2(max(start, 1))
        log_target = math.log2(target)
        ctx = 2 ** (log_start + (log_target - log_start) * progress)
    elif schedule == 'step':
        # Doubling: 256→512→1024→...→target at equal intervals
        log_start = math.log2(max(start, 1))
        log_target = math.log2(target)
        n_doublings = log_target - log_start
        current_doublings = progress * n_doublings
        ctx = 2 ** (log_start + int(current_doublings))
    else:
        # Linear (default, SkyLadder recommended)
        ctx = start + (target - start) * progress

    # Quantize to the next-higher power of two (unless already a power of two).
    # Clamp to [start, target], with start itself rounded up to a power of two
    # for a consistent boundary lattice.
    ctx = max(start, min(int(ctx), target))
    # Round up: ceil(log2(ctx)) → 2 ** that exponent. math.ceil of a log is
    # stable here because ctx is a positive integer.
    ctx_po2 = 1 << (ctx - 1).bit_length() if ctx > 0 else 1
    ctx_po2 = max(start, min(ctx_po2, target))
    return ctx_po2
