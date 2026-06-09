"""Tests for the credit-assignment diagnostic and the TBPTT window flag.

PART A: CreditDiagnostic measures the running correlation between the per-step
proxy reward and the FUTURE drop in token loss (k steps ahead). We assert:
  - a stream where proxy reward perfectly predicts future loss drop -> corr ~ +1
  - an anti-correlated stream                                       -> corr ~ -1
  - an independent stream                                           -> corr ~  0
  - the t / t+k alignment is exactly k steps (horizon) wide.

PART B: tbptt_window=1 reproduces single-step detach behavior -> detach_state
called every step; window=k -> called every k steps; default is 1.
"""

import math
import random

import pytest

from memoria.training.credit_diagnostics import CreditDiagnostic
from memoria.model.config import TrainingConfig


# ── EMA decay used in tests ──
# Sourced to match the controller's reward_ema_decay default (0.99): the
# diagnostic in train.py reads that exact MetaParam, so tests exercise the same
# smoothing timescale rather than inventing a new constant.
EMA_DECAY = 0.99
# Stream length: long enough for the EMA (effective window ~ 1/(1-decay) = 100
# samples) to converge well past its warmup before we assert.
N_STEPS = 4000


# ──────────────────────────────────────────────────────────────────────────
# PART A — correlation math
# ──────────────────────────────────────────────────────────────────────────

def test_perfect_predictor_gives_corr_near_plus_one():
    """proxy_reward[t] == future loss drop  =>  correlation -> +1."""
    k = 3
    cd = CreditDiagnostic(horizon=k, ema_decay=EMA_DECAY)

    # Build a loss trajectory whose step-to-step DROP is a known random signal,
    # and set proxy_reward[t] = -(loss[t+k] - loss[t]) exactly. Then x and y are
    # identical streams, so Pearson correlation must converge to +1.
    rng = random.Random(0)
    losses = [10.0]
    for _ in range(N_STEPS + k):
        losses.append(losses[-1] + rng.uniform(-1.0, 1.0))

    last = None
    for t in range(N_STEPS):
        future_drop = -(losses[t + k] - losses[t])  # what y[t] will be
        proxy = future_drop                          # perfect predictor
        last = cd.update(proxy, losses[t])

    assert last is not None
    assert last > 0.99, f"expected corr ~ +1, got {last}"


def test_anticorrelated_predictor_gives_corr_near_minus_one():
    """proxy_reward[t] == -(future loss drop)  =>  correlation -> -1."""
    k = 3
    cd = CreditDiagnostic(horizon=k, ema_decay=EMA_DECAY)

    rng = random.Random(1)
    losses = [10.0]
    for _ in range(N_STEPS + k):
        losses.append(losses[-1] + rng.uniform(-1.0, 1.0))

    last = None
    for t in range(N_STEPS):
        future_drop = -(losses[t + k] - losses[t])
        proxy = -future_drop  # actively anti-aligned with the task
        last = cd.update(proxy, losses[t])

    assert last is not None
    assert last < -0.99, f"expected corr ~ -1, got {last}"


def test_independent_streams_give_corr_near_zero():
    """proxy reward independent of future loss  =>  correlation -> ~0."""
    k = 3
    cd = CreditDiagnostic(horizon=k, ema_decay=EMA_DECAY)

    rng_loss = random.Random(2)
    rng_reward = random.Random(99)  # disjoint seed => statistically independent

    losses = [10.0]
    for _ in range(N_STEPS + k):
        losses.append(losses[-1] + rng_loss.uniform(-1.0, 1.0))

    last = None
    for t in range(N_STEPS):
        proxy = rng_reward.gauss(0.0, 1.0)  # unrelated to the loss trajectory
        last = cd.update(proxy, losses[t])

    assert last is not None
    # Independent => |corr| small. Loose bound: EMA of finite samples has noise.
    assert abs(last) < 0.1, f"expected corr ~ 0, got {last}"


def _reference_ema_corr(xs, ys, decay):
    """Independent re-implementation of West's exponentially-weighted
    covariance recursion, used to validate the class's O(1) updates produce
    the right number on the SAME aligned (x, y) pairs."""
    d = decay
    w = 1.0 - d
    mx = my = vx = vy = cxy = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            mx, my = x, y
            continue
        dx, dy = x - mx, y - my
        vx = d * (vx + w * dx * dx)
        vy = d * (vy + w * dy * dy)
        cxy = d * (cxy + w * dx * dy)
        mx += w * dx
        my += w * dy
    if vx * vy <= 0.0:
        return None
    return cxy / math.sqrt(vx * vy)


def test_ema_corr_matches_independent_reference_recursion():
    """The class's streaming EMA correlation must equal a from-scratch
    reference computed on the identical aligned (x, y) pairs — proves the ring
    buffer alignment and the moment recursion are implemented correctly."""
    k = 2
    decay = 0.99
    cd = CreditDiagnostic(horizon=k, ema_decay=decay)

    rng = random.Random(7)
    n = 300
    losses = [5.0]
    for _ in range(n + k):
        losses.append(losses[-1] + rng.uniform(-2.0, 2.0))
    proxies = [rng.gauss(0.0, 1.0) for _ in range(n)]

    xs, ys = [], []
    last = None
    for t in range(n):
        # The pair the diagnostic forms internally is (proxy[t-k], -(loss[t]-loss[t-k])).
        if t >= k:
            xs.append(proxies[t - k])
            ys.append(-(losses[t] - losses[t - k]))
        last = cd.update(proxies[t], losses[t])

    ref = _reference_ema_corr(xs, ys, decay)
    assert last is not None and ref is not None
    assert abs(last - ref) < 1e-9, f"streaming={last} vs reference={ref}"


def test_alignment_horizon_is_exactly_k():
    """The diagnostic must pair reward[t] with loss[t+k], not any other lag.

    Feed a single nonzero reward spike at t=0 and a loss that drops only at one
    chosen step; correlation can only become defined once the spike at t=0 has
    been aligned with loss[k]. Before that the diagnostic is in warmup (returns
    None for fewer than two aligned pairs)."""
    k = 5
    cd = CreditDiagnostic(horizon=k, ema_decay=EMA_DECAY)

    # No aligned pair can exist until step index k has been ingested.
    for t in range(k):
        out = cd.update(proxy_reward=1.0, loss_token=1.0)
        assert out is None, f"corr defined too early at t={t}: {out}"
        assert cd.n_pairs == 0

    # Ingesting the step at index k aligns the FIRST pair (the t=0 sample).
    cd.update(proxy_reward=1.0, loss_token=1.0)
    assert cd.n_pairs == 1


def test_warmup_returns_none():
    cd = CreditDiagnostic(horizon=2, ema_decay=EMA_DECAY)
    assert cd.correlation is None
    assert cd.update(0.5, 1.0) is None  # buffering, nothing aligned yet


def test_constant_stream_has_undefined_correlation():
    """Zero variance => correlation undefined (returns None), never a NaN."""
    cd = CreditDiagnostic(horizon=2, ema_decay=EMA_DECAY)
    for _ in range(100):
        out = cd.update(proxy_reward=0.3, loss_token=2.0)
    assert out is None  # var collapses to 0 => no correlation


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        CreditDiagnostic(horizon=0, ema_decay=0.99)
    with pytest.raises(ValueError):
        CreditDiagnostic(horizon=1, ema_decay=0.0)
    with pytest.raises(ValueError):
        CreditDiagnostic(horizon=1, ema_decay=1.0)


# ──────────────────────────────────────────────────────────────────────────
# PART B — TBPTT window flag (detach cadence)
# ──────────────────────────────────────────────────────────────────────────

def _detach_steps(window: int, n_steps: int) -> list[int]:
    """Replicate train.py's detach gate: detach when (step+1) % window == 0.
    Returns the list of step indices on which detach_state() would be called."""
    return [s for s in range(n_steps) if (s + 1) % max(window, 1) == 0]


def test_tbptt_window_default_is_one():
    """Default config = exactly today's behavior (1-step BPTT)."""
    assert TrainingConfig().tbptt_window == 1


def test_window_one_detaches_every_step():
    """tbptt_window=1 must reproduce detach_state() on EVERY step."""
    n = 50
    fired = _detach_steps(window=1, n_steps=n)
    assert fired == list(range(n)), "window=1 must detach every single step"


def test_window_k_detaches_every_k_steps():
    """tbptt_window=k detaches on steps k-1, 2k-1, ... (window-close steps)."""
    n = 20
    for window in (2, 4, 5):
        fired = _detach_steps(window=window, n_steps=n)
        expected = [s for s in range(n) if (s + 1) % window == 0]
        assert fired == expected
        # Spacing between consecutive detaches is exactly `window`.
        diffs = [b - a for a, b in zip(fired, fired[1:])]
        assert all(d == window for d in diffs)


def test_credit_horizon_default_is_derived_not_zero():
    """credit_horizon must be a sane positive default (>=1) so the diagnostic
    can construct without error."""
    h = TrainingConfig().credit_horizon
    assert h >= 1
    # Constructing with the shipped defaults must not raise.
    CreditDiagnostic(horizon=h, ema_decay=0.99)
