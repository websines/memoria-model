"""Credit-assignment diagnostic for the single-step-detach training loop.

The training loop calls ``base_model.detach_state()`` every step (train.py,
right after ``optimizer.step()`` / before ``run_pass2``). This truncates BPTT
to a single step, so no gradient ever links a memory write/store decision at
step ``t`` to the task loss at step ``t + k``. Every long-horizon learner
(controller PPO, structural-plasticity REINFORCE, SRWM, action selector,
curiosity) is therefore trained on a *single-step proxy* reward — a variant of
"reduce my own surprise / free-energy right now" — rather than on downstream
token-prediction utility.

That substitution is a bet: it only pays off if reducing the internal proxy
reward at step ``t`` actually correlates with the token loss dropping at step
``t + k``. The code has never measured that correlation. This module measures
it, O(1) per step, with zero behavioral effect on training.

Math
----
We track the running (EMA-weighted) Pearson correlation between

    x[t] = proxy_reward[t]
    y[t] = -(loss_token[t + k] - loss_token[t])          (the credit target)

The negation makes ``y`` *good when future loss drops*: a positive ``y`` means
the token loss ``k`` steps later is lower than it is now. A correlation near
+1 means the proxy reward is a faithful leading indicator of future loss
improvement (the bet holds); near 0 means the proxy is uninformative about
downstream utility; near -1 means the proxy is actively *anti*-aligned with the
task (optimizing it hurts future tokens).

Because ``y[t]`` needs ``loss_token[t + k]``, we hold the most recent ``k``
samples of ``(proxy_reward, loss_token_at_t)`` in a length-``k`` ring buffer.
When ``loss_token[t + k]`` arrives we pair it with the buffered values from
step ``t`` and feed one ``(x, y)`` pair into the EMA-correlation accumulator.
No buffer grows beyond ``k`` entries.

EMA correlation (West, 1979 — exponentially-weighted Welford). For decay
``d`` in ``(0, 1)`` and weight ``w = 1 - d`` on the newest sample:

    mean_x  <- d * mean_x + w * x
    mean_y  <- d * mean_y + w * y
    var_x   <- d * (var_x  + w * (x - mean_x_old)^2)        # uses pre-update mean
    var_y   <- d * (var_y  + w * (y - mean_y_old)^2)
    cov_xy  <- d * (cov_xy + w * (x - mean_x_old)*(y - mean_y_old))
    corr    =  cov_xy / sqrt(var_x * var_y)

This is the standard exponentially-weighted covariance recursion: each update
is O(1) in time and memory, and the ``var/cov`` updates use the *pre-update*
means so the cross-term bookkeeping is exact (the same form torch's
running-stats and Welford-style estimators use elsewhere in the codebase).
"""

from __future__ import annotations

from collections import deque
import math


class CreditDiagnostic:
    """Online correlation between a per-step proxy reward and future loss drop.

    Args:
        horizon: number of optimizer steps ``k`` between the proxy reward and
            the future token loss it is supposed to predict. Must be >= 1.
        ema_decay: EMA decay ``d`` in ``(0, 1)`` for the running covariance /
            variances. Higher = longer memory. This is NOT a free literal: the
            caller is expected to source it from an existing decay constant
            (e.g. the controller's ``reward_ema_decay`` MetaParam), so the
            diagnostic's smoothing matches the reward EMA the controller itself
            learns against.
    """

    def __init__(self, horizon: int, ema_decay: float):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")

        self.horizon = horizon
        self.ema_decay = ema_decay

        # Ring buffer of the last k pending (proxy_reward, loss_token_at_t)
        # samples awaiting their loss_token[t + k] partner. deque(maxlen=k)
        # auto-evicts, but we pop from the left explicitly once aligned so the
        # buffer holds exactly the k in-flight samples and never grows.
        self._pending: deque[tuple[float, float]] = deque(maxlen=horizon)

        # EMA-weighted moments of the aligned (x, y) stream.
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._var_x = 0.0
        self._var_y = 0.0
        self._cov_xy = 0.0
        # Number of aligned (x, y) pairs ingested. Used to (a) report warmup and
        # (b) skip the first pair's variance update, where the EMA mean is still
        # seeded at the first observation and the deviation is trivially zero.
        self._n_pairs = 0

    def update(self, proxy_reward: float, loss_token: float) -> float | None:
        """Ingest one step's ``(proxy_reward, loss_token)``.

        Returns the current EMA correlation once at least one aligned pair and
        enough variance exists to define it, else ``None`` (warmup / degenerate
        variance). Pure observation — never mutates any training tensor.
        """
        # Align: if a sample from k steps ago is waiting, its target is
        # y = -(loss_token_now - loss_token_then).
        if len(self._pending) == self.horizon:
            x_then, loss_then = self._pending.popleft()
            y = -(loss_token - loss_then)
            self._ingest_pair(float(x_then), float(y))

        # Buffer the current step for its future partner k steps from now.
        self._pending.append((float(proxy_reward), float(loss_token)))

        return self.correlation

    def _ingest_pair(self, x: float, y: float) -> None:
        """Fold one aligned ``(x, y)`` into the EMA moments (West 1979)."""
        d = self.ema_decay
        w = 1.0 - d

        if self._n_pairs == 0:
            # Seed the means on the first pair; deviations are zero so the
            # var/cov updates would be no-ops anyway. This avoids biasing the
            # estimator toward an arbitrary 0.0 initial mean.
            self._mean_x = x
            self._mean_y = y
            self._n_pairs = 1
            return

        dx = x - self._mean_x
        dy = y - self._mean_y

        # var/cov use the PRE-update means (West's exponentially-weighted form).
        self._var_x = d * (self._var_x + w * dx * dx)
        self._var_y = d * (self._var_y + w * dy * dy)
        self._cov_xy = d * (self._cov_xy + w * dx * dy)

        self._mean_x += w * dx
        self._mean_y += w * dy
        self._n_pairs += 1

    @property
    def correlation(self) -> float | None:
        """Current EMA Pearson correlation, or ``None`` if undefined.

        Undefined while in warmup (fewer than two aligned pairs) or when either
        variance has collapsed to ~0 (a constant stream has no correlation).
        """
        # Need at least two pairs for a variance update to have fired.
        if self._n_pairs < 2:
            return None
        denom = self._var_x * self._var_y
        if denom <= 0.0:
            return None
        corr = self._cov_xy / math.sqrt(denom)
        # Clamp tiny floating-point overshoot past the mathematical [-1, 1].
        return max(-1.0, min(1.0, corr))

    @property
    def n_pairs(self) -> int:
        """Number of aligned (x, y) pairs ingested so far (warmup indicator)."""
        return self._n_pairs
