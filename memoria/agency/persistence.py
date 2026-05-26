"""Persistence policy for opt-in self-improvement state.

Checkpointing is not itself learning. This module only persists adaptive
changes that already happened through state updates or optimizer steps. The
policy is explicit so read-only inference stays read-only by default, while an
agent runner can opt into cognitive-only or full online-training checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class SelfImprovementPolicy:
    """Controls when adaptive state is persisted during deployment."""

    enabled: bool = False
    checkpoint_dir: str | Path = "checkpoints/self_improvement"
    interval_steps: int = 100
    min_dirty_updates: int = 1
    cognitive_only: bool = True
    keep_last: int = 5

    def should_persist(
        self,
        step: int,
        dirty_updates: int = 1,
        force: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        if dirty_updates < self.min_dirty_updates:
            return False
        if force:
            return True
        interval = max(int(self.interval_steps), 1)
        return step > 0 and step % interval == 0


def maybe_persist_self_improvement(
    model,
    step: int,
    policy: SelfImprovementPolicy,
    optimizer: Any | None = None,
    dirty_updates: int = 1,
    force: bool = False,
    tag: str | None = None,
) -> Path | None:
    """Persist adaptive state/weights if the policy allows it.

    Args:
        model: object with `.state.state_dict_cognitive()` and optionally
            `.state_dict()` when `policy.cognitive_only` is False.
        step: current deployment/training step.
        policy: persistence policy.
        optimizer: optional optimizer to persist for online training resumes.
        dirty_updates: number of state/skill/weight updates since last save.
        force: persist now if policy is enabled, ignoring the step interval.
        tag: optional filename stem.

    Returns:
        Path to the checkpoint, or None when no checkpoint was written.
    """
    if not policy.should_persist(step, dirty_updates=dirty_updates, force=force):
        return None

    checkpoint_dir = Path(policy.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = tag if tag is not None else f"step_{step}"
    path = checkpoint_dir / f"{stem}.pt"

    payload: dict[str, Any] = {
        'step': step,
        'cognitive_state': model.state.state_dict_cognitive(),
        'cognitive_only': policy.cognitive_only,
        'dirty_updates': dirty_updates,
    }
    if not policy.cognitive_only:
        payload['model_state'] = model.state_dict()
        if optimizer is not None:
            payload['optimizer_state'] = optimizer.state_dict()

    torch.save(payload, path)
    _prune_old_checkpoints(checkpoint_dir, keep_last=policy.keep_last)
    return path


def _prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in checkpoints[keep_last:]:
        old.unlink(missing_ok=True)
