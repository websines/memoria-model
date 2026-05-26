"""Tests for opt-in self-improvement persistence policy."""

import torch
import torch.nn as nn

from memoria.agency.persistence import (
    SelfImprovementPolicy,
    maybe_persist_self_improvement,
)
from memoria.core.state import CognitiveState, StateConfig


class TinyAdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = CognitiveState(
            StateConfig(belief_dim=16, max_beliefs=16, max_edges=32, max_goals=4),
        )
        self.proj = nn.Linear(2, 2)


def test_policy_disabled_writes_nothing(tmp_path):
    model = TinyAdaptiveModel()
    policy = SelfImprovementPolicy(enabled=False, checkpoint_dir=tmp_path)

    path = maybe_persist_self_improvement(
        model,
        step=10,
        policy=policy,
        force=True,
    )

    assert path is None
    assert list(tmp_path.glob("*.pt")) == []


def test_cognitive_only_checkpoint(tmp_path):
    model = TinyAdaptiveModel()
    model.state.allocate_belief(torch.randn(16))
    policy = SelfImprovementPolicy(
        enabled=True,
        checkpoint_dir=tmp_path,
        interval_steps=5,
        cognitive_only=True,
    )

    path = maybe_persist_self_improvement(model, step=10, policy=policy)

    assert path is not None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved['cognitive_only'] is True
    assert 'cognitive_state' in saved
    assert 'model_state' not in saved


def test_full_online_checkpoint_includes_optimizer(tmp_path):
    model = TinyAdaptiveModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    policy = SelfImprovementPolicy(
        enabled=True,
        checkpoint_dir=tmp_path,
        interval_steps=100,
        cognitive_only=False,
    )

    path = maybe_persist_self_improvement(
        model,
        step=7,
        policy=policy,
        optimizer=optimizer,
        force=True,
        tag="forced",
    )

    assert path is not None
    assert path.name == "forced.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved['cognitive_only'] is False
    assert 'model_state' in saved
    assert 'optimizer_state' in saved


def test_policy_keeps_last_checkpoints(tmp_path):
    model = TinyAdaptiveModel()
    policy = SelfImprovementPolicy(
        enabled=True,
        checkpoint_dir=tmp_path,
        interval_steps=1,
        keep_last=2,
    )

    for step in range(1, 5):
        maybe_persist_self_improvement(model, step=step, policy=policy)

    checkpoints = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert len(checkpoints) == 2
    assert checkpoints == ["step_3.pt", "step_4.pt"]
