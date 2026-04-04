"""Tests for D1: Daemon Loop."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.agency.daemon import (
    DaemonLoop,
    Event,
    EventType,
    ActionType,
    DaemonState,
)


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=256, max_goals=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def daemon():
    return DaemonLoop(belief_dim=64)


def test_daemon_init(daemon):
    """Daemon initializes with clean state."""
    assert daemon.daemon_state.step == 0
    assert daemon.daemon_state.idle_steps == 0
    assert daemon.daemon_state.total_actions == 0


def test_perceive_empty(daemon):
    """Empty event list returns idle event."""
    events = daemon.perceive([])
    assert len(events) == 1
    assert events[0].event_type == EventType.IDLE


def test_perceive_sorts_by_priority(daemon):
    """Events are sorted by priority (highest first)."""
    events = [
        Event(EventType.USER_MESSAGE, priority=1.0),
        Event(EventType.CURIOSITY, priority=5.0),
        Event(EventType.TIMER, priority=0.5),
    ]
    sorted_events = daemon.perceive(events)
    priorities = [e.priority for e in sorted_events]
    assert priorities == sorted(priorities, reverse=True)


def test_process_idle_event(daemon, state):
    """Processing idle event increments idle counter."""
    event = Event(EventType.IDLE)
    result = daemon.process_event(event, state)
    assert result['idle_steps'] == 1
    assert result['event_type'] == EventType.IDLE


def test_process_user_message(daemon, state):
    """User message resets idle counter and recommends respond."""
    # First idle
    daemon.process_event(Event(EventType.IDLE), state)
    # Then message
    result = daemon.process_event(Event(EventType.USER_MESSAGE), state)
    assert result['idle_steps'] == 0
    assert result['recommended_action'] == ActionType.RESPOND


def test_process_curiosity_event(daemon, state):
    """Curiosity event recommends explore."""
    result = daemon.process_event(Event(EventType.CURIOSITY), state)
    assert result['recommended_action'] == ActionType.EXPLORE


def test_process_anomaly_event(daemon, state):
    """Anomaly event recommends search."""
    result = daemon.process_event(Event(EventType.ANOMALY), state)
    assert result['recommended_action'] == ActionType.SEARCH


def test_should_consolidate_after_idle(daemon, state):
    """Consolidation triggers after idle threshold."""
    for _ in range(daemon.idle_consolidation_interval + 1):
        daemon.daemon_state.idle_steps += 1
    assert daemon.should_consolidate(state)


def test_should_consolidate_high_beta(daemon, state):
    """Consolidation triggers when beta is high and no goals."""
    state.meta.data[0] = 0.9  # high beta
    assert daemon.should_consolidate(state)


def test_should_consolidate_near_capacity(daemon, state):
    """Consolidation triggers when belief store is near full."""
    for i in range(int(state.config.max_beliefs * 0.95)):
        state.allocate_belief(torch.randn(64) * 0.3)
    assert daemon.should_consolidate(state)


def test_detect_anomaly(daemon, state):
    """Anomaly detection returns bool."""
    error = torch.randn(64)
    result = daemon.detect_anomaly(error, state)
    assert isinstance(result, bool)


def test_record_action(daemon):
    """Action recording works."""
    from memoria.agency.daemon import Action
    action = Action(ActionType.RESPOND, confidence=0.9, efe_score=-0.5)
    daemon.record_action(action)
    assert len(daemon.daemon_state.action_history) == 1


def test_action_history_bounded(daemon):
    """Action history doesn't grow unbounded."""
    from memoria.agency.daemon import Action
    for _ in range(1500):
        daemon.record_action(Action(ActionType.WAIT))
    assert len(daemon.daemon_state.action_history) <= 1000


def test_daemon_summary(daemon):
    """Summary is readable."""
    s = daemon.summary()
    assert 'DaemonLoop' in s
    assert 'step=0' in s


def test_daemon_integrated(state):
    """Daemon is attached to state."""
    assert hasattr(state, 'daemon')
    assert isinstance(state.daemon, DaemonLoop)


def test_step_counter_increments(daemon, state):
    """Each process_event call increments the step counter."""
    daemon.process_event(Event(EventType.IDLE), state)
    daemon.process_event(Event(EventType.USER_MESSAGE), state)
    assert daemon.daemon_state.step == 2
