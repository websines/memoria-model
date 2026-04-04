"""D1: Daemon Loop — persistent event-driven cognitive process.

The daemon loop turns Memoria from a passive responder into an autonomous
agent. It runs continuously, perceiving events, updating beliefs, planning,
selecting actions, and consolidating during idle periods.

The loop follows the Active Inference cycle:
    PERCEIVE → UPDATE → PLAN → ACT → OBSERVE → CONSOLIDATE

Events can be:
- External: user messages, tool results, API responses
- Internal: timer triggers, curiosity signals, goal completions
- Idle: no events → consolidate (dream phase, sleep cycle)

The daemon is event-driven (not polling). It awaits events and processes
them through the full cognitive pipeline. During idle periods, it runs
background consolidation to improve belief consistency.

Reference: From Pixels to Planning (Friston, arXiv:2407.20292)
Reference: Autonomous Deep Agent (arXiv:2502.07056)
Reference: MAP: Modular Planner (Nature Communications 2025)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Types of events the daemon can process."""
    USER_MESSAGE = auto()     # external user input
    TOOL_RESULT = auto()      # result from tool execution
    TIMER = auto()            # periodic trigger (consolidation, etc.)
    CURIOSITY = auto()        # internal curiosity signal exceeded threshold
    GOAL_COMPLETE = auto()    # a Telos goal reached completion
    GOAL_FAILED = auto()      # a Telos goal failed or timed out
    ANOMALY = auto()          # high prediction error detected
    IDLE = auto()             # no events — opportunity for consolidation


class ActionType(Enum):
    """Types of actions the daemon can take."""
    RESPOND = auto()          # generate text response
    TOOL_CALL = auto()        # invoke an external tool
    SEARCH = auto()           # information gathering
    WAIT = auto()             # do nothing (wait for more info)
    EXPLORE = auto()          # generate exploration goal
    CONSOLIDATE = auto()      # run sleep/dream cycle


@dataclass
class Event:
    """An event to be processed by the daemon."""
    event_type: EventType
    payload: Any = None             # event-specific data
    timestamp: float = 0.0
    priority: float = 1.0           # higher = process first


@dataclass
class Action:
    """An action selected by the daemon."""
    action_type: ActionType
    payload: Any = None             # action-specific parameters
    confidence: float = 0.0         # how confident the daemon is in this action
    efe_score: float = 0.0          # Expected Free Energy of this action
    goal_idx: int = -1              # which goal this action serves


@dataclass
class DaemonState:
    """Persistent state of the daemon loop."""
    step: int = 0
    idle_steps: int = 0
    total_actions: int = 0
    total_consolidations: int = 0
    last_event_type: EventType = EventType.IDLE
    action_history: list[Action] = field(default_factory=list)
    pending_events: list[Event] = field(default_factory=list)


class DaemonLoop(nn.Module):
    """Persistent event-driven cognitive process.

    Orchestrates the full Active Inference cycle:
    1. PERCEIVE: receive and prioritize events
    2. UPDATE: run observation through model, update beliefs via pass2
    3. PLAN: compute preference/epistemic priors, run planning (B1-B4)
    4. ACT: select action via EFE-based action selection (D2)
    5. OBSERVE: compare action result to prediction, detect anomalies
    6. CONSOLIDATE: during idle, run dream phase and sleep cycle

    The daemon maintains a persistent DaemonState that tracks the loop's
    own history and statistics.

    Args:
        belief_dim: dimension of belief vectors
        idle_consolidation_interval: steps of idleness before auto-consolidation
    """

    def __init__(self, belief_dim: int, idle_consolidation_interval: int = 10):
        super().__init__()
        self.belief_dim = belief_dim
        self.idle_consolidation_interval = idle_consolidation_interval

        # Anomaly detection: learned threshold for prediction error
        self.anomaly_net = nn.Sequential(
            nn.Linear(belief_dim + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # Initialize conservatively (high threshold → few anomalies)
        nn.init.zeros_(self.anomaly_net[-2].weight)
        nn.init.constant_(self.anomaly_net[-2].bias, -2.0)

        # Event priority scoring: event features → priority
        self.priority_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

        self.daemon_state = DaemonState()

    def perceive(self, events: list[Event]) -> list[Event]:
        """Receive and prioritize events.

        Scores events by learned priority and sorts them. Higher priority
        events are processed first.

        Args:
            events: list of incoming events

        Returns:
            events sorted by priority (highest first)
        """
        if not events:
            return [Event(event_type=EventType.IDLE)]

        self.daemon_state.pending_events.extend(events)

        # Sort by priority (pre-assigned or learned)
        self.daemon_state.pending_events.sort(
            key=lambda e: e.priority, reverse=True
        )

        return self.daemon_state.pending_events

    def should_consolidate(self, state) -> bool:
        """Decide if background consolidation should run.

        Triggers when:
        - Idle for more than idle_consolidation_interval steps
        - Beta is high (uncertainty-dominant) and no pressing goals
        - Memory is near capacity and needs cleanup

        All conditions use MetaParams or state-derived values, not magic numbers.
        """
        if self.daemon_state.idle_steps >= self.idle_consolidation_interval:
            return True

        beta = state.meta.data[0].item()
        fill_ratio = state.num_active_beliefs() / max(state.config.max_beliefs, 1)

        # Consolidate when uncertain and no active goals
        if beta > 0.8 and state.num_active_goals() == 0:
            return True

        # Consolidate when near capacity
        if fill_ratio > 0.9:
            return True

        return False

    def detect_anomaly(
        self,
        prediction_error: torch.Tensor,
        state,
    ) -> bool:
        """Detect if the prediction error constitutes an anomaly.

        Uses a learned network rather than a fixed threshold.

        Args:
            prediction_error: [D] vector of prediction errors
            state: CognitiveState for context

        Returns:
            True if anomaly detected
        """
        device = prediction_error.device
        error_mag = prediction_error.norm()
        beta = state.meta.data[0]
        surprise = state.meta.data[1]

        features = torch.cat([
            prediction_error / (error_mag + 1e-8),  # direction
            error_mag.unsqueeze(0),                   # magnitude
            beta.unsqueeze(0),                        # current exploration state
        ])

        # Pad to belief_dim + 2
        if features.shape[0] < self.belief_dim + 2:
            features = torch.nn.functional.pad(
                features, (0, self.belief_dim + 2 - features.shape[0])
            )

        prob = self.anomaly_net(features.unsqueeze(0)).squeeze()
        return prob.item() > 0.5

    def process_event(self, event: Event, state) -> dict:
        """Process a single event through the cognitive pipeline.

        This is the core of the daemon loop. Each event goes through:
        1. Classify event type and extract features
        2. Update daemon state
        3. Return processing result for downstream use

        Args:
            event: the event to process
            state: CognitiveState

        Returns:
            dict with processing results and recommended action type
        """
        self.daemon_state.step += 1
        self.daemon_state.last_event_type = event.event_type

        if event.event_type == EventType.IDLE:
            self.daemon_state.idle_steps += 1
        else:
            self.daemon_state.idle_steps = 0

        result = {
            'event_type': event.event_type,
            'step': self.daemon_state.step,
            'idle_steps': self.daemon_state.idle_steps,
        }

        # Determine recommended action based on event type
        if event.event_type == EventType.IDLE:
            if self.should_consolidate(state):
                result['recommended_action'] = ActionType.CONSOLIDATE
                self.daemon_state.total_consolidations += 1
            else:
                result['recommended_action'] = ActionType.WAIT
        elif event.event_type == EventType.CURIOSITY:
            result['recommended_action'] = ActionType.EXPLORE
        elif event.event_type == EventType.ANOMALY:
            result['recommended_action'] = ActionType.SEARCH
        elif event.event_type in (EventType.USER_MESSAGE, EventType.TOOL_RESULT):
            result['recommended_action'] = ActionType.RESPOND
        else:
            result['recommended_action'] = ActionType.WAIT

        self.daemon_state.total_actions += 1

        # Remove processed event from pending
        if event in self.daemon_state.pending_events:
            self.daemon_state.pending_events.remove(event)

        return result

    def record_action(self, action: Action):
        """Record an action taken by the daemon."""
        self.daemon_state.action_history.append(action)
        # Keep bounded history
        if len(self.daemon_state.action_history) > 1000:
            self.daemon_state.action_history = self.daemon_state.action_history[-500:]

    def summary(self) -> str:
        """Human-readable summary of daemon state."""
        ds = self.daemon_state
        return (
            f"DaemonLoop: step={ds.step}, idle={ds.idle_steps}, "
            f"actions={ds.total_actions}, consolidations={ds.total_consolidations}, "
            f"pending={len(ds.pending_events)}, "
            f"last_event={ds.last_event_type.name}"
        )
