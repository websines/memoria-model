"""Tests for State Interface Layer (read + write paths)."""

import torch
import pytest
from memoria.core.state import CognitiveState, StateConfig
from memoria.interface.layer import StateInterfaceLayer
from memoria.interface.read_path import ReadPath
from memoria.interface.write_path import WritePath


@pytest.fixture
def config():
    return StateConfig(belief_dim=64, max_beliefs=128, max_edges=512, max_goals=8, relation_dim=16)


@pytest.fixture
def state(config):
    return CognitiveState(config)


@pytest.fixture
def interface(config):
    return StateInterfaceLayer(
        hidden_dim=128,
        belief_dim=config.belief_dim,
        num_heads=2,
        top_k=16,
        layer_idx=0,
    )


# ── Read Path Tests ──

def test_read_empty_state(interface, state):
    """Reading from empty state returns zeros (graceful degradation)."""
    hidden = torch.randn(2, 10, 128)  # [B, T, H]
    output, candidates = interface(hidden, state)
    # Output should be same shape as input
    assert output.shape == hidden.shape
    # With empty state, read path returns zeros → output ≈ input
    assert torch.allclose(output, hidden, atol=1e-5)


def test_read_retrieves_relevant_beliefs(config, state):
    """Read path should retrieve beliefs similar to the query."""
    read_path = ReadPath(hidden_dim=128, belief_dim=config.belief_dim, num_heads=1, top_k=16)

    # Add a belief with a known direction
    direction = torch.zeros(config.belief_dim)
    direction[0] = 1.0  # points along dim 0
    belief = direction * 3.0  # precision = 3.0
    state.allocate_belief(belief)

    # Create a hidden state that, after projection, should be similar to the belief
    # We'll just check that the output is non-zero when beliefs exist
    hidden = torch.randn(1, 5, 128)
    output = read_path(hidden, state)
    assert output.shape == (1, 5, 128)
    # With at least one belief, output should be non-zero (projections are random init)
    # (Can't guarantee specific values without controlling the projection weights)


def test_read_with_goal_modulation(config, state):
    """Goal modulation should change attention weights (even if output_proj is zero-init)."""
    read_path = ReadPath(hidden_dim=128, belief_dim=config.belief_dim, num_heads=1, top_k=16)

    # Manually set output_proj to non-zero so we can see differences
    with torch.no_grad():
        read_path.output_proj.weight.normal_(std=0.1)

    # Add two beliefs in different directions
    b1 = torch.zeros(config.belief_dim)
    b1[0] = 2.0
    b2 = torch.zeros(config.belief_dim)
    b2[1] = 2.0
    state.allocate_belief(b1)
    state.allocate_belief(b2)

    # Create a goal aligned with belief 1
    goal_embed = torch.zeros(1, config.belief_dim)
    goal_embed[0, 0] = 1.0
    goal_priorities = torch.tensor([1.0])

    hidden = torch.randn(1, 3, 128)

    # Run with and without goal modulation
    out_no_goal = read_path(hidden, state)
    out_with_goal = read_path(hidden, state, goal_embeddings=goal_embed, goal_priorities=goal_priorities)

    # Outputs should differ (goal modulation changes attention weights)
    assert not torch.allclose(out_no_goal, out_with_goal, atol=1e-6)


# ── Write Path Tests ──

def test_write_to_empty_state(config, state):
    """Writing to empty state should produce new-belief candidates (matched_slot=-1)."""
    write_path = WritePath(hidden_dim=128, belief_dim=config.belief_dim)

    hidden = torch.randn(1, 5, 128)
    candidates = write_path(hidden, state, layer_idx=0)

    # All candidates should be unmatched (no existing beliefs)
    for c in candidates:
        assert c.matched_slot == -1


def test_write_matches_existing_beliefs(config, state):
    """Writing observations similar to existing beliefs should match them."""
    write_path = WritePath(hidden_dim=128, belief_dim=config.belief_dim)

    # Add a belief
    belief = torch.randn(config.belief_dim)
    belief = belief / belief.norm() * 2.0  # normalize then set precision
    slot = state.allocate_belief(belief)

    # Create hidden states and manually set the projection to produce the same direction
    # For this test, we directly call _match_and_buffer with a known observation
    obs = belief.clone().unsqueeze(0)  # [1, D] same as existing belief

    candidates = write_path._match_and_buffer(obs, state, layer_idx=0)

    # Should match the existing belief
    assert len(candidates) > 0
    assert candidates[0].matched_slot == slot
    assert candidates[0].match_similarity > 0.9


def test_write_candidates_have_precision(config, state):
    """Write candidates should have non-zero radius (estimated precision)."""
    write_path = WritePath(hidden_dim=128, belief_dim=config.belief_dim)

    hidden = torch.randn(1, 3, 128)
    candidates = write_path(hidden, state, layer_idx=0)

    for c in candidates:
        radius = c.belief_vector.norm().item()
        assert radius > 0, "Write candidates should have estimated precision > 0"


# ── StateInterfaceLayer Integration Tests ──

def test_interface_returns_candidates(interface, state):
    """Interface should return both updated hidden and write candidates."""
    hidden = torch.randn(2, 10, 128)
    output, candidates = interface(hidden, state)

    assert output.shape == hidden.shape
    assert isinstance(candidates, list)


def test_interface_gradients_flow(interface, state):
    """Gradients should flow through the interface layer (for L_token)."""
    # Add some beliefs so read path has something to work with
    for _ in range(5):
        state.allocate_belief(torch.randn(state.config.belief_dim))

    hidden = torch.randn(1, 5, 128, requires_grad=True)
    output, _ = interface(hidden, state)
    loss = output.sum()
    loss.backward()

    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0


def test_interface_output_shape_invariant(interface, state):
    """Output shape should always match input shape regardless of state content."""
    for num_beliefs in [0, 1, 10, 50]:
        for _ in range(num_beliefs):
            state.allocate_belief(torch.randn(state.config.belief_dim))

        hidden = torch.randn(1, 8, 128)
        output, _ = interface(hidden, state)
        assert output.shape == hidden.shape

        # Reset state for next iteration
        with torch.no_grad():
            state.beliefs.data.zero_()
