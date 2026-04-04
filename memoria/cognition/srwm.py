"""C1: Self-Referential Weight Matrix (SRWM) for dynamic meta-parameter modulation.

Instead of MetaParams being fixed scalars (sigmoid/softplus of nn.Parameters),
the SRWM produces *context-dependent* modulations based on the current cognitive
state. This creates a two-timescale system:

  Slow (backprop):  MetaParams base values evolve over training
  Fast (Hebbian):   W_fast adapts per-step via outer-product updates

The SRWM maintains a low-rank fast-weight matrix updated by Hebbian learning:
    W_fast = (1 - decay) * W_fast + lr * outer(key, value)

Querying the matrix produces multiplicative modulation factors:
    modulation = 1 + tanh(output_proj(query @ W_fast))
    dynamic_param = base_param * modulation

This means MetaParams.hebbian_lr (for example) is no longer a fixed sigmoid
scalar — it becomes a *function of the current cognitive state*, learned end-to-end.

Reference: Self-Referential Weight Matrix (arXiv:2202.05780, Schmidhuber group)
Reference: REFINE: Reinforced Fast Weights (arXiv:2602.16704)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRWM(nn.Module):
    """Self-Referential Weight Matrix for dynamic meta-parameter modulation.

    Maintains a fast-weight matrix that adapts per-step via Hebbian outer
    products. State features are projected to key/value pairs that update the
    matrix; queries produce modulation factors for MetaParams.

    The rank parameter controls the capacity of the fast-weight matrix.
    Higher rank = more expressive modulations, but more parameters.

    Usage:
        srwm = SRWM(state_dim=256, n_meta_params=52, rank=32)
        # Each step:
        srwm.update(state_features, lr=meta_params.srwm_lr, decay=meta_params.srwm_decay)
        modulation = srwm.query(state_features)  # [n_meta_params], centered ~1.0
        dynamic_lr = base_lr * modulation[0]

    Args:
        state_dim: dimension of input state feature vector
        n_meta_params: number of meta-parameters to modulate
        rank: rank of fast-weight matrix (controls capacity vs cost)
    """

    def __init__(self, state_dim: int, n_meta_params: int, rank: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.n_meta_params = n_meta_params
        self.rank = rank

        # Projections for Hebbian update: state → key, value
        self.key_proj = nn.Linear(state_dim, rank)
        self.value_proj = nn.Linear(state_dim, rank)

        # Projection for querying: state → query
        self.query_proj = nn.Linear(state_dim, rank)

        # Output projection: rank → n_meta_params
        self.output_proj = nn.Linear(rank, n_meta_params)

        # Fast-weight matrix (not a parameter — updated by Hebbian rule)
        self.register_buffer('W_fast', torch.zeros(rank, rank))

        # Running norm tracker for stability (prevents W_fast from exploding)
        self.register_buffer('_update_count', torch.tensor(0, dtype=torch.long))

        # Initialize projections with small weights (modulations start near 0)
        nn.init.xavier_normal_(self.key_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.value_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.query_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def extract_state_features(self, state) -> torch.Tensor:
        """Extract a fixed-size feature vector from cognitive state.

        Computes statistics from the belief, edge, goal, and meta regions.
        All features are normalized to roughly [-1, 1] for stable learning.

        Args:
            state: CognitiveState instance

        Returns:
            [state_dim] feature vector on the same device as state.beliefs
        """
        device = state.beliefs.device
        max_b = max(state.config.max_beliefs, 1)
        max_e = max(state.config.max_edges, 1)
        max_g = max(state.config.max_goals, 1)

        active_mask = state.get_active_mask()
        n_active = active_mask.sum().item()

        # Belief statistics
        radii = state.beliefs.data.norm(dim=-1)
        active_radii = radii[active_mask] if n_active > 0 else radii[:1]
        fill_ratio = n_active / max_b
        mean_radius = active_radii.mean()
        std_radius = active_radii.std() if n_active > 1 else torch.tensor(0.0, device=device)
        mean_var = state.belief_precision_var[active_mask].mean() if n_active > 0 else torch.tensor(1.0, device=device)
        mean_lr_scale = state.belief_lr_scale[active_mask].mean() if n_active > 0 else torch.tensor(1.0, device=device)

        # Edge statistics
        edge_fill = state.num_active_edges() / max_e
        n_causal = (state.edge_causal_obs > 0).sum().float() / max(state.num_active_edges(), 1)

        # Goal statistics
        n_goals = state.num_active_goals()
        goal_fill = n_goals / max_g

        # Meta region
        beta = state.meta.data[0]
        surprise = state.meta.data[1]

        # Pack into feature vector
        # Use first state_dim features, pad if needed
        raw_features = torch.stack([
            torch.tensor(fill_ratio, device=device),
            mean_radius,
            std_radius,
            mean_var,
            mean_lr_scale,
            torch.tensor(edge_fill, device=device),
            n_causal,
            torch.tensor(goal_fill, device=device),
            beta,
            surprise.clamp(-5, 5) / 5.0,  # normalize surprise
        ])

        # Pad to state_dim with zeros
        if raw_features.shape[0] < self.state_dim:
            padding = torch.zeros(self.state_dim - raw_features.shape[0], device=device)
            features = torch.cat([raw_features, padding])
        else:
            features = raw_features[:self.state_dim]

        return features

    def update(self, state_features: torch.Tensor, lr: torch.Tensor, decay: torch.Tensor):
        """Update fast-weight matrix via Hebbian outer product.

        W_fast = (1 - decay) * W_fast + lr * outer(key, value)

        The spectral norm of W_fast is clamped to prevent unbounded growth.
        This is the "write" operation of the self-referential system.

        Args:
            state_features: [state_dim] current state feature vector
            lr: scalar learning rate (from MetaParams.srwm_lr)
            decay: scalar decay rate (from MetaParams.srwm_decay)
        """
        with torch.no_grad():
            key = self.key_proj(state_features.unsqueeze(0)).squeeze(0)     # [rank]
            value = self.value_proj(state_features.unsqueeze(0)).squeeze(0)  # [rank]

            # Hebbian update with decay
            self.W_fast.mul_(1.0 - decay.item())
            self.W_fast.add_(lr.item() * torch.outer(key, value))

            # Spectral norm clamp: prevent explosion
            # Max singular value ≤ 1.0 ensures bounded modulations
            s = torch.linalg.svdvals(self.W_fast)
            max_sv = s[0].item()
            if max_sv > 1.0:
                self.W_fast.div_(max_sv)

            self._update_count += 1

    def query(self, state_features: torch.Tensor) -> torch.Tensor:
        """Query fast-weight matrix for meta-parameter modulations.

        Returns multiplicative factors centered around 1.0:
            modulation = 1 + tanh(output_proj(query @ W_fast))

        So modulation ∈ (0, 2): values > 1 amplify, < 1 attenuate.

        Args:
            state_features: [state_dim] current state feature vector

        Returns:
            [n_meta_params] multiplicative modulation factors
        """
        query = self.query_proj(state_features.unsqueeze(0)).squeeze(0)  # [rank]
        fast_out = query @ self.W_fast  # [rank]
        raw_modulation = self.output_proj(fast_out.unsqueeze(0)).squeeze(0)  # [n_meta_params]
        # tanh bounds to (-1, 1), then shift to (0, 2) centered at 1.0
        return 1.0 + torch.tanh(raw_modulation)

    def get_dynamic_params(self, state, base_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Get dynamically modulated meta-parameters.

        Extracts state features, queries the SRWM, and applies multiplicative
        modulations to the base MetaParams values.

        Args:
            state: CognitiveState instance
            base_params: dict mapping param names to their base (static) values

        Returns:
            dict mapping param names to dynamically modulated values
        """
        features = self.extract_state_features(state)
        modulations = self.query(features)  # [n_meta_params]

        # Apply modulations to base params
        dynamic = {}
        for i, (name, base_val) in enumerate(base_params.items()):
            if i < len(modulations):
                dynamic[name] = base_val * modulations[i]
            else:
                dynamic[name] = base_val

        return dynamic

    def reset_fast_weights(self):
        """Reset the fast-weight matrix to zero (fresh start)."""
        self.W_fast.zero_()
        self._update_count.zero_()

    def summary(self) -> str:
        """Human-readable summary."""
        sv = torch.linalg.svdvals(self.W_fast)
        return (
            f"SRWM: rank={self.rank}, updates={self._update_count.item()}, "
            f"W_fast spectral_norm={sv[0].item():.4f}, "
            f"mean_sv={sv.mean().item():.4f}"
        )
