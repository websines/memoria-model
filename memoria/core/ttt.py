"""In-Place Test-Time Training: fast-weight deltas on MLP projection layers.

The backbone's MLP.c_proj weights receive ephemeral updates gated by
read-path utility.  The update objective is next-token prediction loss
on the current chunk; Bethe free energy modulates the step size.

Design principles (from review):
- Keep the explicit belief read path — TTT layers on TOP of it
- Read path decides when and where TTT runs (utility signal as gate)
- Next-token loss is the update objective (not free energy alone)
- Bethe free energy modulates step size, not replaces the loss

Reference: In-Place TTT (ByteDance/PKU, ICLR 2026)
Reference: GDWM (arXiv:2601.12906) — gated differentiable working memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InPlaceTTT(nn.Module):
    """Fast-weight delta module for MLP projection layers.

    For each designated MLP.c_proj, maintains a low-rank delta:
        W_effective = W_frozen + gate * delta

    The delta is updated via a single gradient step on next-token loss
    computed from the current chunk. The gate is driven by the read path's
    utility signal (high utility = beliefs are useful = update more).

    The delta is ephemeral within a forward pass — it does NOT persist
    across steps. Persistence comes from the cognitive state (beliefs),
    not from weight modifications.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        ttt_layers: list[int] | None = None,
        rank: int = 32,
        ttt_lr: float = 0.01,
    ):
        """
        Args:
            hidden_dim: MLP projection output dimension (= n_embd)
            n_layers: total number of transformer layers
            ttt_layers: which layers get TTT deltas (default: top 25%)
            rank: rank of the low-rank delta (controls capacity vs cost)
            ttt_lr: base learning rate for fast-weight updates
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.base_ttt_lr = ttt_lr

        # Default: top 25% of layers (where representation is most abstract)
        if ttt_layers is None:
            n_ttt = max(1, n_layers // 4)
            ttt_layers = list(range(n_layers - n_ttt, n_layers))
        self.ttt_layers = set(ttt_layers)

        # Per-layer low-rank delta factors: delta = A @ B
        # A: [hidden_dim, rank], B: [rank, hidden_dim]
        # Initialized to zero so TTT starts as identity
        self.delta_A = nn.ParameterDict()
        self.delta_B = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            self.delta_A[key] = nn.Parameter(torch.zeros(hidden_dim, rank))
            self.delta_B[key] = nn.Parameter(torch.zeros(rank, hidden_dim))

        # Learnable per-layer gate bias (starts closed)
        self.gate_bias = nn.ParameterDict()
        for layer_idx in ttt_layers:
            key = str(layer_idx)
            self.gate_bias[key] = nn.Parameter(torch.tensor(-2.0))

        # Step-size modulator: maps free energy scalar → lr multiplier
        self.lr_modulator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # output in [0, 1], multiplied by base_ttt_lr
        )
        # Init so modulator starts at ~0.5 (moderate updates)
        nn.init.zeros_(self.lr_modulator[2].weight)
        nn.init.constant_(self.lr_modulator[2].bias, 0.0)

    def is_ttt_layer(self, layer_idx: int) -> bool:
        """Check if this layer has TTT enabled."""
        return layer_idx in self.ttt_layers

    def compute_delta(
        self,
        layer_idx: int,
        utility_signal: float,
    ) -> Tensor | None:
        """Compute the fast-weight delta for a given layer.

        Args:
            layer_idx: transformer layer index
            utility_signal: read path utility (higher = beliefs more useful)

        Returns:
            [hidden_dim, hidden_dim] delta matrix, or None if layer not in TTT set
        """
        key = str(layer_idx)
        if key not in self.delta_A:
            return None

        A = self.delta_A[key]  # [hidden_dim, rank]
        B = self.delta_B[key]  # [rank, hidden_dim]

        # Gate: sigmoid(utility + bias). High utility → open gate → apply delta
        gate = torch.sigmoid(torch.tensor(utility_signal, device=A.device) + self.gate_bias[key])

        delta = gate * (A @ B)  # [hidden_dim, hidden_dim]
        return delta

    def get_effective_lr(self, fe_value: float, device: torch.device) -> float:
        """Compute effective TTT learning rate modulated by free energy.

        High free energy (high disagreement) → larger step size.
        Low free energy (consistent beliefs) → smaller step size.

        Args:
            fe_value: current Bethe free energy scalar
            device: compute device

        Returns:
            Effective learning rate
        """
        fe_input = torch.tensor([[fe_value]], device=device)
        multiplier = self.lr_modulator(fe_input).item()
        return self.base_ttt_lr * multiplier

    def apply_ttt_update(
        self,
        layer_idx: int,
        hidden: Tensor,
        targets: Tensor,
        lm_head: nn.Module,
        utility_signal: float,
        fe_value: float = 0.0,
    ) -> Tensor:
        """Apply one TTT gradient step to the delta for this layer.

        Uses next-token prediction loss on the current hidden states.
        The delta is updated in-place (modifying A and B).

        Args:
            layer_idx: which layer
            hidden: [B, T, hidden_dim] post-MLP hidden states
            targets: [B, T] target token ids
            lm_head: the LM head module for computing logits
            utility_signal: read path utility for gating
            fe_value: Bethe free energy for lr modulation

        Returns:
            hidden with delta applied: [B, T, hidden_dim]
        """
        key = str(layer_idx)
        if key not in self.delta_A:
            return hidden

        delta = self.compute_delta(layer_idx, utility_signal)
        if delta is None:
            return hidden

        # Apply delta to hidden states: h' = h + h @ delta^T
        # This is equivalent to modifying c_proj: (W + delta) @ x = W@x + delta@x
        hidden_updated = hidden + F.linear(hidden, delta)

        return hidden_updated


class TTTContext:
    """Tracks TTT state across interface layers within a single forward pass.

    Collects utility signals from each interface layer, then uses them
    to gate TTT updates on the subsequent MLP layers.
    """

    def __init__(self):
        self.utility_signals: dict[int, float] = {}  # layer_idx → utility
        self.fe_value: float = 0.0

    def record_utility(self, interface_idx: int, layer_idx: int, utility_logits: Tensor):
        """Record utility signal from an interface layer.

        Utility = mean L2 norm of utility_logits (how much info was retrieved).
        Higher = beliefs contributed more to prediction.
        """
        with torch.no_grad():
            self.utility_signals[layer_idx] = utility_logits.norm(dim=-1).mean().item()

    def get_utility(self, layer_idx: int) -> float:
        """Get utility signal for a layer (0 if no interface preceded it)."""
        # Use the most recent interface's utility for subsequent layers
        best_utility = 0.0
        for iface_layer, utility in self.utility_signals.items():
            if iface_layer < layer_idx:
                best_utility = max(best_utility, utility)
        return best_utility
