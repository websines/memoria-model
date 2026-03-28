"""MemoriaModel: transformer + cognitive state + state interface layers.

This is the full model. It:
1. Runs input through transformer blocks
2. At every K blocks, routes through a state interface layer (read + write)
3. Collects write candidates from all interface layers
4. Returns logits + write candidates (for pass 2)

The cognitive state is attached to the model and persists across forward calls.
Pass 2 is NOT called inside forward() — it's called by the training loop after
computing L_token, so gradients from L_token flow through the interface layers
but not into the state content.

Reference: Griffin (google-deepmind/recurrentgemma) — hybrid architecture
Reference: TTT layers (test-time-training/ttt-lm-pytorch) — mutable state in forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MemoriaConfig
from .transformer import Transformer
from ..core.state import CognitiveState
from ..core.free_energy import compute_free_energy
from ..interface.layer import StateInterfaceLayer
from ..interface.write_path import WriteCandidate


class MemoriaModel(nn.Module):
    """Full Memoria architecture: language backbone + cognitive state.

    Forward pass:
        tokens → transformer blocks interleaved with state interface layers → logits
        + write candidates collected for pass 2

    The model exposes:
        - forward(): standard forward pass, returns logits + candidates
        - compute_loss(): L_token + α * L_fe
        - The cognitive state (self.state) for pass 2 access
    """

    def __init__(self, config: MemoriaConfig):
        super().__init__()
        self.config = config

        # Language backbone
        self.transformer = Transformer(config.transformer)

        # Cognitive state (persistent, not updated by optimizer)
        self.state = CognitiveState(config.state)

        # State interface layers, inserted every K transformer blocks
        n_interfaces = config.transformer.n_layer // config.transformer.interface_every
        self.interfaces = nn.ModuleList([
            StateInterfaceLayer(
                hidden_dim=config.transformer.n_embd,
                belief_dim=config.state.belief_dim,
                num_heads=config.transformer.interface_num_heads,
                top_k=config.transformer.interface_top_k,
                layer_idx=i,
            )
            for i in range(n_interfaces)
        ])

        # Track which transformer block each interface sits after
        self.interface_positions = [
            (i + 1) * config.transformer.interface_every - 1
            for i in range(n_interfaces)
        ]
        # e.g., with 12 layers and interface_every=4: positions [3, 7, 11]

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights."""
        self.transformer.init_weights()
        # Interface layers use default PyTorch init (output projections already zero-init)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> dict:
        """Forward pass through transformer + state interfaces.

        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices (optional, for loss computation)

        Returns:
            dict with:
                logits: [B, T, vocab_size]
                loss_token: scalar (if targets provided)
                candidates: list of WriteCandidate from all interface layers
                hidden_final: [B, T, n_embd] final hidden state
        """
        B, T = idx.size()
        cos = self.transformer.cos[:, :T]
        sin = self.transformer.sin[:, :T]

        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        all_candidates: list[WriteCandidate] = []
        all_utility_logits: list[Tensor] = []
        interface_idx = 0

        for i, block in enumerate(self.transformer.blocks):
            x = block(
                x, x0, cos, sin,
                self.transformer.resid_lambdas[i],
                self.transformer.x0_lambdas[i],
            )

            # Insert state interface after designated blocks
            if interface_idx < len(self.interfaces) and i == self.interface_positions[interface_idx]:
                x, candidates, utility_logits = self.interfaces[interface_idx](x, self.state)
                all_candidates.extend(candidates)
                all_utility_logits.append(utility_logits)
                interface_idx += 1

        logits = self.transformer.head(x)

        result = {
            'logits': logits,
            'candidates': all_candidates,
            'utility_logits': all_utility_logits,
            'hidden_final': x,
        }

        if targets is not None:
            result['loss_token'] = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return result

    def compute_loss(
        self,
        idx: Tensor,
        targets: Tensor,
        alpha: float = 0.0,
    ) -> dict:
        """Compute combined loss: L_token + α * L_fe.

        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices
            alpha: weight for L_fe (0 during phase 1, ramps up in phase 2)

        Returns:
            dict with:
                loss: combined scalar loss for backward()
                loss_token: L_token scalar
                loss_fe: L_fe scalar (0 if alpha=0)
                free_energy_stats: dict from compute_free_energy
                candidates: write candidates for pass 2
                logits: [B, T, vocab_size]
        """
        fwd = self.forward(idx, targets)

        loss_token = fwd['loss_token']
        result = {
            'loss_token': loss_token,
            'candidates': fwd['candidates'],
            'logits': fwd['logits'],
        }

        # Utility loss: do retrieved beliefs help predict next tokens?
        # This provides gradient signal to the read/write paths about state quality.
        loss_utility = torch.tensor(0.0, device=idx.device)
        if alpha > 0 and fwd['utility_logits']:
            lm_head = self.transformer.lm_head
            for util_hidden in fwd['utility_logits']:
                # Project utility hidden through LM head to get token predictions
                util_logits = lm_head(F.rms_norm(util_hidden, (util_hidden.size(-1),))).float()
                loss_utility = loss_utility + F.cross_entropy(
                    util_logits.view(-1, util_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
            loss_utility = loss_utility / len(fwd['utility_logits'])
        result['loss_utility'] = loss_utility

        if alpha > 0:
            fe_stats = compute_free_energy(self.state, self.config.training.fe_temperature)
            loss_fe = fe_stats['free_energy']
            result['loss_fe'] = loss_fe
            result['free_energy_stats'] = fe_stats
            # Utility loss weighted at 0.1× alpha — gentle signal, doesn't dominate
            result['loss'] = loss_token + alpha * loss_fe + alpha * 0.1 * loss_utility
        else:
            result['loss_fe'] = torch.tensor(0.0, device=idx.device)
            result['free_energy_stats'] = {}
            result['loss'] = loss_token

        return result

    def detach_state(self):
        """Detach cognitive state from computation graph.

        Called at sequence boundaries during training to prevent
        BPTT across sequences. Only pass 2 updates the state content.
        """
        with torch.no_grad():
            self.state.beliefs.data = self.state.beliefs.data.clone()
            self.state.edge_relations.data = self.state.edge_relations.data.clone()
            self.state.edge_weights.data = self.state.edge_weights.data.clone()
            self.state.goal_embeddings.data = self.state.goal_embeddings.data.clone()
            self.state.goal_metadata.data = self.state.goal_metadata.data.clone()
            self.state.meta.data = self.state.meta.data.clone()

    def num_parameters(self) -> dict:
        """Count parameters by component."""
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        interface_params = sum(p.numel() for p in self.interfaces.parameters())
        # State params don't count (not trained by optimizer)
        return {
            'transformer': transformer_params,
            'interface': interface_params,
            'total_trainable': transformer_params + interface_params,
            'state_beliefs': self.state.beliefs.numel(),
            'state_edges': self.state.edge_relations.numel() + self.state.edge_weights.numel(),
            'state_goals': self.state.goal_embeddings.numel() + self.state.goal_metadata.numel(),
        }

    def summary(self) -> str:
        """Human-readable model summary."""
        params = self.num_parameters()
        return (
            f"MemoriaModel:\n"
            f"  Transformer: {params['transformer'] / 1e6:.1f}M params\n"
            f"  Interfaces:  {params['interface'] / 1e6:.1f}M params ({len(self.interfaces)} layers)\n"
            f"  Total trainable: {params['total_trainable'] / 1e6:.1f}M params\n"
            f"  {self.state.summary()}\n"
            f"  Interface positions: {self.interface_positions}"
        )
