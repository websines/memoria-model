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


def chunked_cross_entropy(logits: Tensor, targets: Tensor, chunk_size: int = 4096, ignore_index: int = -1) -> Tensor:
    """Cross-entropy computed in chunks to avoid materializing [B*T, vocab] in memory.

    Standard for large-vocab models (151K vocab × 8K tokens = 9+ GiB in float32).
    Chunks along the sequence dimension to keep peak memory bounded.
    """
    BT, V = logits.shape
    if BT <= chunk_size:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    total_loss = 0.0
    n_tokens = 0
    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        valid = (chunk_targets != ignore_index).sum().item()
        if valid > 0:
            total_loss = total_loss + F.cross_entropy(
                chunk_logits, chunk_targets, ignore_index=ignore_index, reduction='sum'
            )
            n_tokens += valid

    return total_loss / max(n_tokens, 1)


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
        alpha: float = 0.0,
    ) -> dict:
        """Forward pass through transformer + state interfaces + loss computation.

        Loss is computed inside forward() so that the large logits tensor
        [B, T, vocab_size] is never returned through DDP (which would clone it).

        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices (optional, for loss computation)
            alpha: weight for L_fe (0 during phase 1)

        Returns:
            dict with scalars + candidates (no large tensors to avoid DDP OOM)
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

        result = {
            'candidates': all_candidates,
        }

        if targets is not None:
            # Compute loss_token inside forward — logits never leave this scope
            logits = self.transformer.head(x)
            result['loss_token'] = chunked_cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # Utility loss
            loss_utility = torch.tensor(0.0, device=idx.device)
            if alpha > 0 and all_utility_logits:
                lm_head = self.transformer.lm_head
                for util_hidden in all_utility_logits:
                    util_logits = lm_head(F.rms_norm(util_hidden, (util_hidden.size(-1),))).float()
                    loss_utility = loss_utility + chunked_cross_entropy(
                        util_logits.view(-1, util_logits.size(-1)),
                        targets.view(-1),
                    )
                loss_utility = loss_utility / len(all_utility_logits)
            result['loss_utility'] = loss_utility

            # Free energy loss
            if alpha > 0:
                fe_stats = compute_free_energy(self.state, self.config.training.fe_temperature)
                result['loss_fe'] = fe_stats['free_energy']
                result['free_energy_stats'] = fe_stats
                result['loss'] = result['loss_token'] + alpha * result['loss_fe'] + alpha * 0.1 * loss_utility
            else:
                result['loss_fe'] = torch.tensor(0.0, device=idx.device)
                result['loss'] = result['loss_token']
        else:
            logits = self.transformer.head(x)
            result['logits'] = logits

        return result

    def compute_loss(
        self,
        idx: Tensor,
        targets: Tensor,
        alpha: float = 0.0,
    ) -> dict:
        """Compute combined loss: L_token + α * L_fe.

        Convenience wrapper around forward() for non-DDP usage and tests.
        """
        return self.forward(idx, targets, alpha=alpha)

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
