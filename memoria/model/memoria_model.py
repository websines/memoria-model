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
from ..core.free_energy import compute_free_energy, compute_bethe_free_energy
from ..core.losses import chunked_cross_entropy, compute_differentiable_free_energy
from ..core.ttt import InPlaceTTT, TTTContext
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

        # In-Place TTT: fast-weight deltas on upper MLP.c_proj layers
        self.ttt = InPlaceTTT(
            hidden_dim=config.transformer.n_embd,
            n_layers=config.transformer.n_layer,
        )

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
        all_read_indices: list[int] = []
        all_attn_weights: list[Tensor] = []
        all_retrieved: list[Tensor] = []
        all_obs_vectors: list[Tensor] = []
        interface_idx = 0
        ttt_ctx = TTTContext()

        for i, block in enumerate(self.transformer.blocks):
            x = block(
                x, x0, cos, sin,
                self.transformer.resid_lambdas[i],
                self.transformer.x0_lambdas[i],
            )

            # Insert state interface after designated blocks
            if interface_idx < len(self.interfaces) and i == self.interface_positions[interface_idx]:
                x, candidates, utility_logits, read_indices, attn_w, retrieved, obs_v = \
                    self.interfaces[interface_idx](x, self.state)
                all_candidates.extend(candidates)
                all_utility_logits.append(utility_logits)
                all_read_indices.extend(read_indices)
                all_attn_weights.append(attn_w)
                all_retrieved.append(retrieved)
                all_obs_vectors.append(obs_v)
                ttt_ctx.record_utility(interface_idx, i, utility_logits)
                interface_idx += 1

            # In-Place TTT: apply fast-weight delta to MLP output on designated layers
            if self.ttt.is_ttt_layer(i):
                utility = ttt_ctx.get_utility(i)
                x = self.ttt.apply_ttt_update(
                    layer_idx=i, hidden=x, targets=targets if targets is not None else idx,
                    lm_head=self.transformer.lm_head, utility_signal=utility,
                    fe_value=ttt_ctx.fe_value,
                )

        result = {
            'candidates': all_candidates,
            'read_indices': all_read_indices,
        }

        if targets is not None:
            # Compute loss_token inside forward — logits never leave this scope
            logits = self.transformer.head(x)
            result['loss_token'] = chunked_cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # Always compute L_fe and L_utility so all interface parameters participate
            # in the computation graph (required by DDP). When α=0, these contribute
            # zero to the loss but keep parameters in the graph for gradient reduction.

            # Utility loss
            loss_utility = torch.tensor(0.0, device=idx.device)
            if all_utility_logits:
                if alpha > 0:
                    # Full utility: run through LM head (materializes [B*T, vocab] per layer)
                    for util_hidden in all_utility_logits:
                        util_logits = self.transformer.head(util_hidden)
                        loss_utility = loss_utility + chunked_cross_entropy(
                            util_logits.view(-1, util_logits.size(-1)),
                            targets.view(-1),
                        )
                    loss_utility = loss_utility / len(all_utility_logits)
                else:
                    # Cheap graph participation: keep utility_head in the computation
                    # graph for DDP without materializing the huge vocab logit tensor
                    for util_hidden in all_utility_logits:
                        loss_utility = loss_utility + util_hidden.sum() * 0.0
            result['loss_utility'] = loss_utility

            # Interface proxy: trains read/write paths (prediction error + attention entropy)
            loss_fe_proxy = compute_differentiable_free_energy(
                all_attn_weights, all_retrieved, all_obs_vectors,
                self.config.state.belief_dim,
                fe_lambda=self.state.meta_params.fe_lambda,
            )

            # Proper Bethe free energy: trains beliefs, edges, relations through the
            # factor graph. Uses Power Spherical entropy with (d_i - 1) counting correction.
            if alpha > 0:
                fe_result = compute_bethe_free_energy(
                    self.state, self.state.meta_params.fe_temperature.item(),
                )
                loss_fe_bethe = fe_result['free_energy']
                result['free_energy_stats'] = fe_result
            else:
                loss_fe_bethe = torch.tensor(0.0, device=idx.device)

            # Combined: proxy trains interfaces, Bethe trains the world model
            loss_fe = loss_fe_proxy + loss_fe_bethe
            result['loss_fe'] = loss_fe
            result['loss_fe_proxy'] = loss_fe_proxy
            result['loss_fe_bethe'] = loss_fe_bethe

            # Telos: differentiable goal system (RND surprise + progress + transitions)
            if alpha > 0:
                active_mask = self.state.get_active_mask()
                # RND surprise loss (trains predictor to match target for seen beliefs)
                loss_surprise = self.state.telos.surprise_loss(
                    self.state.beliefs[active_mask] if active_mask.any()
                    else self.state.beliefs[:0]
                )
                # Goal progress estimation (differentiable)
                active_goals_mask = self.state.goal_status_logits[:, 2] > self.state.goal_status_logits[:, 0]  # active > empty
                if active_goals_mask.any():
                    active_goal_embeds = self.state.goal_embeddings[active_goals_mask]
                    progress = self.state.telos.estimate_progress(
                        active_goal_embeds, self.state.beliefs, active_mask,
                    )
                    # Update status transitions
                    active_status_logits = self.state.goal_status_logits[active_goals_mask]
                    new_logits = self.state.telos.update_status(
                        active_goal_embeds, active_status_logits, progress,
                        self.state.beliefs, active_mask,
                    )
                    # Write back — keep in graph so transition net gets gradients
                    # through the Bethe free energy's telos energy term
                    self.state.goal_status_logits.data[active_goals_mask] = new_logits.detach()
                    # But also add a loss term that depends on new_logits to train the transition net
                    # Encourage active goals (status 2) over stalled/failed — soft prior
                    status_probs = torch.nn.functional.gumbel_softmax(
                        new_logits, tau=self.state.telos.gumbel_tau.item(), hard=False,
                    )
                    # Penalize stalled (3) + failed (5), reward completed (4)
                    loss_surprise = loss_surprise + (
                        (status_probs[:, 3] + status_probs[:, 5] - status_probs[:, 4]).mean() * 0.1
                    )
                    result['goal_progress'] = progress.mean().item() if len(progress) > 0 else 0.0
                else:
                    loss_surprise = loss_surprise + self.state.goal_embeddings.sum() * 0.0  # DDP graph participation
                result['loss_surprise'] = loss_surprise
            else:
                result['loss_surprise'] = torch.tensor(0.0, device=idx.device)
                # DDP graph participation for telos params
                for p in self.state.telos.parameters():
                    if p.requires_grad:
                        result['loss_surprise'] = result['loss_surprise'] + p.sum() * 0.0

            result['loss'] = (
                result['loss_token']
                + alpha * loss_fe
                + alpha * 0.1 * loss_utility
                + alpha * 0.1 * result['loss_surprise']
            )
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
        """Detach state tensors from computation graph before structural pass2 modifications."""
        self.state.beliefs.detach_()
        self.state.edge_weights.detach_()
        self.state.edge_relations.detach_()
        self.state.goal_embeddings.detach_()

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
