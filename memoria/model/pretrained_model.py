"""PretrainedMemoriaModel: bolt cognitive state onto a frozen HuggingFace LLM.

Mode 2 of Memoria. Instead of training a transformer from scratch, we:
1. Load a pretrained LLM (e.g., Qwen3.5-2B-Base) and freeze it completely
2. Insert state interface layers between its transformer blocks
3. Only train the interface layer parameters (~25M)
4. Run pass 2 exactly as before to evolve the cognitive state

The pretrained model handles language. The cognitive state handles memory,
belief tracking, causal reasoning, and goal-directed behavior.

Why this works:
- The LLM already knows language, code, reasoning — skip years of pretraining
- Interface layers are adapters: small, fast to train, proven effective
- Read path output proj initialized to zero → starts as identity (doesn't break LLM)
- The crossover test becomes meaningful: "LLM + state vs bigger LLM"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MemoriaConfig
from ..core.state import CognitiveState
from ..core.free_energy import compute_free_energy
from ..core.losses import chunked_cross_entropy, compute_differentiable_free_energy
from ..interface.layer import StateInterfaceLayer
from ..interface.write_path import WriteCandidate


class PretrainedMemoriaModel(nn.Module):
    """Frozen pretrained LLM + trainable state interface layers + cognitive state.

    The HF model is loaded in eval mode with all params frozen.
    We hook into its forward pass to insert interface layers between
    transformer blocks, giving the model access to persistent cognitive state.
    """

    def __init__(self, config: MemoriaConfig):
        super().__init__()
        self.config = config

        # Load and freeze the pretrained backbone
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading pretrained backbone: {config.pretrained_model}")
        hf_config = AutoConfig.from_pretrained(config.pretrained_model)

        # Handle models with nested text_config (e.g., Qwen3.5 multimodal)
        text_config = getattr(hf_config, 'text_config', hf_config)
        hidden_dim = text_config.hidden_size
        n_layers = text_config.num_hidden_layers

        # Verify dimensions match
        assert hidden_dim == config.transformer.n_embd, (
            f"Config n_embd ({config.transformer.n_embd}) doesn't match "
            f"pretrained hidden_size ({hidden_dim})"
        )
        assert n_layers == config.transformer.n_layer, (
            f"Config n_layer ({config.transformer.n_layer}) doesn't match "
            f"pretrained num_hidden_layers ({n_layers})"
        )

        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Cognitive state (persistent, not updated by optimizer)
        self.state = CognitiveState(config.state)

        # State interface layers, inserted every K transformer blocks
        n_interfaces = n_layers // config.transformer.interface_every
        self.interfaces = nn.ModuleList([
            StateInterfaceLayer(
                hidden_dim=hidden_dim,
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

        # Read path gate: starts at 0, learned scalar that controls how much
        # belief information is injected. Prevents disrupting the pretrained model
        # before interfaces have learned useful projections.
        self.read_gate = nn.Parameter(torch.zeros(n_interfaces))

        self._hidden_dim = hidden_dim
        self._n_layers = n_layers

    @torch.no_grad()
    def init_weights(self):
        """Initialize interface weights. Backbone is already pretrained."""
        # Interface layers use default init (output projections already zero-init in ReadPath)
        # Read gate starts at 0 (already initialized above)
        pass

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        alpha: float = 0.0,
    ) -> dict:
        """Forward pass: frozen backbone with interface layer injection.

        We can't easily hook into HF model internals layer-by-layer in a clean way,
        so we access the model's layers directly and run them manually.

        Args:
            idx: [B, T] token indices
            targets: [B, T] target indices (optional, for loss computation)
            alpha: weight for L_fe (0 during phase 1)

        Returns:
            dict with scalars + candidates
        """
        B, T = idx.size()

        # Access backbone model internals
        # Standard HF structure: model.model.embed_tokens -> model.model.layers[] -> model.model.norm -> model.lm_head
        backbone_model = self.backbone.model  # the inner transformer
        lm_head = self.backbone.lm_head

        # Embedding (no trainable params before first interface, safe to no_grad)
        with torch.no_grad():
            hidden = backbone_model.embed_tokens(idx)

            # Prepare position info for backbone layers
            position_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)

            # Some models (Qwen3.5) need precomputed rotary embeddings
            position_embeddings = None
            if hasattr(backbone_model, 'rotary_emb'):
                position_embeddings = backbone_model.rotary_emb(hidden, position_ids)

        # First interface position — layers before this can use no_grad
        first_interface = self.interface_positions[0] if self.interface_positions else self._n_layers

        all_candidates: list[WriteCandidate] = []
        all_utility_logits: list[Tensor] = []
        all_read_indices: list[int] = []
        all_attn_weights: list[Tensor] = []
        all_retrieved: list[Tensor] = []
        all_obs_vectors: list[Tensor] = []
        interface_idx = 0
        # Track if we've passed any interface (gradient chain must be maintained after)
        grad_active = False

        # Build layer kwargs once
        layer_kwargs = {'use_cache': False}
        if position_embeddings is not None:
            layer_kwargs['position_embeddings'] = position_embeddings
        layer_kwargs['position_ids'] = position_ids

        # Run through transformer layers with interface injection
        for i, layer in enumerate(backbone_model.layers):
            if not grad_active and i <= first_interface:
                # Before first interface: no trainable path yet, save memory
                with torch.no_grad():
                    layer_output = layer(hidden, **layer_kwargs)
                    hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            else:
                # After first interface: gradients must flow through frozen layers
                # to reach interface params. Frozen weights (requires_grad=False) won't
                # accumulate grads, but the computation graph is preserved for inputs.
                # Use gradient checkpointing to trade compute for memory — recompute
                # activations during backward instead of storing them (saves ~10GB).
                hidden = torch.utils.checkpoint.checkpoint(
                    self._run_layer, layer, hidden, layer_kwargs,
                    use_reentrant=False,
                )

            # Insert state interface after designated layers
            if interface_idx < len(self.interfaces) and i == self.interface_positions[interface_idx]:
                # Cast to float32 for interface layers (backbone outputs bfloat16)
                hidden_f32 = hidden.float()

                interface_out, candidates, utility_logits, read_indices, attn_w, retrieved, obs_v = \
                    self.interfaces[interface_idx](hidden_f32, self.state)

                # Gated residual: read_gate controls how much belief info is injected
                gate = torch.sigmoid(self.read_gate[interface_idx])
                hidden = (hidden_f32 + gate * (interface_out - hidden_f32)).to(hidden.dtype)
                # Note: interface_out = hidden + belief_info (from layer.py),
                # so (interface_out - hidden) = belief_info. This applies gate to just the belief signal.

                all_candidates.extend(candidates)
                all_utility_logits.append(utility_logits)
                all_read_indices.extend(read_indices)
                all_attn_weights.append(attn_w)
                all_retrieved.append(retrieved)
                all_obs_vectors.append(obs_v)
                interface_idx += 1
                grad_active = True  # from here on, gradients must flow

        # Final norm — must preserve grad chain
        hidden = backbone_model.norm(hidden)

        result = {
            'candidates': all_candidates,
            'read_indices': all_read_indices,
        }

        if targets is not None:
            # LM head (frozen, but we need grad through interface paths)
            logits = lm_head(hidden)
            result['loss_token'] = chunked_cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # Free energy loss (differentiable — provides actual gradients to interfaces)
            if alpha > 0:
                loss_fe = compute_differentiable_free_energy(
                    all_attn_weights, all_retrieved, all_obs_vectors,
                    self.config.state.belief_dim,
                )
                result['loss_fe'] = loss_fe

                # State free energy for beta/stats only (no gradients)
                with torch.no_grad():
                    fe_stats = compute_free_energy(self.state, self.config.training.fe_temperature)
                    result['free_energy_stats'] = fe_stats

                # Utility loss from interface layers
                loss_utility = torch.tensor(0.0, device=idx.device)
                if all_utility_logits:
                    for util_hidden in all_utility_logits:
                        util_logits = lm_head(util_hidden)
                        loss_utility = loss_utility + chunked_cross_entropy(
                            util_logits.view(-1, util_logits.size(-1)),
                            targets.view(-1),
                        )
                    loss_utility = loss_utility / len(all_utility_logits)
                result['loss_utility'] = loss_utility

                result['loss'] = result['loss_token'] + alpha * loss_fe + alpha * 0.1 * loss_utility
            else:
                result['loss_fe'] = torch.tensor(0.0, device=idx.device)
                result['loss'] = result['loss_token']
        else:
            logits = lm_head(hidden)
            result['logits'] = logits

        return result

    @staticmethod
    def _run_layer(layer, hidden, layer_kwargs):
        """Run a single backbone layer. Wrapped for gradient checkpointing."""
        output = layer(hidden, **layer_kwargs)
        return output[0] if isinstance(output, tuple) else output

    def detach_state(self):
        """No-op. State tensors have requires_grad=False and are accessed via .data,
        so they are never part of the computation graph."""
        pass

    def num_parameters(self) -> dict:
        """Count parameters by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        interface_params = sum(p.numel() for p in self.interfaces.parameters())
        gate_params = self.read_gate.numel()
        return {
            'backbone': backbone_params,
            'backbone_frozen': True,
            'interface': interface_params + gate_params,
            'total_trainable': interface_params + gate_params,
            'state_beliefs': self.state.beliefs.numel(),
            'state_edges': self.state.edge_relations.numel() + self.state.edge_weights.numel(),
            'state_goals': self.state.goal_embeddings.numel() + self.state.goal_metadata.numel(),
        }

    def summary(self) -> str:
        """Human-readable model summary."""
        params = self.num_parameters()
        return (
            f"PretrainedMemoriaModel ({self.config.pretrained_model}):\n"
            f"  Backbone:    {params['backbone'] / 1e9:.2f}B params (frozen)\n"
            f"  Interfaces:  {params['interface'] / 1e6:.1f}M params ({len(self.interfaces)} layers, trainable)\n"
            f"  Total trainable: {params['total_trainable'] / 1e6:.1f}M params\n"
            f"  {self.state.summary()}\n"
            f"  Interface positions: {self.interface_positions}\n"
            f"  Read gates: {[f'{torch.sigmoid(g).item():.3f}' for g in self.read_gate]}"
        )
