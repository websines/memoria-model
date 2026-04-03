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
from ..interface.read_path import BeliefCache
from ..interface.write_path import WriteCandidate


class EngramCache(nn.Module):
    """O(1) static knowledge lookup via N-gram hashing.

    Handles common token patterns (entity names, fixed expressions) via
    deterministic hash-based retrieval, freeing belief slots for dynamic memory.

    Architecture (from DeepSeek Engram, arXiv:2601.07372):
    1. Tokenizer compression: NFKC + lowercase + accent strip so "Apple"
       and "apple" hash to the same slot (~23% vocab reduction in Engram paper)
    2. Compute 2-gram and 3-gram suffix hashes via multiplicative-XOR
    3. Retrieve embeddings from K hash heads per N-gram order
    4. Project and gate using current hidden state (suppress collisions)
    5. Add to residual stream

    The context-aware gate is critical: raw hash lookups are noisy due to
    collisions. The gate uses the transformer's current hidden state to
    suppress irrelevant retrievals.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        table_size: int = 50000,
        n_heads: int = 4,
        embed_dim_per_head: int = 0,
        tokenizer_name: str = "",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        if embed_dim_per_head <= 0:
            embed_dim_per_head = max(16, hidden_dim // (n_heads * 2))
        self.embed_dim = embed_dim_per_head

        # ── Tokenizer Compression (Engram's CompressedTokenizer) ──
        # Build a lookup table that maps raw token IDs to compressed IDs.
        # NFKC normalization + lowercase + accent stripping collapses
        # semantically equivalent tokens: "Apple", "apple", "APPLE" → same ID.
        # This makes the hash space denser and reduces wasted slots.
        compress_table = self._build_compression_table(vocab_size, tokenizer_name)
        self.register_buffer('compress_table', compress_table)

        # Hash tables: separate per head, per ngram order
        self.tables_2gram = nn.ModuleList([
            nn.Embedding(table_size, embed_dim_per_head) for _ in range(n_heads)
        ])
        self.tables_3gram = nn.ModuleList([
            nn.Embedding(table_size, embed_dim_per_head) for _ in range(n_heads)
        ])

        # Random multiplicative hash constants (fixed, odd numbers for distribution)
        self.register_buffer(
            'hash_mult',
            torch.randint(1, max(vocab_size, 2), (3,)) * 2 + 1,
        )
        self.register_buffer('table_size_t', torch.tensor(table_size))

        # Value projection: all retrieved embeddings → hidden_dim
        total_embed = n_heads * 2 * embed_dim_per_head
        self.value_proj = nn.Linear(total_embed, hidden_dim, bias=False)

        # Context-aware gating
        self.gate_norm_h = nn.RMSNorm(hidden_dim)
        self.gate_norm_v = nn.RMSNorm(hidden_dim)

        # Initialize value projection to zero so Engram starts as no-op
        nn.init.zeros_(self.value_proj.weight)

    @staticmethod
    def _build_compression_table(vocab_size: int, tokenizer_name: str) -> Tensor:
        """Build token ID compression table via NFKC + lowercase normalization.

        Maps raw token IDs to compressed IDs where semantically equivalent
        tokens (different casing, accents, whitespace) share the same ID.

        Falls back to identity mapping if tokenizer is unavailable.
        """
        import unicodedata

        if not tokenizer_name:
            # No tokenizer → identity mapping (no compression)
            return torch.arange(vocab_size, dtype=torch.long)

        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception:
            return torch.arange(vocab_size, dtype=torch.long)

        key_to_new = {}
        lookup = torch.arange(vocab_size, dtype=torch.long)
        actual_vocab = min(vocab_size, len(tok))

        for tid in range(actual_vocab):
            try:
                text = tok.decode([tid], skip_special_tokens=False)
                if '\ufffd' in text:
                    # Byte token — keep as-is (can't normalize)
                    key = f"__byte_{tid}"
                else:
                    # NFKC normalize + strip accents + lowercase + collapse whitespace
                    norm = unicodedata.normalize('NFKC', text)
                    # Strip combining marks (accents)
                    norm = ''.join(
                        c for c in unicodedata.normalize('NFD', norm)
                        if unicodedata.category(c) != 'Mn'
                    )
                    key = norm.lower().strip()
                    if not key:
                        key = text

                if key not in key_to_new:
                    key_to_new[key] = len(key_to_new)
                lookup[tid] = key_to_new[key]
            except Exception:
                pass  # keep identity mapping for this token

        return lookup

    def _compress_ids(self, input_ids: Tensor) -> Tensor:
        """Apply tokenizer compression to input IDs."""
        # Clamp to valid range for the lookup table
        clamped = input_ids.long().clamp(0, self.compress_table.shape[0] - 1)
        return self.compress_table[clamped]

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        """Retrieve and gate static N-gram knowledge.

        Args:
            hidden: [B, T, hidden_dim] current hidden states
            input_ids: [B, T] raw token IDs

        Returns:
            [B, T, hidden_dim] gated knowledge to add to residual stream
        """
        B, T = input_ids.shape

        # Compress token IDs (case/accent normalization)
        ids = self._compress_ids(input_ids)

        # Build shifted sequences for 2-gram and 3-gram construction
        shifted_1 = F.pad(ids[:, :-1], (1, 0), value=0)
        shifted_2 = F.pad(ids[:, :-2], (2, 0), value=0)

        # Multiplicative-XOR hash
        m = self.hash_mult
        hash_2 = (ids * m[0]) ^ (shifted_1 * m[1])
        hash_3 = hash_2 ^ (shifted_2 * m[2])

        ts = self.table_size_t.item()

        # Retrieve from all heads
        embeds = []
        for table in self.tables_2gram:
            embeds.append(table((hash_2 % ts).clamp(min=0)))
        for table in self.tables_3gram:
            embeds.append(table((hash_3 % ts).clamp(min=0)))

        e = torch.cat(embeds, dim=-1)
        v = self.value_proj(e)

        # Context-aware gate (sqrt-sign trick from Engram)
        h_norm = self.gate_norm_h(hidden)
        v_norm = self.gate_norm_v(v)
        gate = (h_norm * v_norm).sum(dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()

        return gate * v


class RefinementProbe(nn.Module):
    """Decides when to stop iterative refinement loops.

    Analogous to Mamba's HaltingHead: takes pooled hidden state + normalized
    loop position → P(halt). The loop position scalar is critical — without it,
    the probe has no sense of "how much compute has been spent."

    Reference: HaltingHead in phase14_inner_loop_bypass_trainer.py
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # +1 for loop position scalar (normalized to [0, 1])
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_pooled: Tensor, loop_idx: int, max_loops: int) -> Tensor:
        """Compute P(halt) from pooled hidden state and loop position.

        Args:
            h_pooled: [B, hidden_dim] mean-pooled hidden state
            loop_idx: current loop index (0-based)
            max_loops: maximum number of loops

        Returns:
            [B, 1] halt probability
        """
        B = h_pooled.shape[0]
        pos = torch.full((B, 1), loop_idx / max(max_loops, 1),
                         device=h_pooled.device, dtype=h_pooled.dtype)
        return self.net(torch.cat([h_pooled, pos], dim=-1))


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
                n_interfaces=n_interfaces,
                read_gate_init_bias=config.transformer.read_gate_init_bias,
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

        # ── Working Memory Prefix (Mamba-inspired scratchpad) ──
        # M learnable tokens prepended to hidden stream. Persist across forward
        # passes as fast scratch paper. Unlike beliefs (which require pass 2
        # allocation), working memory is free-form and writable by gradients.
        # Reference: RecursiveMamba2_PrefixScratchpad (batteryphil/mamba2backbonerecursion)
        self.working_memory_size = getattr(config.transformer, 'working_memory_size', 8)
        if self.working_memory_size > 0:
            self.working_memory = nn.Parameter(
                torch.randn(1, self.working_memory_size, config.transformer.n_embd)
                * config.transformer.working_memory_init_scale
            )

        # ── Engram Static Knowledge Cache ──
        # O(1) N-gram hash lookup for common token patterns. Frees belief slots
        # for dynamic/experiential memory by handling static knowledge separately.
        # Reference: Engram (DeepSeek, arXiv:2601.07372) — conditional memory
        # via scalable lookup as a complementary sparsity axis.
        # Tokenizer name for Engram compression table — empty string means no compression
        engram_tokenizer = config.pretrained_model if config.backbone == "pretrained" else ""
        self.engram_cache = EngramCache(
            hidden_dim=config.transformer.n_embd,
            vocab_size=config.transformer.vocab_size,
            table_size=getattr(config.transformer, 'engram_table_size', 50000),
            n_heads=getattr(config.transformer, 'engram_n_heads', 4),
            embed_dim_per_head=getattr(config.transformer, 'engram_embed_dim', 0),
            tokenizer_name=engram_tokenizer,
        )

        # ── Refinement Loops (Mamba-inspired iterative reasoning) ──
        # After the full forward pass, loop upper transformer blocks with the
        # belief-augmented representation. The model reasons over retrieved
        # beliefs before generating output.
        # Reference: phase14_inner_loop_bypass_trainer.py — dark loops
        self.max_refinement_loops = getattr(config.transformer, 'max_refinement_loops', 3)
        if self.max_refinement_loops > 0:
            # Refinement probe: decides when to stop iterating (Mamba's HaltingHead)
            self.refinement_probe = RefinementProbe(config.transformer.n_embd)
            # Lifeline gate: prevents hidden state drift during refinement loops
            # (Mamba's ROM re-injection, learned per-dimension scaling)
            self.refinement_gate = nn.Parameter(
                torch.full((config.transformer.n_embd,), config.transformer.refinement_gate_init)
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

        Architecture (with Engram/Mamba-inspired enhancements):
        1. Embed tokens + prepend working memory prefix (Mamba scratchpad)
        2. Engram static knowledge injection at layer 0 (O(1) hash lookup)
        3. Transformer blocks interleaved with state interface layers
           - Belief prefetch cache shared across all interfaces
           - Depth-conditioned retrieval (early=broad, late=focused)
        4. Refinement loops on upper layers (Mamba dark loops)
           - Retrieve-reason-retrieve cycle (re-query beliefs with refined repr)
           - HaltingHead probe decides when to stop
           - TTT gradient steps during refinement
        5. LM head → loss computation

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
        M = self.working_memory_size if self.working_memory_size > 0 else 0
        T_total = T + M

        self.transformer._ensure_rope(T_total)
        cos = self.transformer.cos[:, :T_total]
        sin = self.transformer.sin[:, :T_total]

        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))

        # ── Working Memory Suffix (Mamba-inspired scratchpad) ──
        # APPEND M learnable tokens after real tokens. In a causal transformer,
        # prefix tokens can only attend to each other (they're blind to future
        # tokens). By appending instead, the working memory can attend to ALL
        # real tokens — making it true scratch paper, not just a soft prompt.
        #
        # Real tokens cannot attend to WM (it's after them causally), but that's
        # fine — the model reads from WM via the interface layers, not attention.
        # WM's purpose is to accumulate state across refinement loops.
        #
        # Mamba's prefix scratchpad works because SSMs are linear recurrence
        # (the prefix CAN see future tokens). In a transformer we need suffix.
        if M > 0:
            wm = self.working_memory.expand(B, -1, -1)  # [B, M, H]
            x = torch.cat([x, wm], dim=1)               # [B, T+M, H]

        x0 = x

        # ── Engram Static Knowledge Injection (layer 0) ──
        # Hash-based O(1) lookup for common N-gram patterns. Runs BEFORE any
        # transformer block, augmenting the embedding with static knowledge.
        # The context-aware gate suppresses hash collision noise.
        # Only applied to real token positions (not working memory suffix).
        engram_out = self.engram_cache(x[:, :T, :], idx)  # [B, T, H]
        x = torch.cat([x[:, :T, :] + engram_out, x[:, T:, :]], dim=1) if M > 0 else x + engram_out

        # ── Belief Prefetch Cache ──
        # Snapshot active beliefs once per forward pass. All interface layers
        # share this cache, eliminating 3x redundant get_active_mask + indexing.
        # Also enables future DRAM offloading (prefetch from CPU during early layers).
        belief_cache = BeliefCache.from_state(self.state)

        all_candidates: list[WriteCandidate] = []
        all_utility_logits: list[Tensor] = []
        all_read_indices: list[int] = []
        all_attn_weights: list[Tensor] = []
        all_retrieved: list[Tensor] = []
        all_obs_vectors: list[Tensor] = []
        interface_idx = 0
        ttt_ctx = TTTContext()

        for i, block in enumerate(self.transformer.blocks):
            if self.training and T_total > self.config.transformer.sliding_window_size:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, x0, cos, sin,
                    self.transformer.resid_lambdas[i],
                    self.transformer.x0_lambdas[i],
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, x0, cos, sin,
                    self.transformer.resid_lambdas[i],
                    self.transformer.x0_lambdas[i],
                )

            # Insert state interface after designated blocks
            # Interface layers process ONLY real token positions (not working memory suffix)
            if interface_idx < len(self.interfaces) and i == self.interface_positions[interface_idx]:
                if M > 0:
                    x_tokens = x[:, :T, :]
                    x_tokens, candidates, utility_logits, read_indices, attn_w, retrieved, obs_v = \
                        self.interfaces[interface_idx](
                            x_tokens, self.state, belief_cache=belief_cache,
                        )
                    x = torch.cat([x_tokens, x[:, T:, :]], dim=1)
                else:
                    x, candidates, utility_logits, read_indices, attn_w, retrieved, obs_v = \
                        self.interfaces[interface_idx](
                            x, self.state, belief_cache=belief_cache,
                        )
                all_candidates.extend(candidates)
                all_utility_logits.append(utility_logits)
                all_read_indices.extend(read_indices)
                all_attn_weights.append(attn_w)
                all_retrieved.append(retrieved)
                all_obs_vectors.append(obs_v)
                ttt_ctx.record_utility(interface_idx, i, utility_logits)
                interface_idx += 1

            # In-Place TTT: apply persistent fast-weight delta
            if self.ttt.is_ttt_layer(i):
                ttt_ctx.save_pre_delta(i, x)
                x = self.ttt.apply_delta(layer_idx=i, hidden=x)

        # ── Refinement Loops (Mamba-inspired iterative reasoning) ──
        # After the full forward pass + belief retrieval, loop upper transformer
        # blocks with the belief-augmented representation. This lets the model
        # REASON OVER retrieved beliefs before generating output.
        #
        # CRITICAL: runs at BOTH training and inference. At inference time the model
        # iteratively refines its representation (dark loops) before the LM head
        # fires. This is the core "think before you speak" mechanism.
        #
        # Three things from Mamba that matter here:
        # 1. Working memory prefix is NOT anchored by the lifeline — it evolves
        #    freely as true scratch paper (Mamba's _lifeline_inject_prompt_only)
        # 2. Each loop gets a unique additive encoding so the model knows which
        #    iteration it's on (Mamba's LoopRoPE per loop_index)
        # 3. The HaltingHead decides when to stop — easy inputs get 1 loop,
        #    hard multi-hop reasoning gets the full budget
        #
        # Reference: Mamba dark loops (phase14_inner_loop_bypass_trainer.py)
        # Reference: RecursiveMamba2_PrefixScratchpad._lifeline_inject_prompt_only
        halt_probs = []
        n_refinement_loops = 0
        if self.max_refinement_loops > 0:
            # Re-run the last interface_every blocks (the final "cycle").
            # For 12 layers, interface_every=4: re-run blocks [8, 9, 10, 11].
            n_upper = self.config.transformer.interface_every
            upper_start = max(0, self.config.transformer.n_layer - n_upper)
            upper_blocks = self.transformer.blocks[upper_start:]
            last_interface = self.interfaces[-1] if len(self.interfaces) > 0 else None

            # Lifeline anchor: token positions ONLY. WM evolves freely.
            if M > 0:
                x_token_anchor = x[:, :T, :].detach()
            else:
                x_token_anchor = x.detach()

            # Training: teacher forcing with random oracle loop count.
            # Without this, the probe only ever sees "halt at max" targets and
            # never learns to halt early. Mamba samples from Uniform(min, max).
            # We sample the oracle loop count per step so the probe sees diverse
            # halt points: sometimes 1 loop is correct, sometimes 3.
            if self.training:
                import random
                oracle_loops = random.randint(1, self.max_refinement_loops)
            else:
                oracle_loops = self.max_refinement_loops  # inference: probe decides

            loop_limit = oracle_loops if self.training else self.max_refinement_loops

            for loop_i in range(loop_limit):
                n_refinement_loops += 1

                # Step 1: Loop-index encoding — additive signal so the model
                # knows which iteration it's on. Separate parameter from lifeline
                # gate to avoid conflating two functions.
                loop_fraction = loop_i / max(self.max_refinement_loops, 1)
                loop_signal = self.refinement_gate * loop_fraction
                x = x + loop_signal.unsqueeze(0).unsqueeze(0)

                # Step 2: Re-run upper transformer layers
                for j, block in enumerate(upper_blocks):
                    layer_global = upper_start + j
                    x = block(
                        x, x0, cos, sin,
                        self.transformer.resid_lambdas[layer_global],
                        self.transformer.x0_lambdas[layer_global],
                    )

                # Step 3: Lifeline — anchor tokens, leave WM alone
                lifeline = self.refinement_gate.unsqueeze(0).unsqueeze(0)
                if M > 0:
                    x_tokens = x[:, :T, :] + lifeline * x_token_anchor
                    x = torch.cat([x_tokens, x[:, T:, :]], dim=1)
                else:
                    x = x + lifeline * x_token_anchor

                # Step 4: Retrieve-reason-retrieve
                if last_interface is not None and loop_i < loop_limit - 1:
                    if M > 0:
                        x_tokens = x[:, :T, :]
                        x_tokens, ref_cands, _, ref_reads, _, _, _ = \
                            last_interface(x_tokens, self.state, belief_cache=belief_cache)
                        x = torch.cat([x_tokens, x[:, T:, :]], dim=1)
                    else:
                        x, ref_cands, _, ref_reads, _, _, _ = \
                            last_interface(x, self.state, belief_cache=belief_cache)
                    all_candidates.extend(ref_cands)
                    all_read_indices.extend(ref_reads)

                # Step 5: TTT gradient step during refinement (training only)
                if targets is not None and self.ttt.should_update(0.0):
                    for layer_idx in self.ttt._ttt_layer_list:
                        if layer_idx >= upper_start:
                            pre = ttt_ctx.get_pre_delta(layer_idx)
                            if pre is not None:
                                pre_tokens = pre[:, :T, :] if M > 0 else pre
                                self.ttt.ttt_step(
                                    layer_idx=layer_idx,
                                    hidden_pre_delta=pre_tokens,
                                    targets=targets,
                                    lm_head_weight=self.transformer.lm_head.weight.data,
                                    vocab_size=self.config.transformer.vocab_size,
                                )

                # Step 6: HaltingHead probe — record P(halt) every loop.
                # Training: runs all oracle_loops (teacher forcing), records probs for loss.
                # Inference: exits when probe says halt.
                h_pooled = x[:, :T, :].mean(dim=1) if M > 0 else x.mean(dim=1)
                p_halt = self.refinement_probe(
                    h_pooled, loop_i, self.max_refinement_loops,
                )
                halt_probs.append(p_halt.mean().item())

                # Inference only: respect the probe's halt decision
                if not self.training:
                    if p_halt.mean().item() > self.config.transformer.refinement_halt_threshold:
                        break

        # ── Slice off working memory suffix before LM head ──
        if M > 0:
            x = x[:, :T, :]

        # TTT gradient step: update persistent deltas using NTP loss on this chunk.
        # This runs at BOTH training and inference — the model improves with every input.
        # Quality gate: RND surprise filters out-of-distribution inputs.
        # Rollback gate: loss increase after update → revert (inside ttt_step).
        if targets is not None:
            # Compute RND surprise on active beliefs as quality signal
            active_mask = self.state.get_active_mask()
            if active_mask.any():
                with torch.no_grad():
                    surprise = self.state.telos.compute_surprise(
                        self.state.beliefs.data[active_mask]
                    )
                    mean_surprise = surprise.mean().item()
                    self.ttt.update_surprise_ema(mean_surprise)
            else:
                mean_surprise = 0.0

            # Only update if input is in the learnable zone (not OOD, not fully predicted)
            if self.ttt.should_update(mean_surprise):
                for layer_idx in self.ttt._ttt_layer_list:
                    pre_delta = ttt_ctx.get_pre_delta(layer_idx)
                    if pre_delta is not None:
                        # Slice off working memory suffix — TTT targets are [B, T], not [B, T+M]
                        pre_delta_tokens = pre_delta[:, :T, :] if M > 0 else pre_delta
                        self.ttt.ttt_step(
                            layer_idx=layer_idx,
                            hidden_pre_delta=pre_delta_tokens,
                            targets=targets,
                            lm_head_weight=self.transformer.lm_head.weight.data,
                            vocab_size=self.config.transformer.vocab_size,
                        )

                # Update beliefs from write candidates (inference-time belief learning)
                self.ttt.ttt_step_beliefs(
                    self.state.beliefs,
                    all_candidates,
                    active_mask,
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

            # Refinement halt loss: teach the probe when to stop.
            # Teacher forcing with random oracle: during training, we sampled a random
            # oracle_loops from Uniform(1, max_loops). The BCE target is:
            #   target[i] = 0 for i < oracle_loops - 1  (keep going)
            #   target[oracle_loops - 1] = 1             (halt here)
            # This teaches the probe DIVERSE halt points, not just "always halt at max."
            # Reference: Mamba HaltingHead (phase14_inner_loop_bypass_trainer.py)
            loss_halt = torch.tensor(0.0, device=idx.device)
            if halt_probs and self.max_refinement_loops > 0:
                n_loops = len(halt_probs)
                halt_targets = torch.zeros(n_loops, device=idx.device)
                halt_targets[-1] = 1.0  # halt at the oracle loop count
                halt_preds = torch.tensor(halt_probs, device=idx.device)
                loss_halt = F.binary_cross_entropy(halt_preds, halt_targets)
            result['loss_halt'] = loss_halt
            result['refinement_loops'] = n_refinement_loops

            w_util = self.config.training.utility_loss_weight
            w_surp = self.config.training.surprise_loss_weight
            w_halt = self.config.training.halt_loss_weight

            result['loss'] = (
                result['loss_token']
                + alpha * loss_fe
                + alpha * w_util * loss_utility
                + alpha * w_surp * result['loss_surprise']
                + alpha * w_halt * loss_halt
            )
        else:
            logits = self.transformer.head(x)
            result['logits'] = logits
            result['refinement_loops'] = n_refinement_loops

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
        self.state.goal_status_logits.detach_()

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
