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
from .dflash_head import DFlashDraftHead


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
        self.table_size_int = table_size  # Python int for torch.compile (avoids .item() graph break)

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

        ts = self.table_size_int

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
        # Outputs raw logits — sigmoid applied externally for inference,
        # binary_cross_entropy_with_logits used for training (autocast-safe).
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_pooled: Tensor, loop_idx: int, max_loops: int) -> Tensor:
        """Compute P(halt) from pooled hidden state and loop position.

        Args:
            h_pooled: [B, hidden_dim] mean-pooled hidden state
            loop_idx: current loop index (0-based)
            max_loops: maximum number of loops

        Returns:
            [B, 1] halt logit (apply sigmoid for probability)
        """
        B = h_pooled.shape[0]
        pos = torch.full((B, 1), loop_idx / max(max_loops, 1),
                         device=h_pooled.device, dtype=h_pooled.dtype)
        return self.net(torch.cat([h_pooled, pos], dim=-1))


class RefinementRouter(nn.Module):
    """Per-position continue/halt decision for adaptive refinement loops.

    Determines which positions need further refinement based on the prediction
    error (delta between loop iterations). Positions with small, unimportant
    deltas are gated toward zero — later loops focus compute where it matters.

    Three complementary mechanisms:
    1. SCORE-style contractive scaling: dt = (1-rate)^l shrinks later loops
    2. Per-position routing: lightweight net on the delta → gate ∈ (0, 1)
    3. Error-gated retrieval: skip belief re-query for low-delta positions

    All thresholds are learned MetaParams — no hardcoded magic numbers.

    Reference: MoR (NeurIPS 2025, arXiv:2507.10524) — per-token adaptive recursion
    Reference: SCORE (arXiv:2603.10544) — contractive recurrent depth
    Reference: PonderNet (arXiv:2107.05407) — learned halting
    Reference: Two-Scale Latent Dynamics (NeurIPS 2025, arXiv:2509.23314) —
               later iterations produce smaller, increasingly orthogonal updates
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Lightweight projection: delta vector + loop fraction → scalar gate.
        # Bottleneck at hidden_dim//8 keeps this cheap (~2% of a block's FLOPs).
        # The router sees the CONTENT of the delta (not just magnitude), so it
        # can learn which kinds of changes matter (e.g., belief-retrieval-induced
        # vs noise-induced changes).
        bottleneck = max(32, hidden_dim // 8)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        # Init: first layer uses default Kaiming init (so the network can
        # differentiate inputs from step 1). Output layer zero-init with bias=1.0
        # so the gate starts at sigmoid(1) ≈ 0.73 (mostly open) regardless of input.
        # Gradients from the ponder loss propagate through the non-zero first layer
        # weights, breaking the symmetry and teaching the router which deltas matter.
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, 1.0)  # sigmoid(1) ≈ 0.73 — starts mostly open

    def forward(self, delta: Tensor, loop_fraction: float) -> Tensor:
        """Compute per-position continue gate from the loop delta.

        Args:
            delta: [B, T(+M), D] hidden state change from this loop iteration
            loop_fraction: current_loop / max_loops ∈ [0, 1]

        Returns:
            [B, T(+M), 1] gate ∈ (0, 1), higher = keep refining this position
        """
        B, S, D = delta.shape
        frac = delta.new_full((B, S, 1), loop_fraction)
        return torch.sigmoid(self.net(torch.cat([delta, frac], dim=-1)))


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
                parallel_goals=getattr(config.transformer, 'parallel_goals', True),
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
        blt_on = getattr(config.transformer, 'blt_enabled', False)
        self.engram_cache = EngramCache(
            # BLT mode: EngramCache operates on byte IDs at local_dim
            # Token mode: operates on token IDs at n_embd
            hidden_dim=config.transformer.blt_local_dim if blt_on else config.transformer.n_embd,
            vocab_size=config.transformer.blt_byte_vocab if blt_on else config.transformer.vocab_size,
            table_size=getattr(config.transformer, 'engram_table_size', 50000),
            n_heads=getattr(config.transformer, 'engram_n_heads', 4),
            embed_dim_per_head=getattr(config.transformer, 'engram_embed_dim', 0),
            tokenizer_name="" if blt_on else engram_tokenizer,  # no tokenizer compression for bytes
        )

        # ── Refinement Loops (Mamba-inspired iterative reasoning) ──
        # After the full forward pass, loop upper transformer blocks with the
        # belief-augmented representation. The model reasons over retrieved
        # beliefs before generating output.
        # Reference: phase14_inner_loop_bypass_trainer.py — dark loops
        self.max_refinement_loops = getattr(config.transformer, 'max_refinement_loops', 3)
        self.predictive_refinement = getattr(config.transformer, 'predictive_refinement', True)
        if self.max_refinement_loops > 0:
            # Refinement probe: decides when to stop iterating (Mamba's HaltingHead)
            self.refinement_probe = RefinementProbe(config.transformer.n_embd)
            # Lifeline gate: prevents hidden state drift during refinement loops
            # (Mamba's ROM re-injection, learned per-dimension scaling)
            self.refinement_gate = nn.Parameter(
                torch.full((config.transformer.n_embd,), config.transformer.refinement_gate_init)
            )
            # Per-position adaptive routing (MoR + SCORE + error-gated retrieval)
            if self.predictive_refinement:
                self.refinement_router = RefinementRouter(config.transformer.n_embd)

        # ── LSR Strategy Bank initialization ──
        # CognitiveState doesn't know n_embd (it's a TransformerConfig field),
        # so strategy bank is lazily initialized here where both are available.
        if self.max_refinement_loops > 0:
            self.state.init_strategy_bank(config.transformer.n_embd)

        # Kendall/Gal uncertainty weighting (CVPR 2018, arXiv:1705.07115)
        # Learns optimal per-loss-group weighting automatically.
        # Three groups: token loss, free energy loss, auxiliary losses
        self.log_sigma = nn.ParameterDict({
            'token': nn.Parameter(torch.tensor(0.0)),
            'fe': nn.Parameter(torch.tensor(0.0)),
            'aux': nn.Parameter(torch.tensor(0.0)),
        })

        # ── BLT Byte Latent Transformer (arXiv:2412.09871) ──
        # Replaces token embedding/LM head with byte encoder/decoder.
        # Global backbone operates on patches, local layers handle byte I/O.
        # Eliminates 117M LM head → 197K byte head. No tokenizer needed.
        self.blt_enabled = getattr(config.transformer, 'blt_enabled', False)
        if self.blt_enabled:
            from .blt import ByteEncoder, ByteDecoder
            tc = config.transformer
            self.byte_encoder = ByteEncoder(
                local_dim=tc.blt_local_dim,
                global_dim=tc.n_embd,
                patch_size=tc.blt_patch_size,
                n_layers=tc.blt_local_layers,
                byte_vocab=tc.blt_byte_vocab,
                head_dim=getattr(tc, 'blt_head_dim', 64),
            )
            self.byte_decoder = ByteDecoder(
                local_dim=tc.blt_local_dim,
                global_dim=tc.n_embd,
                patch_size=tc.blt_patch_size,
                n_layers=tc.blt_local_layers,
                byte_vocab=tc.blt_byte_vocab,
                n_byte_heads=getattr(tc, 'blt_n_byte_heads', 4),
                head_dim=getattr(tc, 'blt_head_dim', 64),
            )

        # ── DFlash Block Diffusion Draft Head (arXiv:2602.06036) ──
        # Small draft module for speculative decoding at inference time.
        # Cross-attends to target hidden states at selected layers + active
        # belief embeddings → predicts block_size tokens in parallel.
        # Trained jointly via auxiliary loss; shared LM head avoids 117M duplication.
        self.dflash_enabled = getattr(config.transformer, 'dflash_enabled', False)
        if self.dflash_enabled:
            tc = config.transformer
            self.dflash_head = DFlashDraftHead(
                hidden_dim=tc.n_embd,
                n_heads=tc.n_head,
                n_layers=tc.dflash_n_layers,
                block_size=tc.dflash_block_size,
                max_block_size=getattr(tc, 'dflash_max_block_size', tc.dflash_block_size),
                n_target_layers=tc.n_layer,
                # belief_dim is needed so DFlashDraftHead can project beliefs
                # from belief_dim → hidden_dim before using them as draft
                # cross-attention context. Without this, the draft layers'
                # norm_ctx (RMSNorm(hidden_dim)) crashes on belief-shaped
                # inputs.
                belief_dim=config.state.belief_dim,
            )

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights."""
        self.transformer.init_weights()
        # Interface layers use default PyTorch init (output projections already zero-init)

        # DSA: set up belief projections on Lightning Indexers now that belief_dim is known
        if self.config.transformer.dsa_enabled:
            belief_dim = self.config.state.belief_dim
            from .transformer import MLACausalSelfAttention
            for block in self.transformer.blocks:
                if isinstance(block.attn, MLACausalSelfAttention) and block.attn.dsa_enabled:
                    indexer = block.attn.indexer
                    if belief_dim > 0 and not indexer.has_beliefs:
                        indexer.belief_proj = nn.Linear(
                            belief_dim, indexer.index_dim, bias=False,
                        ).to(indexer.q_proj.weight.device)
                        indexer.has_beliefs = True
                        # Small init — belief conditioning starts subtle
                        nn.init.normal_(indexer.belief_proj.weight, mean=0.0, std=0.01)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        alpha: float = 0.0,
        update_state: bool = True,
        attn_mask: Tensor | None = None,
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
            update_state: if False, skip all in-place TTT/belief mutations so
                the forward is fully read-only. Default True preserves the
                "live self-improvement at both training and inference" property.
                Set to False for measurement passes (eval, belief-advantage
                probe) where mutation would contaminate the comparison.

        Returns:
            dict with scalars + candidates (no large tensors to avoid DDP OOM)
        """
        # Gate write-path state mutations on update_state. Touches every
        # recursive write inside the read/write paths (e.g.,
        # `state.touch_beliefs` called from read_path.py), not just the
        # explicit TTT/belief-update block gated later in this method.
        # Restored right before `return result`; we trust that `forward`
        # does not early-return on happy paths (verified: one return at
        # end). If forward raises, the flag is left at `update_state`
        # until the next forward restores it — acceptable because a
        # mid-forward crash already means state is in an undefined
        # transitional shape.
        _prev_updates_enabled = self.state._updates_enabled
        self.state._updates_enabled = update_state

        B, T = idx.size()
        M = self.working_memory_size if self.working_memory_size > 0 else 0

        # ── BLT: byte encoding → patches (replaces token embedding) ──
        _blt_byte_hidden = None  # decoder skip connection (only set when BLT active)
        _blt_T_bytes = T         # original byte count (for decoder)
        if self.blt_enabled:
            patches, _blt_byte_hidden = self.byte_encoder(idx)
            T = patches.shape[1]  # global backbone sees P patches, not T bytes
            x = patches  # already RMS-normed by ByteEncoder.out_norm
        else:
            x = self.transformer.wte(idx)
            x = F.rms_norm(x, (x.size(-1),))

        T_total = T + M

        self.transformer._ensure_rope(T_total)
        cos = self.transformer.cos[:, :T_total]
        sin = self.transformer.sin[:, :T_total]

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
        # Only applied to real token/byte positions (not working memory suffix).
        # BLT mode: EngramCache hashes byte N-grams at local_dim, injected into
        # byte_hidden (the encoder skip connection) so both encoder and decoder benefit.
        if self.blt_enabled:
            # Inject into byte encoder's hidden stream (local_dim space)
            engram_out = self.engram_cache(_blt_byte_hidden, idx)  # [B, T_bytes, local_dim]
            _blt_byte_hidden = _blt_byte_hidden + engram_out
        else:
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

        # DFlash: track which layers to tap for draft head conditioning
        dflash_tap_set = set(self.dflash_head.tap_indices) if self.dflash_enabled else set()
        dflash_tapped: dict[int, Tensor] = {}

        # DSA: extract active belief vectors for belief-conditioned attention
        # belief_cache.beliefs is already [N_active, D] (filtered by from_state)
        dsa_beliefs = None
        if self.config.transformer.dsa_enabled and belief_cache.n_active > 0:
            dsa_beliefs = belief_cache.beliefs

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
                    beliefs=dsa_beliefs,
                    attn_mask=attn_mask,
                )

            # DFlash: save hidden states at tapped layers
            # NOT detached — gradient from L_draft flows into backbone, providing
            # implicit multi-token prediction signal (the draft head predicts k
            # tokens ahead, and that gradient reaches these tapped layers).
            if i in dflash_tap_set:
                dflash_tapped[i] = x[:, :T, :] if M > 0 else x

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
        halt_logit_list = []
        ponder_costs = []  # per-loop mean gate values for ponder regularization
        n_refinement_loops = 0
        mean_gate_value = 1.0  # tracking for diagnostics
        retrieval_skips = 0  # count of error-gated retrieval skips
        strategy_step_weights = None  # accumulated strategy selection weights
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

            # Predictive refinement state: SCORE contraction rate + router
            use_pred_refine = self.predictive_refinement and self.max_refinement_loops > 1
            if use_pred_refine:
                contraction_rate = self.state.meta_params.refinement_contraction_rate
                retrieval_threshold = self.state.meta_params.refinement_retrieval_threshold

            # Training: teacher forcing with random oracle loop count.
            # Without this, the probe only ever sees "halt at max" targets and
            # never learns to halt early. Mamba samples from Uniform(min, max).
            # We sample the oracle loop count per step so the probe sees diverse
            # halt points: sometimes 1 loop is correct, sometimes 3.
            # Derived from input hash so all DDP ranks get the same oracle_loops
            # (ranks see the same data via broadcast).
            if self.training:
                oracle_loops = int(idx[0, 0].item() + idx[0, -1].item()) % self.max_refinement_loops + 1
            else:
                oracle_loops = self.max_refinement_loops  # inference: probe decides

            loop_limit = oracle_loops if self.training else self.max_refinement_loops

            # ── LSR Strategy Bank: setup for goal-conditioned perturbation ──
            # Compute the mean active goal embedding for strategy selection.
            # Reset per-step accumulator for fitness attribution in Pass 2.
            use_strategy = self.state._strategy_bank_initialized
            if use_strategy:
                strategy_step_weights = torch.zeros(
                    self.state.config.max_strategies,
                    device=x.device,
                )
                # Mean active goal embedding for the selector
                goal_indices, goal_embeds, _ = self.state.get_active_goals()
                if len(goal_indices) > 0:
                    goal_embed_mean = goal_embeds.mean(dim=0)  # [belief_dim]
                else:
                    goal_embed_mean = torch.zeros(
                        self.state.config.belief_dim, device=x.device,
                    )
                # Strategy perturbation scale: MetaParam * controller action
                strat_base_scale = self.state.meta_params.strategy_perturbation_scale

            for loop_i in range(loop_limit):
                n_refinement_loops += 1

                # Save pre-loop state for delta computation (predictive refinement)
                x_pre_loop = x if not use_pred_refine else x.clone()

                # Step 1: Loop-index encoding via goal-conditioned strategy
                # perturbation (LSR Strategy Bank). Replaces the old single-
                # direction loop_signal with diverse, learned perturbation
                # vectors selected based on (hidden_state, goal, iteration).
                #
                # Backward compatible: if strategy bank is not initialized
                # (max_refinement_loops was 0 at config time), falls back to
                # the original refinement_gate * loop_fraction signal.
                #
                # Reference: LSR (dl1683/Latent-Space-Reasoning) — trajectory
                #   perturbation via orthogonal soft prompt injection
                loop_fraction = loop_i / max(self.max_refinement_loops, 1)

                if use_strategy and self.state.strategy_selector is not None:
                    # Goal-conditioned strategy selection (Tier 2)
                    h_pooled = x[:, :T, :].mean(dim=1) if M > 0 else x.mean(dim=1)
                    perturbation = self.state.strategy_selector(
                        h_pooled, goal_embed_mean, loop_fraction,
                        self.state.strategy_bank,
                    )  # [B, n_embd]

                    # Scale by MetaParam (learned) — controller modulates further
                    loop_signal = perturbation * strat_base_scale
                    x = x + loop_signal.unsqueeze(1)  # [B, 1, n_embd] broadcast over T

                    # Accumulate selection weights for fitness attribution.
                    # The selector uses entmax15, so weights are sparse — only
                    # the 1-2 selected strategies accumulate credit.
                    with torch.no_grad():
                        sel_logits = self.state.strategy_selector.proj(
                            torch.cat([
                                h_pooled.mean(dim=0, keepdim=True),
                                goal_embed_mean.unsqueeze(0),
                                torch.tensor([[loop_fraction]], device=x.device),
                            ], dim=-1),
                        )
                        try:
                            from entmax import entmax15
                            sel_weights = entmax15(sel_logits, dim=-1).squeeze(0)
                        except ImportError:
                            sel_weights = torch.softmax(sel_logits, dim=-1).squeeze(0)
                        strategy_step_weights += sel_weights
                else:
                    # Fallback: original refinement_gate * loop_fraction
                    loop_signal = self.refinement_gate * loop_fraction
                    x = x + loop_signal.unsqueeze(0).unsqueeze(0)

                # Step 2: Re-run upper transformer layers
                # Gradient checkpointing is REQUIRED here: the halt_logit_list
                # keeps tensors in the graph across loop iterations. Without
                # checkpointing, compiled blocks' saved tensors from iteration N
                # get version-clobbered by iteration N+1, causing RuntimeError
                # during backward ("modified by an inplace operation").
                for j, block in enumerate(upper_blocks):
                    layer_global = upper_start + j
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, x0, cos, sin,
                        self.transformer.resid_lambdas[layer_global],
                        self.transformer.x0_lambdas[layer_global],
                        use_reentrant=False,
                        beliefs=dsa_beliefs,
                    )

                # Step 3: Lifeline — anchor tokens, leave WM alone
                lifeline = self.refinement_gate.unsqueeze(0).unsqueeze(0)
                if M > 0:
                    x_tokens = x[:, :T, :] + lifeline * x_token_anchor
                    x = torch.cat([x_tokens, x[:, T:, :]], dim=1)
                else:
                    x = x + lifeline * x_token_anchor

                # ── Predictive refinement: contractive scaling + per-position routing ──
                # After the block computation + lifeline, compute the delta from the
                # pre-loop state and gate it. Loop 0 runs at full strength (no gating).
                # Later loops: SCORE-style contraction shrinks the global step size,
                # and the MoR-style router further gates per-position.
                #
                # This replaces nothing — it modulates. The blocks, lifeline, TTT, and
                # halt probe all function identically. The router just controls how much
                # of each loop's delta survives into the next iteration.
                if use_pred_refine and loop_i > 0:
                    delta = x - x_pre_loop  # [B, S, D] — what this loop changed

                    # SCORE-style contractive scaling: dt = (1 - rate)^l
                    # Later loops contribute geometrically smaller corrections.
                    # Two-Scale Latent Dynamics (arXiv:2509.23314) empirically confirms
                    # this: later iterations produce smaller, orthogonal updates.
                    dt = (1.0 - contraction_rate) ** loop_i  # scalar, differentiable

                    # MoR-style per-position routing: the router sees the delta content
                    # (not just magnitude) so it can learn which changes matter.
                    position_gate = self.refinement_router(delta, loop_fraction)  # [B, S, 1]

                    # Apply: x = x_pre + dt * gate * delta
                    x = x_pre_loop + dt * position_gate * delta

                    # Track for ponder loss + diagnostics
                    ponder_costs.append(position_gate.mean())
                    mean_gate_value = position_gate.mean().item()

                    # Compute per-position delta norms for error-gated retrieval
                    # Normalized by hidden dim so threshold is scale-invariant
                    delta_norms = (dt * position_gate * delta).norm(dim=-1)  # [B, S]
                    hidden_scale = x.norm(dim=-1).mean().detach().clamp(min=1e-6)
                    relative_delta = delta_norms / hidden_scale  # fraction of hidden norm
                else:
                    relative_delta = None

                # Step 4: Retrieve-reason-retrieve
                # With predictive refinement: skip re-query for positions where the
                # representation barely changed (error-gated retrieval). The threshold
                # is a learned MetaParam, not hardcoded.
                # Reference: DeltaLLM (arXiv:2507.19608) — temporal sparsity via delta
                if last_interface is not None and loop_i < loop_limit - 1:
                    # Error-gated retrieval: check if enough positions changed to
                    # justify the cost of belief re-query
                    should_requery = True
                    if use_pred_refine and relative_delta is not None:
                        # Fraction of positions with meaningful delta
                        active_frac = (relative_delta > retrieval_threshold).float().mean()
                        # Skip if fewer than 10% of positions changed meaningfully
                        # (the 10% is derived from retrieval_threshold, not hardcoded —
                        # when threshold is high, fewer positions pass, so skip happens
                        # more often. The system learns the right balance.)
                        should_requery = active_frac.item() > retrieval_threshold.item()
                        if not should_requery:
                            retrieval_skips += 1

                    if should_requery:
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

                # Step 5: TTT gradient step during refinement (training only).
                # Gated on update_state so measurement passes (eval perplexity,
                # belief-advantage probe) stay read-only. Without this gate the
                # refinement loop would keep mutating deltas during eval —
                # note that should_update() returns True on cold start
                # (ttt.py: `if mean < 1e-8: return True`), so untrained
                # configs with refinement enabled had TTT firing on every
                # eval step before this fix.
                if targets is not None and update_state and self.ttt.should_update(0.0):
                    # BLT: use patch-level targets and composed projection
                    # (global_dim → local_dim → byte_vocab via down_proj @ byte_head)
                    if self.blt_enabled:
                        ref_lm_weight = (self.byte_decoder.byte_heads[0].weight.data
                                         @ self.byte_decoder.down_proj.weight.data)  # [260, 768]
                        ref_vocab = self.config.transformer.blt_byte_vocab
                        ps = self.config.transformer.blt_patch_size
                        pad_len = (ps - _blt_T_bytes % ps) % ps
                        tgt_padded = F.pad(targets, (0, pad_len), value=-1) if pad_len > 0 else targets
                        ref_targets = tgt_padded.view(B, -1, ps)[:, :T, -1]
                    else:
                        ref_lm_weight = self.transformer.lm_head.weight.data
                        ref_vocab = self.config.transformer.vocab_size
                        ref_targets = targets
                    for layer_idx in self.ttt._ttt_layer_list:
                        if layer_idx >= upper_start:
                            pre = ttt_ctx.get_pre_delta(layer_idx)
                            if pre is not None:
                                pre_tokens = pre[:, :T, :] if M > 0 else pre
                                self.ttt.ttt_step(
                                    layer_idx=layer_idx,
                                    hidden_pre_delta=pre_tokens,
                                    targets=ref_targets,
                                    lm_head_weight=ref_lm_weight,
                                    vocab_size=ref_vocab,
                                )

                # Step 6: HaltingHead probe — record P(halt) every loop.
                # Training: runs all oracle_loops (teacher forcing), records probs for loss.
                # Inference: exits when probe says halt.
                h_pooled = x[:, :T, :].mean(dim=1) if M > 0 else x.mean(dim=1)
                halt_logit = self.refinement_probe(
                    h_pooled, loop_i, self.max_refinement_loops,
                )
                halt_logit_list.append(halt_logit.mean())  # keep tensor for gradient flow

                # Inference only: respect the probe's halt decision
                if not self.training:
                    if halt_logit.mean().sigmoid().item() > self.config.transformer.refinement_halt_threshold:
                        break

        # ── Slice off working memory suffix before LM head ──
        if M > 0:
            x = x[:, :T, :]

        # TTT gradient step: update persistent deltas using NTP loss on this chunk.
        # This runs at BOTH training and inference — the model improves with every input.
        # Quality gate: RND surprise filters out-of-distribution inputs.
        # Rollback gate: loss increase after update → revert (inside ttt_step).
        # With BLT: hidden states are patch-level but targets are byte-level.
        # Create patch-level targets: last byte per patch (the byte the patch
        # hidden state has accumulated). TTT learns at patch granularity.
        # Gated on update_state so measurement passes (eval, belief-advantage
        # probe) stay read-only and don't contaminate the state being measured.
        if targets is not None and update_state:
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
                # TTT uses LM head weight for next-token loss. With BLT,
                # TTT operates at patch level using the decoder's primary head.
                # Patch-level targets: last byte per patch (the byte accumulated
                # by the strided pooling at each patch position).
                if self.blt_enabled:
                    # Compose: global_dim → local_dim → byte_vocab
                    ttt_lm_weight = (self.byte_decoder.byte_heads[0].weight.data
                                     @ self.byte_decoder.down_proj.weight.data)  # [260, 768]
                    ttt_vocab = self.config.transformer.blt_byte_vocab
                    ps = self.config.transformer.blt_patch_size
                    T_b = _blt_T_bytes
                    P = T  # current T is already patch count
                    # Downsample byte targets to patch: take last byte per patch
                    pad_len = (ps - T_b % ps) % ps
                    tgt_padded = F.pad(targets, (0, pad_len), value=-1) if pad_len > 0 else targets
                    ttt_targets = tgt_padded.view(B, -1, ps)[:, :P, -1]  # [B, P]
                else:
                    ttt_lm_weight = self.transformer.lm_head.weight.data
                    ttt_vocab = self.config.transformer.vocab_size
                    ttt_targets = targets
                for layer_idx in self.ttt._ttt_layer_list:
                    pre_delta = ttt_ctx.get_pre_delta(layer_idx)
                    if pre_delta is not None:
                        # Slice off working memory suffix — TTT targets are [B, T], not [B, T+M]
                        pre_delta_tokens = pre_delta[:, :T, :] if M > 0 else pre_delta
                        self.ttt.ttt_step(
                            layer_idx=layer_idx,
                            hidden_pre_delta=pre_delta_tokens,
                            targets=ttt_targets,
                            lm_head_weight=ttt_lm_weight,
                            vocab_size=ttt_vocab,
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
            if self.blt_enabled:
                # BLT: byte-level loss via decoder (260 classes, not 151K)
                # x is [B, P, n_embd] patches, byte_hidden is [B, T_bytes, local_dim]
                loss_byte, byte_stats = self.byte_decoder.compute_loss(
                    x, _blt_byte_hidden, targets,
                )
                result['loss_token'] = loss_byte
                result['byte_stats'] = byte_stats
            else:
                # Fused head + loss: never materializes [B*T, 151K] logits tensor.
                # At 128K context this saves ~38 GB by computing in 2K-position chunks.
                from ..core.losses import fused_chunked_cross_entropy
                result['loss_token'] = fused_chunked_cross_entropy(
                    x.reshape(-1, x.size(-1)),
                    targets.reshape(-1),
                    self.transformer.lm_head.weight,
                )

            # Always compute L_fe and L_utility so all interface parameters participate
            # in the computation graph (required by DDP). When α=0, these contribute
            # zero to the loss but keep parameters in the graph for gradient reduction.

            # Utility loss
            loss_utility = torch.tensor(0.0, device=idx.device)
            if all_utility_logits:
                if alpha > 0 and not self.blt_enabled:
                    # Full utility: fused head + loss (same chunking as main loss)
                    from ..core.losses import fused_chunked_cross_entropy
                    for util_hidden in all_utility_logits:
                        loss_utility = loss_utility + fused_chunked_cross_entropy(
                            util_hidden.reshape(-1, util_hidden.size(-1)),
                            targets.reshape(-1),
                            self.transformer.lm_head.weight,
                        )
                    loss_utility = loss_utility / len(all_utility_logits)
                else:
                    # Cheap graph participation: keep utility_head in the computation
                    # graph for DDP without materializing the huge vocab logit tensor
                    # (Also used for BLT — utility runs at patch level, not byte level)
                    for util_hidden in all_utility_logits:
                        loss_utility = loss_utility + util_hidden.sum() * 0.0
            result['loss_utility'] = loss_utility

            # Interface proxy: trains read/write paths (prediction error + attention entropy)
            loss_fe_proxy = compute_differentiable_free_energy(
                all_attn_weights, all_retrieved, all_obs_vectors,
                self.config.state.belief_dim,
                fe_lambda=self.state.meta_params.fe_lambda,
                huber_delta=self.state.meta_params.huber_delta,
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
            if halt_logit_list and self.max_refinement_loops > 0:
                n_loops = len(halt_logit_list)
                halt_targets = torch.zeros(n_loops, device=idx.device)
                halt_targets[-1] = 1.0  # halt at the oracle loop count
                halt_logits = torch.stack(halt_logit_list)  # [n_loops] — tensors with grad
                loss_halt = F.binary_cross_entropy_with_logits(halt_logits.float(), halt_targets.float())
            result['loss_halt'] = loss_halt
            result['refinement_loops'] = n_refinement_loops

            # Ponder loss: per-position penalty for continuing refinement.
            # Encourages the router to gate positions toward zero (halt early)
            # when their representation has converged. The ponder cost is a learned
            # MetaParam — the system discovers the right compute/quality tradeoff.
            # Reference: PonderNet (arXiv:2107.05407) — regularized halting
            loss_ponder = torch.tensor(0.0, device=idx.device)
            if ponder_costs:
                ponder_weight = self.state.meta_params.refinement_ponder_cost
                # Mean gate value across loops — higher = more positions kept refining
                loss_ponder = ponder_weight * torch.stack(ponder_costs).mean()
            result['loss_ponder'] = loss_ponder
            result['refinement_mean_gate'] = mean_gate_value
            result['refinement_retrieval_skips'] = retrieval_skips

            # Store strategy selection weights for Pass 2 fitness attribution
            if strategy_step_weights is not None:
                with torch.no_grad():
                    self.state.strategy_step_weights.copy_(strategy_step_weights)
            result['strategy_step_weights'] = strategy_step_weights

            w_util = self.config.training.utility_loss_weight
            w_surp = self.config.training.surprise_loss_weight
            w_halt = self.config.training.halt_loss_weight

            # DEQ Jacobian regularization: ensures the message passing fixed-point
            # map stays contractive as relation_transform is trained by Bethe FE.
            # Only computed periodically (every 10 steps) to minimize overhead.
            jac_loss = torch.tensor(0.0, device=idx.device)
            if (alpha > 0 and self.training
                    and hasattr(self.state, 'message_passing')
                    and hasattr(self.state.message_passing, '_deq')
                    and self.state.message_passing._deq is not None
                    and self.state.edge_active.any()):
                mp_result = self.state.message_passing(self.state)
                jac_loss = mp_result.get('jac_loss', torch.tensor(0.0, device=idx.device))
            result['loss_jac'] = jac_loss
            result['deq_solver_steps'] = (
                self.state.message_passing._last_info.get('nstep', 0)
                if hasattr(self.state, 'message_passing') else 0
            )

            # ── DFlash draft loss (arXiv:2602.06036 + DMax arXiv:2604.08302) ──
            # Train the draft head to predict next-block tokens conditioned on
            # tapped hidden states + beliefs. Includes OPUT on-policy self-correction:
            # the draft head trains on its own (possibly wrong) predictions via SPD
            # hybrid embeddings. NOT alpha-gated — trains during phase 1.
            loss_draft = torch.tensor(0.0, device=idx.device)
            loss_self_correct = torch.tensor(0.0, device=idx.device)
            if self.dflash_enabled and dflash_tapped and self.training:
                block_size = self.config.transformer.dflash_block_size
                # Pick a starting position derived from input (DDP-safe: same data on all ranks)
                if T > block_size + 1:
                    t_start = int(idx[0, T // 2].item()) % (T - block_size)
                    # Target: the next block_size tokens after t_start
                    draft_targets = targets[:, t_start + 1: t_start + 1 + block_size]
                    if draft_targets.shape[1] == block_size:
                        # Extract features: tapped hidden states at position t_start
                        # Use a window of context (last 64 positions up to t_start)
                        ctx_start = max(0, t_start - 63)
                        ctx_hiddens = {
                            k: v[:, ctx_start:t_start + 1, :]
                            for k, v in dflash_tapped.items()
                        }
                        # Get active beliefs for extra context. BeliefCache
                        # stores the already-filtered active belief cartesians
                        # under `.beliefs` (see read_path.BeliefCache.__slots__);
                        # there is no `.active_beliefs` attribute — other call
                        # sites use `.beliefs` directly (line 607, read_path.py:315).
                        active_beliefs = None
                        if belief_cache.n_active > 0:
                            active_beliefs = belief_cache.beliefs

                        context, injection = self.dflash_head.extract_features(
                            ctx_hiddens, active_beliefs,
                        )
                        # Streak distillation: learned MetaParams control
                        # position weighting and expected streak bonus
                        streak_decay = self.state.meta_params.dflash_streak_decay
                        streak_weight = self.state.meta_params.dflash_streak_weight
                        oput_weight = self.state.meta_params.oput_self_correct_weight
                        # BLT: DFlash predicts at patch level using byte decoder head
                        dflash_head_fn = self.transformer.head
                        dflash_embed_fn = self.transformer.wte
                        if self.blt_enabled:
                            # down_proj: n_embd→local_dim, then byte_head: local_dim→byte_vocab
                            _dp = self.byte_decoder.down_proj
                            _bh = self.byte_decoder.byte_heads[0]
                            dflash_head_fn = lambda h: _bh(
                                F.rms_norm(_dp(h), (_dp.out_features,))
                            )
                            # Compose byte_embed (→local_dim) + up_proj (→n_embd)
                            # so OPUT hybrid embeddings match the draft head dimension
                            _be = self.byte_encoder.byte_embed
                            _up = self.byte_encoder.up_proj
                            dflash_embed_fn = lambda ids: _up(_be(ids))
                        # DDTree-aware training: pass tree MetaParams when DDTree is enabled.
                        # These add three terms to the draft loss:
                        # 1. Position blend: decay↔uniform interpolation
                        # 2. Tree prefix mass bonus: uniform cumprod of P(correct)
                        # 3. Top-K recall: penalty when target outside draft's top-K
                        ddtree_on = getattr(self.config.transformer, 'ddtree_enabled', False)
                        ddtree_kwargs = {}
                        if ddtree_on:
                            ddtree_kwargs = dict(
                                ddtree_position_blend=self.state.meta_params.ddtree_position_blend,
                                ddtree_prefix_weight=self.state.meta_params.ddtree_prefix_weight,
                                ddtree_recall_weight=self.state.meta_params.ddtree_recall_weight,
                                ddtree_train_budget=getattr(
                                    self.config.transformer, 'ddtree_train_budget', 64,
                                ),
                            )
                        loss_draft, loss_self_correct = self.dflash_head.compute_draft_loss(
                            context, injection, draft_targets,
                            lm_head_fn=dflash_head_fn,
                            embed_fn=dflash_embed_fn,
                            streak_decay=streak_decay,
                            streak_weight=streak_weight,
                            oput_weight=oput_weight,
                            **ddtree_kwargs,
                        )
            result['loss_draft'] = loss_draft
            result['loss_self_correct'] = loss_self_correct
            # Store tap hiddens for spec_generate reuse (avoids redundant _collect_tap_hiddens)
            if self.dflash_enabled and dflash_tapped:
                result['dflash_tapped'] = dflash_tapped

            # DSA KL alignment loss: collect from all MLA layers with indexers
            loss_dsa_kl = torch.tensor(0.0, device=idx.device)
            if self.config.transformer.dsa_enabled:
                from .transformer import MLACausalSelfAttention
                n_dsa = 0
                for block in self.transformer.blocks:
                    if isinstance(block.attn, MLACausalSelfAttention) and block.attn.dsa_enabled:
                        kl = block.attn._last_dsa_kl_loss
                        if kl is not None:
                            loss_dsa_kl = loss_dsa_kl + kl
                            n_dsa += 1
                            block.attn._last_dsa_kl_loss = None  # consumed
                if n_dsa > 0:
                    loss_dsa_kl = loss_dsa_kl / n_dsa
            result['loss_dsa_kl'] = loss_dsa_kl

            # Kendall/Gal uncertainty weighting (CVPR 2018)
            def _uw(loss_term: Tensor, log_s: Tensor) -> Tensor:
                """Uncertainty-weighted loss: L/(2*exp(2s)) + s"""
                return loss_term / (2.0 * torch.exp(2.0 * log_s)) + log_s

            L_token_w = _uw(result['loss_token'], self.log_sigma['token'])

            L_fe_w = _uw(loss_fe, self.log_sigma['fe'])

            w_draft = self.config.transformer.dflash_loss_weight if self.dflash_enabled else 0.0
            L_aux = (
                w_util * loss_utility
                + w_surp * result['loss_surprise']
                + w_halt * loss_halt
                + loss_ponder
                + 0.01 * jac_loss
            )
            L_aux_w = _uw(L_aux, self.log_sigma['aux'])

            # DSA KL weight: full weight during phase 1 (indexer alignment while
            # dense attention is cheap), reduced weight after (maintenance).
            dsa_kl_w = 0.0
            if self.config.transformer.dsa_enabled and loss_dsa_kl.item() > 0:
                tc = self.config.training
                dsa_kl_w = tc.dsa_kl_weight if alpha == 0.0 else tc.dsa_kl_weight_after

            # Alpha gating: phase 1 = token only, phase 2-3 = all terms.
            # Draft + DSA KL are NOT alpha-gated — they train from step 0.
            # Self-correction loss is NOT alpha-gated — it trains from step 0.
            result['loss'] = (L_token_w + alpha * L_fe_w + alpha * L_aux_w
                              + w_draft * loss_draft
                              + w_draft * loss_self_correct
                              + dsa_kl_w * loss_dsa_kl)

            result['log_sigma_token'] = self.log_sigma['token'].item()
            result['log_sigma_fe'] = self.log_sigma['fe'].item()
            result['log_sigma_aux'] = self.log_sigma['aux'].item()
        else:
            if self.blt_enabled:
                # BLT: decode patches → byte logits (260 classes)
                byte_logits, _ = self.byte_decoder(x, _blt_byte_hidden)
                result['logits'] = byte_logits
            else:
                logits = self.transformer.head(x)
                result['logits'] = logits
            result['refinement_loops'] = n_refinement_loops
            # Store tap hiddens for spec_generate reuse (inference path)
            if self.dflash_enabled and dflash_tapped:
                result['dflash_tapped'] = dflash_tapped

        # Restore write gate (see docstring at top of forward).
        self.state._updates_enabled = _prev_updates_enabled
        return result

    def compute_loss(
        self,
        idx: Tensor,
        targets: Tensor,
        alpha: float = 0.0,
        update_state: bool = False,
    ) -> dict:
        """Compute combined loss: L_token + α * L_fe.

        Convenience wrapper around forward() for non-DDP usage and tests.

        **Defaults to update_state=False** — safe for the typical
        "result = model.compute_loss(...); result['loss'].backward()" pattern.
        If you pass update_state=True, the forward pass will mutate TTT deltas
        and state.beliefs in place, and you MUST wrap the call + backward in
        `torch.autograd.graph.allow_mutation_on_saved_tensors()`, otherwise
        backward will fail with "one of the variables needed for gradient
        computation has been modified by an inplace operation".

        The production training loop does NOT use compute_loss — it calls
        model() directly inside the mutation context (see
        training/train.py). This wrapper exists for tests and ad-hoc
        non-DDP usage; keeping its default read-only prevents the subtle
        autograd-version trap from reappearing in new test code.
        """
        return self.forward(idx, targets, alpha=alpha, update_state=update_state)

    @torch.no_grad()
    def incremental_forward(
        self,
        new_token_ids: Tensor,
        incremental_state: 'IncrementalState',
        position_offset: int,
        commit: bool = False,
        attn_mask: Tensor | None = None,
    ) -> dict:
        """Process only new tokens using cached state from previous rounds.

        This is the incremental counterpart to forward(). Instead of processing
        the full sequence, it:
        1. Embeds only new tokens + applies engram
        2. Runs new tokens through transformer blocks with cached state:
           - H/D layers: use saved recurrent state as initial_state
           - L layers: attend new queries to cached KV + new KV
        3. Applies LM head → logits for new tokens only

        Skips: state interfaces, refinement loops, working memory prefix.
        These are training-time features not needed for verification.
        The logits produced are sufficient for DDTree tree walking.

        Args:
            new_token_ids: [1, T_new] token IDs for new positions only.
            incremental_state: IncrementalState with per-layer caches.
            position_offset: absolute position of first new token (= seq_len so far).
            commit: if True, update persistent caches. If False (speculative
                verify), process without persisting state changes.
            attn_mask: optional attention mask for tree verification.
                [1, 1, T_new, T_past + T_new] if provided.

        Returns:
            dict with 'logits': [1, T_new, vocab] and 'dflash_tapped': tap hiddens.
        """
        from .kv_cache import IncrementalState

        B, T_new = new_token_ids.shape
        assert B == 1, "incremental_forward only supports batch_size=1"

        device = new_token_ids.device

        # ── Embed new tokens ──
        x = self.transformer.wte(new_token_ids)
        x = F.rms_norm(x, (x.size(-1),))

        # Engram: only on new tokens (hash-based lookup, position-independent)
        engram_out = self.engram_cache(x, new_token_ids)
        x = x + engram_out

        # x0 for residual scaling (same as forward)
        x0 = x

        # RoPE: generate cos/sin for the NEW token positions
        T_total = position_offset + T_new
        self.transformer._ensure_rope(T_total)
        cos = self.transformer.cos[:, position_offset:T_total]
        sin = self.transformer.sin[:, position_offset:T_total]

        # ── Process through transformer blocks with cached state ──
        dflash_tapped = {}
        tap_set = set(self.dflash_head.tap_indices) if self.dflash_enabled else set()

        for i, block in enumerate(self.transformer.blocks):
            layer_cache = incremental_state.layer_caches.get(i)
            x = block(
                x, x0, cos, sin,
                self.transformer.resid_lambdas[i],
                self.transformer.x0_lambdas[i],
                layer_cache=layer_cache,
                commit=commit,
                attn_mask=attn_mask,
            )
            # Collect tap hiddens for DFlash draft head
            if i in tap_set:
                dflash_tapped[i] = x.clone()

        # ── LM head ──
        x = F.rms_norm(x, (x.size(-1),))
        if self.blt_enabled:
            logits = self.byte_decoder(x)
        else:
            logits = self.transformer.head(x)

        if commit:
            incremental_state.seq_len = T_total

        return {
            'logits': logits,
            'dflash_tapped': dflash_tapped if dflash_tapped else None,
        }

    @torch.no_grad()
    def spec_generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
    ) -> Tensor:
        """Generate tokens using DFlash speculative decoding with DDTree verification.

        DDTree mode (ddtree_enabled=True): builds a draft tree from the per-position
        marginals of a single DFlash forward pass and verifies the entire tree in
        one target-model forward pass using ancestor-only tree attention. This
        explores multiple continuations per round instead of just one, increasing
        the expected acceptance length by ~1.5-2.1× over single-trajectory DFlash.

        DFlash mode (ddtree_enabled=False): falls back to single-trajectory
        verification with adaptive block size (original behavior).

        Both modes preserve the target model's output distribution (lossless).

        Optimized draft-verify loop: the verify step from round N serves as
        the prefill for round N+1. This eliminates one full model forward per
        round, doubling effective throughput.

        Improvements:
        1. KV injection: target features injected into draft layer K/V projections
        2. DDTree: multi-branch tree verification (when enabled)
        3. Adaptive block size: entropy-based cutoff (DFlash fallback)
        4. Verify-as-prefill reuse: no redundant forward pass per round
        5. Tap hidden caching: forward() stores tap hiddens in result dict

        Falls back to standard autoregressive if DFlash is not enabled.

        Reference: DDTree (Ringel & Romano — liranringel.github.io/ddtree/)
        Reference: DFlash (arXiv:2602.06036) — block diffusion drafting

        Args:
            input_ids: [1, T] prompt token ids
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            stop_token_ids: stop generation when any of these are produced

        Returns:
            [1, T + N] generated sequence
        """
        assert input_ids.shape[0] == 1, "spec_generate only supports batch_size=1"
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        stop_set = set(stop_token_ids) if stop_token_ids else set()

        if not self.dflash_enabled:
            return self._generate_autoregressive(
                generated, max_new_tokens, temperature, stop_set,
            )

        # Dispatch: DDTree (tree verify) or DFlash (single-trajectory verify)
        ddtree_enabled = getattr(self.config.transformer, 'ddtree_enabled', False)
        if ddtree_enabled:
            result = self._spec_generate_ddtree(
                generated, max_new_tokens, temperature, stop_set,
            )
        else:
            result = self._spec_generate_dflash(
                generated, max_new_tokens, temperature, stop_set,
            )

        self.train()
        return result

    def _spec_generate_dflash(
        self,
        generated: Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_set: set[int],
    ) -> Tensor:
        """Single-trajectory DFlash speculative decoding (original behavior)."""
        device = generated.device
        max_block = self.dflash_head.max_block_size
        entropy_threshold = self.state.meta_params.dflash_entropy_threshold.item()
        tokens_generated = 0

        # ── Initial prefill (only once) ──
        result = self.forward(generated)
        cached_logits = result['logits']
        cached_taps = result.get('dflash_tapped')

        while tokens_generated < max_new_tokens:
            next_token = self._sample_token(cached_logits[:, -1, :], temperature)

            if next_token.item() in stop_set:
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                break

            if cached_taps is not None:
                tap_hiddens = cached_taps
            else:
                tap_hiddens = self._collect_tap_hiddens(generated)

            ctx_start = max(0, generated.shape[1] - 64)
            ctx_hiddens = {k: v[:, ctx_start:, :] for k, v in tap_hiddens.items()}

            active_mask = self.state.get_active_mask()
            active_beliefs = self.state.beliefs.data[active_mask] if active_mask.any() else None

            context, injection = self.dflash_head.extract_features(
                ctx_hiddens, active_beliefs,
            )

            remaining = max_new_tokens - tokens_generated - 1
            draft_length = min(max_block, max(remaining, 1))
            spec_head_fn = self._get_spec_head_fn()
            draft_logits = self.dflash_head(
                context, lm_head_fn=spec_head_fn,
                injection=injection, draft_length=draft_length,
            )

            entropy = self.dflash_head.compute_entropy(draft_logits)
            above_threshold = (entropy.squeeze(0) > entropy_threshold)
            if above_threshold.any():
                adaptive_size = above_threshold.nonzero(as_tuple=True)[0][0].item()
                adaptive_size = max(adaptive_size, 1)
            else:
                adaptive_size = draft_length

            draft_tokens = self._sample_token(
                draft_logits[:, :adaptive_size].squeeze(0), temperature,
            )

            candidate_tokens = torch.cat([next_token, draft_tokens], dim=0)
            candidate_seq = torch.cat([
                generated, candidate_tokens.unsqueeze(0),
            ], dim=1)

            verify_result = self.forward(candidate_seq)
            verify_logits = verify_result['logits']

            T_orig = generated.shape[1]
            n_accepted = 1

            for j in range(adaptive_size):
                verify_token = self._sample_token(
                    verify_logits[:, T_orig + j, :], temperature,
                )
                if verify_token.item() == candidate_tokens[1 + j].item():
                    n_accepted += 1
                else:
                    candidate_tokens[1 + j] = verify_token.squeeze()
                    n_accepted += 1
                    break

            accepted = candidate_tokens[:n_accepted]
            generated = torch.cat([generated, accepted.unsqueeze(0)], dim=1)
            tokens_generated += n_accepted

            cached_logits = verify_logits
            cached_taps = verify_result.get('dflash_tapped')

            if stop_set and any(t.item() in stop_set for t in accepted):
                break

        return generated

    def _spec_generate_ddtree(
        self,
        generated: Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_set: set[int],
    ) -> Tensor:
        """DDTree speculative decoding with incremental KV cache.

        Three-phase loop per round:
        1. Draft: DFlash head produces per-position marginals from cached taps
        2. Build tree + verify (speculative): incremental forward on tree tokens
           with checkpointed recurrent state and tree attention mask
        3. Walk + commit: determine accepted path, rollback recurrent state,
           replay only accepted tokens to update persistent caches

        The incremental KV cache eliminates full-sequence reprocessing:
        - H/D layers: save/restore Fenwick tree recurrent state
        - L layers: cache assembled K/V, attend new queries to cached + new
        - Per-round cost: O(B + accepted) instead of O(T_past + B)

        Reference: DDTree (Ringel & Romano — liranringel.github.io/ddtree/)
        """
        from .ddtree import build_ddtree, compile_ddtree, follow_verified_tree
        from .kv_cache import IncrementalState

        device = generated.device
        max_block = self.dflash_head.max_block_size

        # Compute effective tree budget from config × learned scale
        default_budget = getattr(self.config.transformer, 'ddtree_default_budget', 256)
        budget_scale = self.state.meta_params.ddtree_budget_scale.item()
        tree_budget = max(1, round(default_budget * budget_scale))

        tokens_generated = 0

        # ── Phase 0: Prefill — full forward on prompt, build initial caches ──
        # This runs once. All subsequent rounds use incremental_forward.
        inc_state = IncrementalState.build_for_model(self)
        prefill_result = self.forward(generated)
        cached_logits = prefill_result['logits']
        cached_taps = prefill_result.get('dflash_tapped')

        # Populate caches from prefill: extract KV and recurrent states
        # For the first round, we need the caches populated. Since forward()
        # doesn't directly expose per-layer state, we run a commit pass on
        # the prompt tokens to seed the caches.
        _commit_result = self.incremental_forward(
            generated, inc_state,
            position_offset=0, commit=True,
        )
        # Use the full forward's logits (which include refinement) over the
        # incremental forward's logits (which skip refinement).
        # cached_logits stays from prefill_result.

        while tokens_generated < max_new_tokens:
            # ── Step 1: Get bonus token from cached logits ──
            bonus_token = self._sample_token(cached_logits[:, -1, :], temperature)

            if bonus_token.item() in stop_set:
                generated = torch.cat([
                    generated, bonus_token.unsqueeze(0).unsqueeze(0),
                ], dim=1)
                break

            # ── Step 2: Draft — DFlash head produces per-position marginals ──
            if cached_taps is not None:
                tap_hiddens = cached_taps
            else:
                tap_hiddens = self._collect_tap_hiddens(generated)

            ctx_start = max(0, generated.shape[1] - 64)
            ctx_hiddens = {k: v[:, ctx_start:, :] for k, v in tap_hiddens.items()}

            active_mask = self.state.get_active_mask()
            active_beliefs = self.state.beliefs.data[active_mask] if active_mask.any() else None

            context, injection = self.dflash_head.extract_features(
                ctx_hiddens, active_beliefs,
            )

            remaining = max_new_tokens - tokens_generated - 1
            draft_length = min(max_block, max(remaining, 1))
            spec_head_fn = self._get_spec_head_fn()
            draft_logits = self.dflash_head(
                context, lm_head_fn=spec_head_fn,
                injection=injection, draft_length=draft_length,
            )  # [1, draft_length, vocab]

            # ── Step 3: Build tree (CPU, O(B log B)) ──
            tree = build_ddtree(draft_logits[0], tree_budget)

            if tree.node_count == 0:
                # Empty tree → just accept bonus token
                bonus_ids = bonus_token.unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, bonus_ids], dim=1)
                tokens_generated += 1
                # Commit the single bonus token
                self.incremental_forward(
                    bonus_ids, inc_state,
                    position_offset=inc_state.seq_len, commit=True,
                )
                # Get fresh logits via full forward (fallback for empty tree)
                cached_logits = self.forward(generated)['logits']
                cached_taps = None
                if bonus_token.item() in stop_set:
                    break
                continue

            # ── Step 4: Compile tree → verification tensors ──
            T_past = inc_state.seq_len
            tree_input_ids, tree_position_ids, tree_attn_mask = compile_ddtree(
                tree,
                root_token_id=bonus_token.item(),
                start_position=T_past,
                past_length=T_past,
                device=device,
                dtype=next(self.parameters()).dtype,
            )

            # ── Step 5: Speculative verify (incremental, no commit) ──
            # Checkpoint recurrent states so we can rollback after verify.
            # Attention caches are append-only — we don't commit during verify.
            inc_state.checkpoint()

            # Build tree attention mask for incremental verify:
            # The attention layers have cached KV from positions 0..T_past-1.
            # New tokens are the tree (1+N positions). The mask controls how
            # tree tokens attend to each other (ancestor-only visibility).
            # Attention to cached KV is handled by the KV cache (always causal).
            # So the mask only needs to cover the tree-to-tree portion.
            T_tree = tree_input_ids.shape[1]
            mask_val = torch.finfo(next(self.parameters()).dtype).min
            tree_only_mask = torch.full(
                (1, 1, T_tree, T_past + T_tree), mask_val,
                dtype=next(self.parameters()).dtype, device=device,
            )
            # Tree tokens attend to all past (columns 0..T_past-1)
            tree_only_mask[:, :, :, :T_past] = 0.0
            # Tree-to-tree: ancestor-only visibility
            tree_only_mask[0, 0, :, T_past:].masked_fill_(
                tree.visibility.to(device), 0.0,
            )

            verify_result = self.incremental_forward(
                tree_input_ids, inc_state,
                position_offset=T_past,
                commit=False,  # speculative — don't persist
                attn_mask=tree_only_mask,
            )
            verify_logits = verify_result['logits']  # [1, 1+N, vocab]

            # ── Step 6: Walk tree → accepted path ──
            accepted_indices, next_bonus_token = follow_verified_tree(
                tree, verify_logits, temperature,
            )

            # ── Step 7: Rollback + commit accepted path only ──
            inc_state.restore()  # rollback recurrent states

            # Gather accepted token IDs
            accepted_idx_tensor = torch.tensor(
                accepted_indices, dtype=torch.long, device=device,
            )
            accepted_tokens = tree_input_ids.index_select(1, accepted_idx_tensor)

            # Commit: replay accepted tokens through backbone incrementally.
            # This updates recurrent state correctly (only accepted tokens affect it)
            # and appends only accepted KV to attention caches.
            commit_result = self.incremental_forward(
                accepted_tokens, inc_state,
                position_offset=T_past,
                commit=True,
            )

            # Also commit the next bonus token (first unmatched verifier token)
            next_bonus_ids = torch.tensor(
                [[next_bonus_token]], dtype=torch.long, device=device,
            )
            bonus_commit = self.incremental_forward(
                next_bonus_ids, inc_state,
                position_offset=T_past + len(accepted_indices),
                commit=True,
            )

            # Update generated sequence
            new_tokens = torch.cat([accepted_tokens, next_bonus_ids], dim=1)
            generated = torch.cat([generated, new_tokens], dim=1)
            n_accepted = len(accepted_indices)
            tokens_generated += n_accepted

            # Cache logits: the verify pass has logits for all tree nodes.
            # The last accepted node's logits are used for drafting next round.
            # Use the verify logits at the last accepted index for the bonus token.
            cached_logits = bonus_commit['logits']  # logits after bonus token
            cached_taps = bonus_commit.get('dflash_tapped')

            # Check for stop tokens
            if stop_set:
                all_new = new_tokens[0].tolist()
                if any(t in stop_set for t in all_new):
                    break

        return generated

    def _get_spec_head_fn(self):
        """Get the LM head function for speculative decoding (token or byte mode)."""
        if self.blt_enabled:
            _dp = self.byte_decoder.down_proj
            _bh = self.byte_decoder.byte_heads[0]
            return lambda h: _bh(F.rms_norm(_dp(h), (_dp.out_features,)))
        return self.transformer.head

    def _collect_tap_hiddens(self, input_ids: Tensor) -> dict[int, Tensor]:
        """Run transformer blocks and collect hidden states at DFlash tap layers."""
        B, T = input_ids.size()
        M = self.working_memory_size if self.working_memory_size > 0 else 0
        T_total = T + M

        self.transformer._ensure_rope(T_total)
        cos = self.transformer.cos[:, :T_total]
        sin = self.transformer.sin[:, :T_total]

        x = self.transformer.wte(input_ids)
        x = F.rms_norm(x, (x.size(-1),))

        if M > 0:
            wm = self.working_memory.expand(B, -1, -1)
            x = torch.cat([x, wm], dim=1)

        x0 = x

        engram_out = self.engram_cache(x[:, :T, :], input_ids)
        x = torch.cat([x[:, :T, :] + engram_out, x[:, T:, :]], dim=1) if M > 0 else x + engram_out

        tap_set = set(self.dflash_head.tap_indices)
        tapped: dict[int, Tensor] = {}

        for i, block in enumerate(self.transformer.blocks):
            x = block(
                x, x0, cos, sin,
                self.transformer.resid_lambdas[i],
                self.transformer.x0_lambdas[i],
            )
            if i in tap_set:
                tapped[i] = x[:, :T, :] if M > 0 else x

        return tapped

    @staticmethod
    def _sample_token(logits: Tensor, temperature: float) -> Tensor:
        """Sample from logits (greedy if temperature < 1e-5)."""
        if temperature < 1e-5:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        if probs.dim() == 1:
            return torch.multinomial(probs.unsqueeze(0), 1).squeeze()
        return torch.multinomial(probs, 1).squeeze(-1)

    def _generate_autoregressive(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_set: set[int],
    ) -> Tensor:
        """Standard autoregressive generation (fallback when DFlash disabled)."""
        for _ in range(max_new_tokens):
            result = self.forward(tokens)
            logits = result['logits']
            next_token = self._sample_token(logits[:, -1, :], temperature)
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() in stop_set:
                break
        return tokens

    def detach_state(self):
        """Detach state tensors from computation graph before structural pass2 modifications."""
        self.state.beliefs.detach_()
        self.state.edge_weights.detach_()
        self.state.edge_relations.detach_()
        self.state.goal_embeddings.detach_()
        self.state.goal_status_logits.detach_()
        self.state.edge_direction.detach_()

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
