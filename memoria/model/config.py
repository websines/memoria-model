"""Model configurations.

Three sizes targeting different hardware:
- small:  ~125M params, single 3090, rapid iteration (2-4 hours)
- medium: ~300M params, 2x 3090, serious training (8-12 hours)
- large:  ~500M params, B200 or multi-GPU, crossover experiment

Architecture based on autoresearch/train.py (Karpathy) with additions:
- State interface layers every K transformer blocks
- Cognitive state attached to model
- Training with L_token + L_fe
"""

from dataclasses import dataclass, field
from memoria.core.state import StateConfig


@dataclass
class TransformerConfig:
    """Configuration for the transformer backbone."""
    vocab_size: int = 151936       # Qwen3 tokenizer vocab size (byte-level BPE, 2026)
    sequence_len: int = 2048
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

    # Attention pattern: H = Log-Linear DeltaProduct, D = DeltaProduct, E = Log-Linear GDN,
    #                    S = Sliding window, L = Long/global (MLA)
    # "HHHHL" = 4 Log-Linear DeltaProduct₃ + 1 MLA per cycle (maximum expressivity)
    # "DDDEL" = 3 DeltaProduct₃ + 1 Log-Linear GDN + 1 MLA (fast training, separate layers)
    # "DDDML" = 3 DeltaProduct₃ + 1 MLA per cycle (simplest upgrade from Mamba-2)
    # "S" = all sliding window (legacy, O(T×W))
    window_pattern: str = "HHHHL"
    sliding_window_size: int = 4096   # local window for S layers

    # GatedDeltaProduct parameters (for D and H layers)
    deltaproduct_n_householder: int = 3       # Householder reflections per token (expressivity knob)
    deltaproduct_head_dim: int = 128          # key dimension per head
    deltaproduct_expand_v: int = 2            # value dim = head_dim × expand_v
    deltaproduct_allow_neg_eigval: bool = True  # [-1,1] eigenvalues — REQUIRED for state tracking
    deltaproduct_conv_size: int = 4           # short causal convolution width
    deltaproduct_use_forget_gate: bool = True  # scalar forget gate
    deltaproduct_use_short_conv: bool = True   # pre-SSM convolution

    # Log-Linear parameters (for H and E layers)
    loglinear_chunk_size: int = 64            # chunk size for Fenwick tree (must be 64 for Log-Linear)

    # Legacy Mamba-2 parameters (for M layers — kept for backward compat with old configs)
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # RoPE position encoding — native long context, no scaling
    # High base frequency for native 200K support (like Llama 3 at 128K)
    max_position: int = 204800        # max position for RoPE
    rope_scaling: str = "none"        # "none" or "yarn" (yarn only for extending pretrained models)
    rope_scaling_factor: float = 1.0  # only used when rope_scaling="yarn"
    rope_base: int = 500000           # high base for native long context (Llama 3 style)

    # Multi-Head Latent Attention (MLA) for L layers — DeepSeek V3 style
    # Only used when window_pattern contains "L" layers.
    # Set mla_latent_dim=0 to disable (L layers use standard full attention).
    mla_latent_dim: int = 0           # KV compression dimension (0 = disabled)
    mla_rope_dim: int = 64            # RoPE dimensions kept uncompressed in MLA
    mla_window_size: int = 0          # sliding window for MLA layers (0 = full attention)
                                      # At long context (>128K), use 65536-131072 to avoid O(T²)
                                      # Cognitive state + DeltaProduct handle beyond-window coherence

    # State interface placement
    interface_every: int = 4          # insert state interface every N layers
    interface_num_heads: int = 4      # retrieval heads in read path
    interface_top_k: int = 32         # max beliefs to attend over

    # Working memory prefix (Mamba-inspired scratchpad)
    working_memory_size: int = 8      # M learnable tokens prepended to hidden stream

    # Engram static knowledge cache
    engram_table_size: int = 50000    # entries per hash table
    engram_n_heads: int = 4           # hash heads per N-gram order
    engram_embed_dim: int = 0         # per-head embed dim (0 = auto)

    # Refinement loops (Mamba-inspired iterative reasoning)
    max_refinement_loops: int = 3     # max loops on upper layers (0 to disable)
    refinement_halt_threshold: float = 0.7  # P(halt) above this exits the loop
    refinement_gate_init: float = 0.1       # lifeline gate initial value (per-dim)

    # Predictive refinement (MoR + SCORE + error-gated retrieval)
    # Per-position routing: lightweight router decides which positions need further
    # refinement. Contractive scaling: later loops contribute geometrically smaller
    # corrections. Error-gated retrieval: skip belief re-query for small-delta positions.
    # All thresholds are learned MetaParams — no hardcoded magic numbers.
    # Reference: MoR (NeurIPS 2025, arXiv:2507.10524) — per-token adaptive recursion
    # Reference: SCORE (arXiv:2603.10544) — contractive recurrent depth
    # Reference: PonderNet (arXiv:2107.05407) — learned halting
    # Reference: Two-Scale Latent Dynamics (NeurIPS 2025, arXiv:2509.23314)
    predictive_refinement: bool = True  # enable per-position adaptive refinement

    # PARL-style internal parallel goal pursuit
    # Multi-head goal-modulated attention: different read heads pursue different
    # active goals simultaneously. Batched autoresearch generates hypotheses for
    # multiple goals in parallel with fair round-robin allocation. Staged reward
    # shaping in the cognitive controller prevents serial collapse (defaulting to
    # one goal at a time). All weights are learned MetaParams — no hardcoded numbers.
    # Reference: PARL (Kimi K2.5, arXiv:2602.02276) — staged reward, serial collapse
    # Reference: MoH (arXiv:2410.11842) — mixture-of-head attention routing
    # Reference: MOORE (arXiv:2311.11385) — orthogonal expert specialization
    # Reference: GCR-PPO (arXiv:2509.14816) — per-objective gradient decomposition
    parallel_goals: bool = True  # enable PARL-style parallel goal pursuit

    # Read gate initial bias (sigmoid(x) ≈ opening fraction)
    read_gate_init_bias: float = 2.0        # sigmoid(2.0) ≈ 0.88 — starts mostly open

    # Working memory init scale (small normal around zero)
    working_memory_init_scale: float = 0.02

    # DSA (DeepSeek Sparse Attention) — belief-conditioned Lightning Indexer
    # Replaces windowed MLA with sparse global attention: a lightweight indexer
    # scores all tokens, selects top-k, then full MLA runs on the sparse subset.
    # Indexer keys compressed via RotorQuant STE QAT (same pipeline as KV/weights).
    # Belief conditioning: active beliefs bias indexer scores toward relevant tokens.
    # Set dsa_enabled=False to disable (MLA uses windowed or full causal attention).
    dsa_enabled: bool = False          # enable belief-conditioned sparse attention
    dsa_index_dim: int = 32            # indexer projection dimension (small = cheap scoring)
    dsa_index_heads: int = 4           # indexer heads (parallel scoring channels)
    dsa_top_k: int = 2048              # tokens selected per query position at inference
    dsa_top_k_ratio: float = 0.25      # fraction selected during training (when T < dsa_top_k)
    dsa_index_bits: int = 3            # RotorQuant bits for indexer key compression
    dsa_belief_lambda: float = 0.1     # belief conditioning strength (0 = no belief bias)

    # Weight QAT (Quantization-Aware Training) via RotorQuant + CAGE
    # Applies STE quantization noise to weight matrices during forward pass,
    # training the model to produce weight distributions robust to low-bit compression.
    # CAGE post-step correction nudges weights toward the quantization grid.
    # Set weight_qat_bits=0 to disable.
    weight_qat_bits: int = 4           # default bit-width for weight QAT (0 = disabled)
    weight_qat_mlp_bits: int = 3       # MLP weights get more aggressive quantization (0 = use weight_qat_bits)

    # BLT (Byte Latent Transformer) — tokenizer-free byte-level I/O
    # Replaces token embedding/LM head with byte encoder/decoder.
    # Global backbone operates on patches (compressed byte groups).
    # Eliminates: 117M-param LM head, softmax bottleneck, gradient bottleneck.
    # LM head drops to 197K params (768 × 260 byte vocab).
    # Local layers use DeltaProduct (O(T), same kernel as backbone).
    # Reference: BLT (Meta, arXiv:2412.09871) — byte latent transformer
    # Reference: MambaByte (arXiv:2401.13660) — byte-level SSM
    # Reference: EvaByte (2025) — linear attention + bytes
    blt_enabled: bool = False          # enable byte-level I/O (replaces token vocab)
    blt_local_dim: int = 384           # local encoder/decoder hidden dim (< n_embd)
    blt_patch_size: int = 6            # bytes per patch (6 ≈ BPE token length)
    blt_local_layers: int = 2          # DeltaProduct layers in encoder AND decoder
    blt_byte_vocab: int = 260          # 256 bytes + BOS + EOS + PAD + SEP
    blt_n_byte_heads: int = 4          # multi-byte prediction heads (for DFlash)
    blt_head_dim: int = 64             # head dim for local DeltaProduct layers

    # DFlash block diffusion draft head (speculative decoding)
    # Three improvements over baseline block diffusion:
    # 1. KV injection: per-layer K/V projections for tapped features (not concat)
    # 2. Streak distillation: position-weighted CE + expected streak bonus
    # 3. Adaptive block size: entropy-based cutoff, draft max but verify fewer
    # All thresholds are learned MetaParams — no hardcoded magic numbers.
    # Reference: DFlash (arXiv:2602.06036) — KV injection, block diffusion
    # Reference: SpecDiff-2 (arXiv:2511.00606) — streak distillation
    # Reference: FailFast (arXiv:2512.20573) — adaptive block size
    # Reference: DEER (arXiv:2512.15176) — single-step diffusion drafting
    dflash_enabled: bool = False       # enable DFlash draft head
    dflash_n_layers: int = 3           # draft head layers (small — this is the "fast" path)
    dflash_block_size: int = 8         # training window (tokens per training sample)
    dflash_max_block_size: int = 32    # inference max (adaptive cutoff reduces to block_size in hard regions)
    dflash_loss_weight: float = 0.1    # weight for draft training loss


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Batch
    total_batch_size: int = 2**17     # ~128K tokens per step
    device_batch_size: int = 64       # per-device, adjust for VRAM

    # Optimizer (Muon + AdamW from autoresearch)
    matrix_lr: float = 0.04
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    scalar_lr: float = 0.5
    interface_lr: float = 0.01        # for state interface layer params
    belief_lr: float = 0.0001         # slow LR for cognitive state (beliefs, edges)
    cognitive_meta_lr: float = 0.001  # LR for learned meta-parameters
    weight_decay: float = 0.2
    adam_betas: tuple = (0.8, 0.95)

    # Schedule
    warmup_ratio: float = 0.02         # 2% warmup — stabilizes Adam estimates early
    warmdown_ratio: float = 0.2        # 20% warmdown (WSD schedule)
    warmdown_type: str = "linear"      # "linear" (decay-to-zero) or "cosine"
    final_lr_frac: float = 0.0
    grad_clip_norm: float = 1.0        # max gradient norm (0 to disable)

    # Free energy
    alpha_max: float = 0.1            # max weight for L_fe
    alpha_warmup_steps: int = 1000    # KL annealing: α ramps from 0 to alpha_max
    fe_temperature: float = 5.0       # temperature for energy computation

    # Auxiliary loss weights (relative to alpha)
    utility_loss_weight: float = 0.1   # weight for L_utility (belief usefulness)
    surprise_loss_weight: float = 0.1  # weight for L_surprise (telos learning)
    halt_loss_weight: float = 0.1      # weight for L_halt (refinement probe)

    # Training phases
    phase1_steps: int = 2000          # L_token only (language foundation)
    phase2_steps: int = 3000          # α ramps up (cognitive awakening)
    # phase 3: full training continues indefinitely

    # SkyLadder progressive context extension
    # Ramps sequence_len from skyladder_start to sequence_len over skyladder_ratio
    # of total training. Short context is cheaper (less attention compute) and
    # empirically produces better representations (SkyLadder, NeurIPS 2025).
    # Set skyladder_ratio=0 to disable (fixed sequence_len throughout).
    skyladder_ratio: float = 0.6      # fraction of training spent ramping (0 = disabled)
    skyladder_start: int = 256        # initial context length (small = fast early training)
    skyladder_schedule: str = "linear"  # "linear", "exponential", or "step"

    # CAGE (Curvature-Aware Gradient Estimation) for weight QAT
    # Post-optimizer correction that pushes weights toward quantization grid.
    # Silent during phase 1, ramps during phase 2, full strength in phase 3.
    # Reference: CAGE (arXiv 2510.18784, IST-DASLab 2025)
    cage_lambda_base: float = 10.0     # CAGE correction strength at full ramp
    cage_silence_ratio: float = 0.0    # fraction of training with CAGE silent (0 = use phase-aligned schedule)

    # DSA indexer KL alignment loss
    # Trains the Lightning Indexer to predict which tokens full attention would focus on.
    # Active during phase 1 (dense attention is cheap at short context, provides KL target).
    # Continues at reduced weight during phase 2+3 for alignment maintenance.
    dsa_kl_weight: float = 1.0         # KL alignment loss weight during phase 1
    dsa_kl_weight_after: float = 0.1   # KL weight after phase 1 (maintenance)

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000


@dataclass
class MemoriaConfig:
    """Full configuration combining all components."""
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    state: StateConfig = field(default_factory=StateConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Backbone mode: "scratch" trains transformer from scratch, "pretrained" bolts onto HF model
    backbone: str = "scratch"
    pretrained_model: str = ""  # HuggingFace model name (only used when backbone="pretrained")


# ── Preset Configurations ──

def small_config() -> MemoriaConfig:
    """~125M params (+ ~117M embedding). Single 3090. Rapid iteration.
    Total ~245M params. Fits in <10GB VRAM.

    All sliding window attention (4K) with RotorQuant KV compression.
    Cognitive state handles global context. Native 200K via high-base RoPE.

    KV QAT (quantization-aware training via STE) during training teaches
    the model to produce representations robust to 3-bit compression.

    State scaled 4x (RotorQuant checkpoint compression keeps disk flat):
    - 16K beliefs (was 4K) — 1 belief per 12.5 tokens at 200K
    - 65K edges (was 16K) — richer causal graph
    - 256 goals (was 64) — more concurrent objectives
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
            window_pattern="HHHHL",  # 4 Log-Linear DeltaProduct₃ + 1 MLA per 5-layer cycle
            mla_latent_dim=192,      # enable MLA for L layers (latent = n_embd/4)
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=16384, max_edges=65536,
            max_goals=256, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=2,
        ),
    )


def medium_config() -> MemoriaConfig:
    """~300M params (+ ~156M embedding). 2x 3090. Serious training.
    Total ~456M params.

    State scaled 4x (RotorQuant compressed):
    - 32K beliefs — 1 belief per 6.25 tokens at 200K
    - 131K edges — dense causal graph
    - 256 goals
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=24, n_head=8, n_kv_head=8, n_embd=1024,
            window_pattern="HHHHL",  # 4 Log-Linear DeltaProduct₃ + 1 MLA per 5-layer cycle
            mla_latent_dim=256,      # enable MLA for L layers (latent = n_embd/4)
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=32768, max_edges=131072,
            max_goals=256, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=4,
        ),
    )


def large_config() -> MemoriaConfig:
    """~500M params (+ ~194M embedding). B200 or multi-GPU. Crossover experiment.
    Total ~694M params.

    State scaled 4x (RotorQuant compressed):
    - 65K beliefs — 1 belief per 3 tokens at 200K (very dense coverage)
    - 262K edges — massive causal graph
    - 512 goals
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=24, n_head=10, n_kv_head=10, n_embd=1280,
            window_pattern="HHHHL",  # 4 Log-Linear DeltaProduct₃ + 1 MLA per 5-layer cycle
            mla_latent_dim=320,      # enable MLA for L layers (latent = n_embd/4)
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=65536, max_edges=262144,
            max_goals=512, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=2,
            alpha_max=0.1,
        ),
    )


def lfm2_config() -> MemoriaConfig:
    """LFM2.5-350M with cognitive state bolted on.

    Backbone is frozen. Only interface layers (~15M params) are trained.
    LFM2: 16 layers (10 conv + 6 attn), 1024 hidden, 65536 vocab.
    Interface layers inserted after attention layers [2, 5, 8, 10, 12, 14].

    Why LFM2.5-350M:
    - 350M params — small enough that cognitive state must earn its keep
    - 28T tokens training — best language capability per parameter at this scale
    - Hybrid conv+attention — conv handles local, cognitive state handles global
    - 1024 hidden dim — matches Memoria medium config
    - 128K native context
    """
    return MemoriaConfig(
        backbone="pretrained",
        pretrained_model="LiquidAI/LFM2.5-350M",
        transformer=TransformerConfig(
            vocab_size=65536,
            sequence_len=2048,
            n_layer=16,
            n_head=16,
            n_kv_head=8,
            n_embd=1024,
            # Interface placement is overridden to attention-only positions
            # in PretrainedMemoriaModel when backbone_type == "lfm2".
            # interface_every is approximate — actual positions: [2,5,8,10,12,14]
            interface_every=3,
            interface_num_heads=4,
            interface_top_k=48,
            max_position=128000,
            rope_scaling="none",   # LFM2 handles its own RoPE
            rope_base=1000000,     # from LFM2 config.json rope_theta
        ),
        state=StateConfig(
            belief_dim=256,
            max_beliefs=16384,
            max_edges=65536,
            max_goals=256,
            relation_dim=64,
        ),
        training=TrainingConfig(
            total_batch_size=2**13,   # 8192 tokens/step
            device_batch_size=2,      # 350M fits easily on 24GB GPU
            interface_lr=0.001,
            phase1_steps=200,         # very short — backbone already knows language
            alpha_warmup_steps=300,
            alpha_max=0.1,
            fe_temperature=5.0,
        ),
    )


def qwen_config() -> MemoriaConfig:
    """Qwen3.5-2B-Base with cognitive state bolted on.

    Backbone is frozen. Only interface layers (~25M params) are trained.
    Qwen3.5-2B: 24 layers, 2048 hidden, 8 heads, 2 KV heads, 248320 vocab.
    Hybrid architecture: linear_attention + full_attention (every 4th layer).
    Interface layers inserted after layers 5, 11, 17, 23 (every 6 layers).

    Much faster training: only adapters + pass 2. Should work on a single 24GB GPU
    with small batch since the backbone runs in eval mode with no grad.
    """
    return MemoriaConfig(
        backbone="pretrained",
        pretrained_model="Qwen/Qwen3.5-2B-Base",
        transformer=TransformerConfig(
            # These must match Qwen3.5-2B architecture for interface layer dims
            vocab_size=248320,
            sequence_len=2048,
            n_layer=24,
            n_head=8,
            n_kv_head=2,
            n_embd=2048,
            interface_every=6,  # 4 interface layers at positions 5, 11, 17, 23
            interface_num_heads=4,
            interface_top_k=64,  # more beliefs since the model is more capable
            # Pretrained mode: backbone handles its own attention.
            # Qwen3.5 uses YaRN internally — we extend its RoPE for long context.
            max_position=204800,
            rope_scaling="yarn",
            rope_scaling_factor=100.0,
        ),
        state=StateConfig(
            belief_dim=512,  # larger to match richer representations from 2B model
            max_beliefs=32768,   # 4x from 8K — RotorQuant compressed
            max_edges=262144,    # 8× beliefs — 2B model creates edges fast
            max_goals=256,
            relation_dim=64,
        ),
        training=TrainingConfig(
            total_batch_size=2**13,  # 8192 tokens/step — adapters don't need 131K
            device_batch_size=1,  # 2B model + grad checkpointing on 24GB GPU
            interface_lr=0.001,  # lower LR for adapters on pretrained backbone
            phase1_steps=500,    # shorter phase 1: backbone already knows language
            alpha_warmup_steps=500,
            alpha_max=0.1,
            fe_temperature=5.0,
        ),
    )


def full_config() -> MemoriaConfig:
    """Full-featured Memoria from scratch. 2x 3090 (48GB). All systems active.

    BLT byte-level I/O eliminates the 233M-param embedding/LM head bottleneck.
    ~122M active params. 128K byte context via SkyLadder exponential ramp.
    Full curated data mix with 45% state-essential data.

    All cognitive systems enabled:
    - 12 Pass 2 operations (beliefs, edges, goals, autoresearch, sleep, planning...)
    - Refinement loops with predictive refinement (MoR + SCORE)
    - PARL parallel goal pursuit
    - BLT byte-level I/O (tokenizer-free)
    - DFlash speculative decoding
    - DSA belief-conditioned sparse attention
    - Weight QAT + CAGE
    - SkyLadder progressive context (512 → 128K bytes)
    - TTT in-place test-time training
    - Engram static cache
    - Kendall/Gal uncertainty weighting

    VRAM estimate per GPU: ~12 GB at peak context (batch_size=1, 128K bytes).

    Training profile (SkyLadder exponential):
      0-10%:  512-1K bytes → blazing fast, high effective batch
      10-30%: 1K-4K bytes → fast
      30-50%: 4K-16K bytes → moderate
      50-60%: 16K-128K bytes → slower, but H layers are O(T), DSA is O(T×K)
      60-100%: 128K bytes → steady state, full context
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
            window_pattern="HHHHL",
            mla_latent_dim=192,

            # State interface
            interface_every=4,
            interface_num_heads=4,
            interface_top_k=32,

            # Working memory + Engram
            working_memory_size=8,
            engram_table_size=50000,
            engram_n_heads=4,

            # Refinement loops (latent CoT)
            max_refinement_loops=3,
            predictive_refinement=True,

            # PARL parallel goals
            parallel_goals=True,

            # BLT byte-level (eliminates 233M embedding bottleneck)
            blt_enabled=True,
            blt_local_dim=384,
            blt_patch_size=6,
            blt_local_layers=2,
            blt_n_byte_heads=4,

            # DFlash speculative decoding
            dflash_enabled=True,
            dflash_n_layers=3,
            dflash_block_size=8,

            # Weight QAT
            weight_qat_bits=4,
            weight_qat_mlp_bits=3,

            # DSA sparse attention
            dsa_enabled=True,
            dsa_index_dim=32,
            dsa_top_k=2048,

            # 128K byte context (≈36K tokens equivalent)
            sequence_len=131072,
        ),
        state=StateConfig(
            belief_dim=256,
            max_beliefs=8192,
            max_edges=32768,
            max_goals=128,
            relation_dim=64,
        ),
        training=TrainingConfig(
            total_batch_size=2**17,    # 128K bytes/step
            device_batch_size=1,       # 1 seq/GPU at peak context

            phase1_steps=2000,
            phase2_steps=3000,
            alpha_warmup_steps=1000,
            alpha_max=0.1,

            # SkyLadder: 512 → 128K bytes, exponential (more time at cheap short ctx)
            skyladder_ratio=0.6,
            skyladder_start=512,
            skyladder_schedule="exponential",

            # CAGE
            cage_lambda_base=10.0,
        ),
    )
