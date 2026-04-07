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

    # Attention pattern: M = Mamba-2, S = Sliding window, L = Long/global (MLA)
    # "M" = all Mamba (cognitive state handles global context, O(T) linear)
    # "MMMML" = 4 Mamba + 1 MLA per cycle (hybrid, recommended for scratch training)
    # "S" = all sliding window (legacy, O(T×W))
    # "SSSL" = 3 sliding + 1 global per cycle (for models without cognitive state)
    window_pattern: str = "MMMML"
    sliding_window_size: int = 4096   # local window for S layers

    # Mamba-2 parameters (for M layers)
    mamba_d_state: int = 64           # SSM state expansion factor (64 or 128)
    mamba_d_conv: int = 4             # local convolution width
    mamba_expand: int = 2             # block expansion factor (inner_dim = expand * d_model)

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

    # Read gate initial bias (sigmoid(x) ≈ opening fraction)
    read_gate_init_bias: float = 2.0        # sigmoid(2.0) ≈ 0.88 — starts mostly open

    # Working memory init scale (small normal around zero)
    working_memory_init_scale: float = 0.02

    # DFlash block diffusion draft head (speculative decoding)
    dflash_enabled: bool = False       # enable DFlash draft head
    dflash_n_layers: int = 3           # draft head layers (small — this is the "fast" path)
    dflash_block_size: int = 8         # tokens to draft per speculation step
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
            window_pattern="MMMML",  # 4 Mamba + 1 MLA per 5-layer cycle
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
            window_pattern="MMMML",  # 4 Mamba + 1 MLA per 5-layer cycle
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
            window_pattern="MMMML",  # 4 Mamba + 1 MLA per 5-layer cycle
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
