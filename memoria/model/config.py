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
    window_pattern: str = "SSSL"     # sliding window pattern from autoresearch

    # State interface placement
    interface_every: int = 4          # insert state interface every N layers
    interface_num_heads: int = 4      # retrieval heads in read path
    interface_top_k: int = 32         # max beliefs to attend over


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
    weight_decay: float = 0.2
    adam_betas: tuple = (0.8, 0.95)

    # Schedule
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    # Free energy
    alpha_max: float = 0.1            # max weight for L_fe
    alpha_warmup_steps: int = 1000    # KL annealing: α ramps from 0 to alpha_max
    fe_temperature: float = 5.0       # temperature for energy computation

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


# ── Preset Configurations ──

def small_config() -> MemoriaConfig:
    """~125M params (+ ~117M embedding). Single 3090. Rapid iteration.
    Total ~245M params. Fits in <10GB VRAM at B=8.
    1B tokens ≈ 12 hours on single 3090, 6 hours on 2x 3090.
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=4096, max_edges=16384,
            max_goals=64, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=2,  # 151K vocab → logits gradient is [B*T, 151K]. B=2 fits in 24GB.
        ),
    )


def medium_config() -> MemoriaConfig:
    """~300M params (+ ~156M embedding). 2x 3090. Serious training.
    Total ~456M params. ~14GB VRAM at B=4.
    2B tokens ≈ 12 hours on 2x 3090.
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=24, n_head=8, n_kv_head=8, n_embd=1024,
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=8192, max_edges=32768,
            max_goals=64, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=4,  # bigger model + big vocab = smaller batch
        ),
    )


def large_config() -> MemoriaConfig:
    """~500M params (+ ~194M embedding). B200 or multi-GPU. Crossover experiment.
    Total ~694M params. Needs B200 or model parallel on 2x 3090.
    """
    return MemoriaConfig(
        transformer=TransformerConfig(
            n_layer=24, n_head=10, n_kv_head=10, n_embd=1280,
            interface_every=4,
        ),
        state=StateConfig(
            belief_dim=256, max_beliefs=16384, max_edges=65536,
            max_goals=128, relation_dim=64,
        ),
        training=TrainingConfig(
            device_batch_size=2,  # large model on B200 can use bigger batch
            alpha_max=0.1,
        ),
    )
