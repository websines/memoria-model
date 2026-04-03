# Plan: Integrate LFM2.5-350M as Memoria Backbone

> Replaces Qwen3.5-2B-Base as the pretrained backbone.
> Goal: 350M frozen backbone + ~25M trainable interface layers + cognitive state.

---

## Why LFM2.5-350M

- **350M params** — small enough that cognitive state must earn its keep (vs 2B Qwen where backbone handles everything alone)
- **28T tokens training** — best language capability per parameter at this scale
- **Hybrid conv+attention** — conv handles local patterns, cognitive state handles global memory (complementary, not competing)
- **1024 hidden dim** — matches Memoria medium config
- **128K native context** — long range built in
- **CC-BY-4.0-equivalent** for research (LFM1.0 license, free under $10M revenue)

---

## LFM2 Internal Architecture

### Model Structure

```
Lfm2ForCausalLM
  .model: Lfm2Model
    .embed_tokens:    Embedding(65536, 1024, padding_idx=0)
    .layers:          ModuleList(16 x Lfm2DecoderLayer)
    .rotary_emb:      Lfm2RotaryEmbedding
    .embedding_norm:   Lfm2RMSNorm(1024, eps=1e-05)   # NOT .norm!
  .lm_head:           Linear(1024, 65536, bias=False)  # tied to embed_tokens
```

### Layer Pattern (16 layers)

```
Index:  0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
Type:  conv  conv  ATTN  conv  conv  ATTN  conv  conv  ATTN  conv  ATTN  conv  ATTN  conv  ATTN  conv
```

- **10 conv layers** (Lfm2ShortConv): gated depthwise conv, kernel=3, local context only
- **6 attention layers** (Lfm2Attention): GQA with 16 heads, 8 KV heads, full causal attention

### Unified Layer Forward Signature

Both conv and attention layers share the same `Lfm2DecoderLayer.forward()`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,          # [B, T, 1024]
    position_embeddings: tuple | None,     # (cos, sin) from rotary_emb
    attention_mask: torch.Tensor | None,   # 4D causal for attn, 2D bool for conv
    position_ids: torch.LongTensor | None,
    past_key_values: Lfm2HybridConvCache | None = None,
    **kwargs,
) -> torch.Tensor:                         # returns PLAIN TENSOR, not tuple!
```

### Key Differences from Standard HF Models

| Aspect | Standard (Llama/Qwen) | LFM2 |
|--------|----------------------|------|
| Final norm | `model.model.norm` | `model.model.embedding_norm` |
| Layer return | `(hidden, attn_weights, ...)` tuple | Plain `torch.Tensor` |
| Layer types | All transformer blocks | Mixed conv + attention |
| Attention mask | Same for all layers | Different per layer type |
| lm_head | Independent weights | **Tied** to embed_tokens |
| Cache | `DynamicCache` | `Lfm2HybridConvCache` |

---

## Implementation Plan

### Step 1: Add LFM2 Config Preset

**File:** `memoria/model/config.py`

```python
def lfm2_config() -> MemoriaConfig:
    """LFM2.5-350M with cognitive state bolted on.

    Backbone is frozen. Only interface layers (~15M params) are trained.
    LFM2: 16 layers (10 conv + 6 attn), 1024 hidden, 65536 vocab.
    Interface layers inserted after attention layers [2, 5, 8, 10, 12, 14].
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
            # Interface after every attention layer (6 total)
            # Attention layers are at indices [2, 5, 8, 10, 12, 14]
            # We can't use interface_every since layers aren't uniform.
            # Instead, we'll specify positions explicitly (see Step 2).
            interface_every=3,     # approximate — overridden by explicit positions
            interface_num_heads=4,
            interface_top_k=48,    # more than small (32), less than qwen (64)
            max_position=128000,
            rope_scaling="none",   # LFM2 handles its own RoPE
            rope_base=1000000,     # from config.json rope_theta
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
```

**Also update:** `__main__.py` to add `"lfm2"` choice alongside `"qwen"`.

### Step 2: Refactor PretrainedMemoriaModel for Multi-Backend Support

**File:** `memoria/model/pretrained_model.py`

The current code hardcodes assumptions about standard HF models. We need to
abstract the backbone-specific details.

#### 2a. Detect backbone type and configure accordingly

In `__init__`, after loading the model:

```python
# Detect backbone architecture
self._backbone_type = self._detect_backbone_type(hf_config)

if self._backbone_type == "lfm2":
    self._final_norm_attr = "embedding_norm"
    self._layer_returns_tuple = False
    # Interface after attention layers only (conv layers are local-only)
    attn_indices = [i for i, lt in enumerate(hf_config.layer_types)
                    if lt == "full_attention"]
    self.interface_positions = attn_indices  # [2, 5, 8, 10, 12, 14]
    n_interfaces = len(attn_indices)
else:
    # Standard transformer (Qwen, Llama, etc.)
    self._final_norm_attr = "norm"
    self._layer_returns_tuple = True
    n_interfaces = n_layers // config.transformer.interface_every
    self.interface_positions = [
        (i + 1) * config.transformer.interface_every - 1
        for i in range(n_interfaces)
    ]

@staticmethod
def _detect_backbone_type(hf_config) -> str:
    model_type = getattr(hf_config, 'model_type', '')
    if model_type == 'lfm2':
        return 'lfm2'
    return 'standard'
```

#### 2b. Fix final norm access

Replace line 219:
```python
# OLD:
hidden = backbone_model.norm(hidden)

# NEW:
final_norm = getattr(backbone_model, self._final_norm_attr)
hidden = final_norm(hidden)
```

#### 2c. Fix layer output handling

Replace lines 188-190:
```python
# OLD:
layer_output = layer(hidden, **layer_kwargs)
hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output

# NEW (already handles both, but simplify):
hidden = layer(hidden, **layer_kwargs)
if isinstance(hidden, tuple):
    hidden = hidden[0]
```

#### 2d. Handle per-layer attention masks

LFM2 passes different masks to conv vs attention layers. For training
(no padding, no cache), passing `None` for both works. But for proper
support:

```python
# In forward(), before the layer loop:
if self._backbone_type == "lfm2":
    # LFM2 needs layer-type-aware masking
    layer_types = self.backbone.config.layer_types
    # For training: None works for both types (no padding)
    # For inference with padding: would need causal_mask vs bool_mask
    layer_kwargs_per_type = {
        'full_attention': {**layer_kwargs},
        'conv': {**layer_kwargs},
    }
```

For now (training only, no padding), `None` mask is fine for both.

#### 2e. Rotary embeddings

LFM2 computes rotary embeddings in `Lfm2Model.forward()`, not per-layer.
The current code already handles this:

```python
if hasattr(backbone_model, 'rotary_emb'):
    position_embeddings = backbone_model.rotary_emb(hidden, position_ids)
```

This works for LFM2. Conv layers receive `position_embeddings` but ignore them internally.

### Step 3: Interface Layer Placement Strategy

With 6 attention layers at indices [2, 5, 8, 10, 12, 14], we place
interfaces after each one — giving us **6 interface layers**.

Why after attention layers only:
- Conv layers (kernel=3) only see 3 tokens of local context
- Attention layers see the full sequence — the hidden state contains global information
- Reading/writing beliefs after a conv layer would be reading from a locally-mixed representation (low quality)
- Reading/writing after attention gets a globally-contextualized representation

This is actually better than the Qwen setup (4 interfaces every 6 layers)
because every interface sits at a natural global-context synchronization point.

```
Layer 0 (conv) → Layer 1 (conv) → Layer 2 (ATTN) → [INTERFACE 0]
Layer 3 (conv) → Layer 4 (conv) → Layer 5 (ATTN) → [INTERFACE 1]
Layer 6 (conv) → Layer 7 (conv) → Layer 8 (ATTN) → [INTERFACE 2]
Layer 9 (conv) → Layer 10 (ATTN) → [INTERFACE 3]
Layer 11 (conv) → Layer 12 (ATTN) → [INTERFACE 4]
Layer 13 (conv) → Layer 14 (ATTN) → [INTERFACE 5]
Final norm → LM head
```

6 interfaces × ~2.5M params each ≈ **15M trainable parameters** (less than Qwen's 25M
because hidden_dim is 1024 vs 2048).

### Step 4: Read Gate Initialization

For LFM2, start the read gate more conservatively than Qwen:

```python
# Qwen (2B backbone, can tolerate disruption):
self.read_gate = nn.Parameter(torch.full((n_interfaces,), -5.0))
# sigmoid(-5) ≈ 0.007

# LFM2 (350M backbone, more fragile):
self.read_gate = nn.Parameter(torch.full((n_interfaces,), -8.0))
# sigmoid(-8) ≈ 0.0003 — even more conservative
```

The smaller backbone is more sensitive to perturbation. Start near-zero
and let the gate learn to open.

### Step 5: Tokenizer Handling

LFM2 uses its own tokenizer (65536 vocab). Update `tokenizer.py`:

```python
TOKENIZER_PREFERENCES = [
    "LiquidAI/LFM2.5-350M",                  # 65K vocab, LFM2 tokenizer
    "Qwen/Qwen3-0.6B",                        # 151K vocab, byte-level BPE
    "meta-llama/Meta-Llama-3-8B",              # 128K vocab
    ...
]
```

Or better: auto-detect from the pretrained model name:

```python
if config.backbone == "pretrained":
    tokenizer = get_tokenizer(name=config.pretrained_model,
                              vocab_size=config.transformer.vocab_size)
```

### Step 6: Optimizer Adjustments

**File:** `memoria/training/optimizer.py` — `_setup_pretrained_optimizer()`

Changes for LFM2:
1. Backbone is still fully frozen (same as Qwen)
2. Interface LR can be slightly higher (0.002 vs 0.001) since the backbone is smaller and more stable
3. Read gate LR stays at 10× interface LR
4. Device batch size can be 2-4 (350M backbone fits easily)

### Step 7: Training Phase Adjustments

**File:** `memoria/training/train.py`

For LFM2 the backbone already handles language. Phase 1 should be minimal:

```python
# LFM2: backbone knows language, skip to cognitive awakening fast
phase1_steps=200,        # vs 500 for Qwen, 2000 for scratch
alpha_warmup_steps=300,  # vs 500 for Qwen, 3000 for scratch
```

Also: `torch.compile` for LFM2 layers. The hybrid architecture may need
per-layer compilation:

```python
if self._backbone_type == "lfm2":
    for i, layer in enumerate(backbone_model.layers):
        backbone_model.layers[i] = torch.compile(layer)
```

### Step 8: TTT Compatibility

The In-Place TTT module applies low-rank deltas to MLP projection layers.
LFM2's MLP is `Lfm2MLP` with `w1`, `w2`, `w3` (SwiGLU). The TTT deltas
apply to the hidden state AFTER the layer, not inside it, so no changes needed.

The TTT quality gate uses RND surprise from the Telos module — also unchanged.

The only adjustment: TTT layer selection. Default is top 25% of layers
(layers 12-15 for 16 layers). For LFM2, we want TTT on the upper
**attention** layers specifically:

```python
# Default for LFM2: TTT on the last 3 attention layers [10, 12, 14]
ttt_layers = [10, 12, 14]  # upper attention layers only
```

Conv layers have kernel=3 local context — TTT deltas on them would only
adapt local patterns, not the global representations we care about.

### Step 9: Working Memory / Refinement Loops

These features are currently only in `MemoriaModel` (scratch mode), not
`PretrainedMemoriaModel`. For the LFM2 integration:

- **Working memory suffix**: Can be added to PretrainedMemoriaModel. Append M
  learnable tokens after the real tokens. LFM2 handles variable-length sequences fine.
- **Refinement loops**: Re-run the last attention block cycle (layers 13-14 + interface 5).
  The conv layers in between can be skipped during refinement (local patterns
  don't change with re-reading beliefs).
- **Engram cache**: Works unchanged — hash-based lookup at layer 0.

These are optional enhancements for after the basic integration is validated.

### Step 10: Checkpoint Handling

LFM2 backbone weights should NOT be saved in checkpoints (same as Qwen):

```python
# Already handled in save_checkpoint():
if hasattr(model, 'backbone'):
    model_state = {k: v for k, v in model_state.items()
                   if not k.startswith('backbone.')}
```

On resume, the backbone is re-loaded from HuggingFace. Only interface layers
+ cognitive state are checkpointed.

---

## Implementation Order

```
Phase 1: Core Integration (get it running)
  1. Add lfm2_config() preset to config.py + __main__.py
  2. Refactor PretrainedMemoriaModel.__init__() for backbone detection
  3. Fix final norm (.embedding_norm vs .norm)
  4. Fix layer output handling (tensor vs tuple)
  5. Set interface_positions to attention layer indices [2,5,8,10,12,14]
  6. Update tokenizer auto-detection for pretrained mode
  7. Test: model builds, forward pass runs, loss computes

Phase 2: Training Validation (get it learning)
  8. Update optimizer for LFM2 param groups
  9. Adjust training phases (short phase 1)
  10. Run 100 steps, verify loss decreases and beliefs populate
  11. Verify belief advantage measurement works

Phase 3: Optimizations (get it fast)
  12. torch.compile per-layer
  13. TTT layer selection (attention layers only)
  14. Conservative read gate init (-8.0)
  15. Gradient checkpointing strategy (checkpoint conv layers, not attention)

Phase 4: Enhancements (port from scratch model)
  16. Working memory suffix
  17. Refinement loops (attention layers only)
  18. Engram cache
```

---

## Estimated Parameter Counts

```
LFM2.5-350M backbone:         350M  (frozen)
6 interface layers:             15M  (trainable)
  - ReadPath (per layer):       ~1.6M (query_proj + output_proj + utility + gate + conv)
  - WritePath (per layer):      ~0.8M (obs_proj + gate + precision)
  - Norm + depth_bias:          ~0.01M
Read gates:                     6    (trainable)
Cognitive state:                ~4M  (beliefs + edges + goals, updated by pass2)
  - 16K beliefs × 256:         4M
  - 65K edges × 64:            4M
  - 256 goals × 256:           65K
MetaParams:                     15   (trainable)
TelosModule:                    ~0.5M (trainable)
EdgeProposer:                   ~0.3M (trainable)
CognitiveController:            ~0.05M (trainable)
TTT deltas (3 layers):          ~0.2M (updated by TTT, not optimizer)
─────────────────────────────────────────
Total trainable:                ~16M
Total frozen:                   350M
Cognitive state (pass2):        ~8M
```

~16M trainable on top of 350M frozen. **Training cost ~20x cheaper than the scratch 245M model** since we don't train the backbone and device_batch_size can be 2-4.

---

## Step 11: Merge & Push to HuggingFace

**Target repo:** `Mazino0/memoria-prototype`

After training, merge everything into a single self-contained HF model that
anyone can load with `AutoModelForCausalLM.from_pretrained()`.

### What gets merged

```
LFM2 backbone (frozen, bf16):     ~700MB
Interface layers (bf16):            ~32MB
Read gates:                         24 bytes
TTT deltas (bf16):                  ~1MB
TelosModule:                        ~1MB
EdgeProposer:                       ~0.6MB
CognitiveController:                ~0.1MB
MetaParams:                         ~60 bytes
Cognitive state (PolarQuant 3-bit): ~5MB
──────────────────────────────────────────
Total model:                       ~740MB
```

### Custom modeling file

Create `modeling_memoria.py` that:
1. Subclasses or wraps `Lfm2ForCausalLM`
2. Adds interface layers, cognitive state, TTT as persistent modules
3. Overrides `forward()` to route through interfaces
4. Overrides `generate()` to keep cognitive state alive across generate steps
5. Registers as a custom AutoModel so `from_pretrained` works

```python
# Usage after push:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Mazino0/memoria-prototype",
    trust_remote_code=True,  # loads our custom modeling_memoria.py
)
tokenizer = AutoTokenizer.from_pretrained("Mazino0/memoria-prototype")

# Model self-improves on every input:
for doc in documents:
    output = model.generate(tokenizer(doc, return_tensors="pt").input_ids)
    # cognitive state + TTT deltas updated internally
    
# Save evolved state for next session:
model.save_pretrained("./evolved-checkpoint")
```

### Repo structure on HuggingFace

```
Mazino0/memoria-prototype/
├── config.json              # LFM2 config + Memoria extensions
├── modeling_memoria.py      # Custom model class (loaded via trust_remote_code)
├── configuration_memoria.py # Custom config class
├── model.safetensors        # Merged weights (backbone + interfaces + TTT)
├── cognitive_state.pt       # Compressed cognitive state (PolarQuant)
├── tokenizer.json           # LFM2 tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md                # Model card with architecture diagram, benchmarks
```

### Push script (runs at end of training)

```python
def push_merged_model(model, tokenizer, repo_id, token):
    """Merge backbone + adapters + state into a single HF model."""
    from huggingface_hub import HfApi
    import tempfile, json, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save merged model weights (backbone + interface + TTT)
        merged_state = {}
        # Backbone
        for k, v in model.backbone.state_dict().items():
            merged_state[f'backbone.{k}'] = v
        # Everything else (interfaces, gates, telos, edge_proposal, controller, ttt)
        for k, v in model.state_dict().items():
            if not k.startswith('backbone.'):
                merged_state[k] = v

        # Save as safetensors
        from safetensors.torch import save_file
        save_file(merged_state, os.path.join(tmpdir, "model.safetensors"))

        # 2. Save cognitive state (compressed)
        torch.save(
            model.state.state_dict_cognitive(compress=True),
            os.path.join(tmpdir, "cognitive_state.pt"),
        )

        # 3. Save tokenizer
        tokenizer.save_pretrained(tmpdir)

        # 4. Save config (LFM2 base + Memoria extensions)
        config = model.backbone.config.to_dict()
        config['memoria'] = {
            'interface_positions': model.interface_positions,
            'belief_dim': model.config.state.belief_dim,
            'max_beliefs': model.config.state.max_beliefs,
            'max_edges': model.config.state.max_edges,
            'max_goals': model.config.state.max_goals,
            'num_interfaces': len(model.interfaces),
            'ttt_layers': sorted(model.ttt.ttt_layers),
        }
        config['auto_map'] = {
            'AutoModelForCausalLM': 'modeling_memoria.MemoriaForCausalLM',
            'AutoConfig': 'configuration_memoria.MemoriaConfig',
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 5. Copy modeling + config code
        # These files define the custom classes for trust_remote_code loading
        import shutil
        shutil.copy("memoria/model/hf_export/modeling_memoria.py", tmpdir)
        shutil.copy("memoria/model/hf_export/configuration_memoria.py", tmpdir)

        # 6. Push everything
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Memoria-LFM2: merged model + cognitive state",
        )
    
    print(f"Pushed to https://huggingface.co/{repo_id}")
```

### What `save_pretrained` does after inference (preserving evolved state)

```python
def save_pretrained(self, path):
    """Save model with CURRENT cognitive state (evolved from use)."""
    # Weights don't change at inference (backbone frozen, interfaces frozen)
    # But cognitive state + TTT deltas DO change with every input
    super().save_pretrained(path)
    torch.save(
        self.state.state_dict_cognitive(compress=True),
        os.path.join(path, "cognitive_state.pt"),
    )
    # Next load picks up the evolved state
```

This means: train once → push to HF → anyone downloads → uses it → saves →
the model they saved is **better** than the one they downloaded, because the
cognitive state evolved during use.

### Implementation order for Step 11

```
11a. Create memoria/model/hf_export/modeling_memoria.py
     - MemoriaForCausalLM wrapping LFM2 + interfaces + state
     - forward(), generate(), save_pretrained(), from_pretrained()
     
11b. Create memoria/model/hf_export/configuration_memoria.py
     - MemoriaConfig extending LFM2 config with Memoria fields

11c. Add push_merged_model() to training/train.py
     - Called at end of training (replaces current _push_to_hub_async)

11d. Test: push to Mazino0/memoria-prototype, load back, verify forward pass
```

---

## Risk Factors

1. **Conv layers may not benefit from interface reads.** If the conv layers
   immediately overwrite the belief-enriched representation with local patterns,
   the interface contribution gets washed out. Mitigation: interfaces are placed
   AFTER attention (which sees everything), and the next conv only mixes locally
   within the attention's global representation.

2. **Weight tying (lm_head ↔ embed_tokens).** TTT uses `lm_head.weight.data`
   for gradient computation. Since it's tied, modifying one modifies the other.
   This is fine for `.data` access (TTT reads it, doesn't write to it), but
   verify no edge case.

3. **350M may be too small.** If the backbone can't produce useful hidden
   representations for the interface layers to project from, the cognitive state
   won't help. Mitigation: LFM2.5 at 350M is VERY well-trained (28T tokens) —
   its representations should be rich. The belief advantage measurement will
   tell us quickly.

4. **Hybrid cache incompatibility.** For inference with KV caching, the custom
   `Lfm2HybridConvCache` may interact poorly with interface layer injection.
   For training (no cache), this is a non-issue. Address in inference optimization
   phase.
