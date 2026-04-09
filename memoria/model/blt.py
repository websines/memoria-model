"""Byte Latent Transformer (BLT) — tokenizer-free byte-level I/O for Memoria.

Replaces the 151K-token embedding/LM head with a byte-level encoder/decoder.
The global DeltaProduct backbone operates on patches (compressed byte groups),
while lightweight local encoder/decoder handle byte↔patch conversion.

Architecture:
  bytes [B, T] → ByteEncoder → patches [B, P, D] → Global Backbone → [B, P, D] → ByteDecoder → byte_logits [B, T, 260]

Why this matters:
- LM head drops from 117M params (151K vocab) to 197K params (260 byte vocab)
- Eliminates softmax bottleneck (768-dim can easily represent 260 distributions)
- Gradient pass-through: 31% (was 0.5% with 151K vocab)
- Per-token bandwidth: LM head goes from 233 MB to 0.4 MB
- No tokenizer artifacts — model sees raw bytes, learns its own representations
- DeltaProduct's O(T) scaling handles 4-6x longer byte sequences naturally

Local layers use FLA's GatedDeltaProduct (same kernel as global backbone) with:
- Fewer Householder reflections (1 vs 3) — local context is simpler
- Smaller hidden dim (local_dim vs n_embd) — bytes need less capacity
- 2 layers each (encoder + decoder) — byte patterns are shallow

Strided Conv1d pooling compresses bytes to patches (each patch = patch_size bytes).
The decoder expands patches back and adds encoder skip connections for byte detail.
Multi-byte prediction heads enable DFlash-style speculative decoding at byte level.

Reference: BLT (Meta, arXiv:2412.09871) — byte latent transformer
Reference: MambaByte (arXiv:2401.13660) — byte-level SSM
Reference: EvaByte (HKU NLP, 2025) — linear attention + bytes + multi-byte heads
Reference: Bolmo (Allen AI, arXiv:2512.15586) — mLSTM byte encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import FLA's GatedDeltaProduct for O(T) local layers
_fla_available = False
try:
    from fla.layers import GatedDeltaProduct
    _fla_available = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Local DeltaProduct block — lightweight, for byte-level processing
# ═══════════════════════════════════════════════════════════════════════════

class LocalDeltaProduct(nn.Module):
    """Lightweight DeltaProduct for byte-level sequences.

    Same kernel as the global backbone but configured for local context:
    - 1 Householder reflection (vs 3 in global) — simpler patterns
    - Smaller hidden dim — bytes don't need as much capacity
    - No RoPE needed — DeltaProduct uses recurrent state for ordering

    Falls back to causal conv + MLP on CPU (no FLA Triton kernels).
    """

    def __init__(self, dim: int, head_dim: int = 64, layer_idx: int = 0):
        super().__init__()
        self._use_fla = _fla_available
        if _fla_available:
            self.dp = GatedDeltaProduct(
                hidden_size=dim,
                head_dim=head_dim,
                num_heads=dim // head_dim,
                expand_v=2,
                num_householder=1,       # simpler than global's 3
                allow_neg_eigval=True,
                use_forget_gate=True,
                use_short_conv=True,
                conv_size=4,
                use_output_gate=True,
                mode='chunk',
                layer_idx=layer_idx,
            )
        else:
            # CPU/test fallback: causal conv + gated MLP (same interface)
            self.fallback = nn.Sequential(
                nn.Conv1d(dim, dim, 4, padding=3, groups=dim),
                nn.GELU(),
            )
            self.gate = nn.Linear(dim, dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self._use_fla:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out, _, _ = self.dp(x)
            return out.to(x.dtype)
        else:
            # Fallback: causal conv + gated output
            T = x.shape[1]
            h = self.fallback(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
            return self.proj(h * self.gate(x).sigmoid())


# ═══════════════════════════════════════════════════════════════════════════
# Byte N-gram convolution — learned local byte context (replaces EngramCache)
# ═══════════════════════════════════════════════════════════════════════════

class ByteNgramConv(nn.Module):
    """Causal convolution capturing byte N-gram patterns.

    Replaces hash-table N-gram lookup (EngramCache) with a learned causal
    convolution. At byte-level, N-grams up to 8 bytes cover common patterns
    (words, UTF-8 sequences, common prefixes). Depthwise-separable for
    efficiency.

    Equivalent to BLT's hash N-gram embeddings but learned end-to-end.
    """

    def __init__(self, dim: int, kernel_size: int = 8):
        super().__init__()
        # Depthwise causal conv (each channel independently)
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size - 1,  # causal: only look backward
            groups=dim,
        )
        # Pointwise mixing
        self.pw_conv = nn.Conv1d(dim, dim, 1)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, D] → [B, T, D]"""
        T = x.shape[1]
        h = x.transpose(1, 2)                      # [B, D, T]
        h = self.dw_conv(h)[:, :, :T]              # causal trim
        h = F.gelu(h)
        h = self.pw_conv(h)                         # [B, D, T]
        h = h.transpose(1, 2)                       # [B, T, D]
        return self.norm(h)


# ═══════════════════════════════════════════════════════════════════════════
# Byte Encoder — bytes → patches for the global model
# ═══════════════════════════════════════════════════════════════════════════

class ByteEncoder(nn.Module):
    """Encode raw bytes into patch representations for the global backbone.

    Pipeline:
    1. Byte embedding (260 classes: 256 bytes + BOS/EOS/PAD/SEP)
    2. Causal N-gram convolution (local byte context, 8-byte receptive field)
    3. N DeltaProduct layers (O(T) causal processing of byte sequence)
    4. Strided Conv1d pooling (compress patch_size bytes → 1 patch)
    5. Project up to global dimension

    Returns both patches (for global model) and byte_hidden (skip connection
    for the decoder — preserves byte-level detail that pooling loses).
    """

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        patch_size: int,
        n_layers: int,
        byte_vocab: int = 260,
        head_dim: int = 64,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.patch_size = patch_size
        self.byte_vocab = byte_vocab

        # 1. Byte embedding
        self.byte_embed = nn.Embedding(byte_vocab, local_dim)

        # 2. Causal N-gram convolution (local context)
        self.ngram_conv = ByteNgramConv(local_dim, kernel_size=8)

        # 3. DeltaProduct local layers
        self.layers = nn.ModuleList([
            LocalDeltaProduct(local_dim, head_dim, layer_idx=i)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.RMSNorm(local_dim) for _ in range(n_layers)
        ])

        # 4. Strided pooling: compress patch_size bytes → 1 patch
        # Conv1d kernel learns which byte features to extract per patch
        self.patch_pool = nn.Conv1d(
            local_dim, local_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # 5. Project to global backbone dimension
        self.up_proj = nn.Linear(local_dim, global_dim, bias=False)
        self.out_norm = nn.RMSNorm(global_dim)

        # Initialize: small weights for gradual activation
        nn.init.normal_(self.up_proj.weight, std=0.02)

    def forward(self, byte_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode bytes to patches.

        Args:
            byte_ids: [B, T_bytes] raw byte values (0-259)

        Returns:
            patches: [B, P, global_dim] for global backbone (P = ceil(T/patch_size))
            byte_hidden: [B, T_bytes, local_dim] encoder output (decoder skip conn)
        """
        B, T = byte_ids.shape

        # Byte embedding + N-gram context
        x = self.byte_embed(byte_ids)                # [B, T, local_dim]
        x = x + self.ngram_conv(x)                   # residual N-gram features

        # DeltaProduct local layers (O(T), causal)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        byte_hidden = x                              # save for decoder

        # Pad to multiple of patch_size for clean pooling
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))        # pad sequence dim

        # Strided pooling → patches
        patches = self.patch_pool(x.transpose(1, 2))  # [B, local_dim, P]
        patches = patches.transpose(1, 2)              # [B, P, local_dim]

        # Project up to global dim
        patches = self.up_proj(patches)                # [B, P, global_dim]
        patches = self.out_norm(patches)

        return patches, byte_hidden


# ═══════════════════════════════════════════════════════════════════════════
# Byte Decoder — patches → byte predictions
# ═══════════════════════════════════════════════════════════════════════════

class ByteDecoder(nn.Module):
    """Decode patch representations back to byte-level predictions.

    Pipeline:
    1. Project patches down to local dimension
    2. Expand to byte positions (repeat_interleave by patch_size)
    3. Add intra-patch positional encoding (position within each patch)
    4. Add encoder skip connection (byte_hidden from ByteEncoder)
    5. N DeltaProduct layers (O(T) causal refinement)
    6. Multi-byte prediction heads (for DFlash speculative decoding)

    Head 0 predicts the next byte at each position (standard autoregressive).
    Heads 1..N predict bytes 2..N+1 ahead (for multi-byte DFlash speculation).
    """

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        patch_size: int,
        n_layers: int,
        byte_vocab: int = 260,
        n_byte_heads: int = 1,
        head_dim: int = 64,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.patch_size = patch_size
        self.byte_vocab = byte_vocab

        # 1. Project down from global to local
        self.down_proj = nn.Linear(global_dim, local_dim, bias=False)

        # 2. Intra-patch positional encoding
        self.intra_pos = nn.Embedding(patch_size, local_dim)

        # 3. Skip connection gate (blend encoder bytes + decoded patches)
        self.skip_gate = nn.Linear(local_dim * 2, local_dim, bias=False)

        # 4. DeltaProduct local layers
        self.layers = nn.ModuleList([
            LocalDeltaProduct(local_dim, head_dim, layer_idx=100 + i)  # offset idx
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.RMSNorm(local_dim) for _ in range(n_layers)
        ])

        # 5. Output norm
        self.out_norm = nn.RMSNorm(local_dim)

        # 6. Multi-byte prediction heads
        # Head 0: predict next byte (standard)
        # Head k: predict byte k+1 ahead (for multi-step DFlash)
        self.byte_heads = nn.ModuleList([
            nn.Linear(local_dim, byte_vocab, bias=False)
            for _ in range(n_byte_heads)
        ])

        # Initialize
        nn.init.normal_(self.down_proj.weight, std=0.02)
        for head in self.byte_heads:
            nn.init.normal_(head.weight, std=0.01)

    def forward(
        self,
        patch_hidden: Tensor,
        byte_hidden: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Decode patches to byte predictions.

        Args:
            patch_hidden: [B, P, global_dim] from global backbone
            byte_hidden: [B, T_bytes, local_dim] from encoder (skip connection)

        Returns:
            logits: [B, T_bytes, byte_vocab] primary byte predictions (head 0)
            all_logits: list of [B, T_bytes, byte_vocab] for each head
        """
        B, T_bytes, _ = byte_hidden.shape

        # Project down
        x = self.down_proj(patch_hidden)              # [B, P, local_dim]

        # Expand to byte positions
        x = x.repeat_interleave(self.patch_size, dim=1)  # [B, P*ps, local_dim]
        x = x[:, :T_bytes, :]                         # trim to actual byte count

        # Intra-patch positional encoding
        pos_ids = torch.arange(T_bytes, device=x.device) % self.patch_size
        x = x + self.intra_pos(pos_ids)

        # Gated skip connection: blend decoded patches with encoder byte features
        combined = torch.cat([x, byte_hidden], dim=-1)  # [B, T, 2*local_dim]
        x = self.skip_gate(combined)                    # [B, T, local_dim]

        # DeltaProduct local layers (causal refinement)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        x = self.out_norm(x)

        # Multi-byte prediction
        all_logits = [head(x) for head in self.byte_heads]

        return all_logits[0], all_logits

    def compute_loss(
        self,
        patch_hidden: Tensor,
        byte_hidden: Tensor,
        target_bytes: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute byte prediction loss (all heads).

        Args:
            patch_hidden: [B, P, global_dim]
            byte_hidden: [B, T_bytes, local_dim]
            target_bytes: [B, T_bytes] target byte values

        Returns:
            loss: scalar (sum of all head losses, head k targets byte k+1 ahead)
            stats: dict with per-head loss values
        """
        logits_primary, all_logits = self.forward(patch_hidden, byte_hidden)
        B, T, V = logits_primary.shape
        stats = {}

        # Head 0: standard next-byte loss
        loss = F.cross_entropy(
            logits_primary.reshape(-1, V),
            target_bytes.reshape(-1),
            ignore_index=-1,
        )
        stats['loss_byte_h0'] = loss.item()

        # Heads 1+: multi-byte ahead prediction (for DFlash multi-step)
        for k in range(1, len(all_logits)):
            # Head k predicts byte k positions ahead
            if T > k:
                shifted_targets = target_bytes[:, k:]    # [B, T-k]
                head_logits = all_logits[k][:, :-k, :]   # [B, T-k, V]
                head_loss = F.cross_entropy(
                    head_logits.reshape(-1, V),
                    shifted_targets.reshape(-1),
                    ignore_index=-1,
                )
                loss = loss + head_loss
                stats[f'loss_byte_h{k}'] = head_loss.item()

        # Average over number of heads
        loss = loss / len(all_logits)
        stats['loss_byte_total'] = loss.item()

        return loss, stats
