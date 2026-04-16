"""Distributed ops for cognitive state (not handled by Accelerate).

Accelerate handles DDP model wrapping, gradient sync, and device placement.
This module provides the cognitive-state-specific distributed operations:
- broadcast_state: rank 0 → all ranks before each forward pass
- gather_candidates: all ranks → rank 0 after forward pass
- gather_read_indices: all ranks → rank 0 for Hebbian learning
- sync_ranks: barrier after Pass 2

Single-GPU mode: all ops are no-ops, no overhead.
"""

import hashlib
import torch
import torch.distributed as dist


def _state_fingerprint(state) -> int:
    """Compute a deterministic fingerprint of (name, shape, dtype) across all
    recursive buffers and parameters of `state`. Used to detect cross-rank
    structural drift BEFORE we issue any broadcasts — a mismatch here would
    otherwise produce a mid-iteration NCCL hang that's painful to debug.
    """
    parts = []
    for name, buf in sorted(state.named_buffers()):
        if buf is None:
            continue
        parts.append(f"B:{name}:{tuple(buf.shape)}:{buf.dtype}")
    for name, param in sorted(state.named_parameters()):
        if param is None:
            continue
        parts.append(f"P:{name}:{tuple(param.shape)}:{param.dtype}")
    joined = "|".join(parts)
    # Use SHA-256 truncated to 63 bits so it fits in int64 comfortably.
    h = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _state_content_fingerprint(state, device: torch.device) -> torch.Tensor:
    """Compute a content fingerprint over all recursive buffers/parameters.

    Returns a single float64 scalar tensor on `device`. Walks tensors in the
    same sorted order as the broadcast loop and accumulates
    `tensor.to(float64).sum()`. After a successful `dist.broadcast(src=0)`,
    every rank holds bit-identical data, so the fingerprints must be
    bit-identical too (no FP-precision tolerance needed).

    Catches content drift introduced AFTER the broadcast — e.g., a future
    code path that writes to state on rank 0 only and forgets to re-sync.
    Without this post-check, that kind of drift is invisible until it
    causes a downstream collective hang or, worse, silently trained-in
    rank-divergent behaviour.
    """
    acc = torch.zeros(1, dtype=torch.float64, device=device)
    for _, buf in sorted(state.named_buffers()):
        if buf is None:
            continue
        acc += buf.detach().to(torch.float64).sum().view(1)
    for _, param in sorted(state.named_parameters()):
        if param is None:
            continue
        acc += param.detach().to(torch.float64).sum().view(1)
    return acc


def broadcast_state(state, rank: int, world_size: int):
    """Broadcast cognitive state from rank 0 to all ranks.

    Recursively walks ALL buffers and parameters of `state` (including nested
    submodules like `state.telos`, `state.controller`, `state.safety_gate`,
    `state.srwm`, `state.adaptive_depth`), in a deterministic sorted order,
    so the broadcast sequence is identical on every rank regardless of dict
    insertion order. Before issuing any broadcasts, all ranks exchange a
    structural fingerprint via all_reduce so any cross-rank divergence (buffer
    added/removed, shape mismatch, dtype change) raises a clear error instead
    of hanging mid-broadcast.

    Historical note: the previous implementation iterated only `state._buffers`
    and `state._parameters` (top-level only), which (a) silently failed to sync
    buffers inside submodules like SRWM.W_fast and controller.reward_mean, and
    (b) used a `if buf is not None` skip that made the emitted broadcast count
    content-dependent. Both were fragile under DDP and contributed to the NCCL
    BROADCAST watchdog timeouts seen in multi-rank runs.

    State is small (~4MB at max capacity) so the extra collectives are cheap.
    Only needed in multi-GPU mode.
    """
    if world_size <= 1:
        return

    device = next(
        (p.device for _, p in state.named_parameters() if p is not None),
        torch.device("cpu"),
    )

    # ── Pre-flight symmetry check ────────────────────────────────────────────
    # Every rank computes its own fingerprint, then we all_reduce(MAX) and
    # all_reduce(MIN) so any divergence shows up as MAX != MIN.
    fp_local = _state_fingerprint(state)
    fp_max = torch.tensor([fp_local], dtype=torch.int64, device=device)
    fp_min = fp_max.clone()
    dist.all_reduce(fp_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(fp_min, op=dist.ReduceOp.MIN)
    if fp_max.item() != fp_min.item():
        # Dump this rank's view to help diagnose the mismatch.
        names = []
        for n, b in sorted(state.named_buffers()):
            if b is not None:
                names.append(f"B:{n}:{tuple(b.shape)}:{b.dtype}")
        for n, p in sorted(state.named_parameters()):
            if p is not None:
                names.append(f"P:{n}:{tuple(p.shape)}:{p.dtype}")
        raise RuntimeError(
            f"[broadcast_state] Cross-rank state structure mismatch on rank {rank}: "
            f"fingerprint min={fp_min.item()} max={fp_max.item()}. "
            f"Rank {rank} sees {len(names)} items: {names[:8]}..."
        )

    # ── Deterministic-order broadcast ────────────────────────────────────────
    # Barrier before ensures all ranks reach this point together, so any later
    # hang is clearly attributable to the broadcast loop itself and not to
    # earlier drift. Barrier after ensures the next training-loop stage doesn't
    # race ahead while collectives are still draining.
    dist.barrier()

    with torch.no_grad():
        for _, buf in sorted(state.named_buffers()):
            if buf is not None:
                dist.broadcast(buf, src=0)
        for _, param in sorted(state.named_parameters()):
            if param is not None:
                # In-place broadcast into the parameter's underlying storage.
                # No `.data` rebind — we want the existing storage preserved so
                # DDP gradient buckets and optimizer state keep their references.
                dist.broadcast(param, src=0)

    dist.barrier()

    # ── Post-broadcast content verification ──────────────────────────────────
    # After a successful broadcast(src=0) chain, every rank holds bit-identical
    # state. We verify this by computing a float64 content fingerprint and
    # checking MAX == MIN across ranks. The check is paranoia for the current
    # implementation (we KNOW broadcasts are copies), but it's load-bearing as
    # a regression trap: if a future code path starts mutating state between
    # broadcast_state and the next sync point on some ranks only, this assert
    # turns that silent drift into a clear, early failure at the next step.
    with torch.no_grad():
        content_local = _state_content_fingerprint(state, device)
    content_max = content_local.clone()
    content_min = content_local.clone()
    dist.all_reduce(content_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(content_min, op=dist.ReduceOp.MIN)
    if content_max.item() != content_min.item():
        # Identify which specific tensors contain NaN/Inf for debugging.
        nan_tensors = []
        with torch.no_grad():
            for name, buf in sorted(state.named_buffers()):
                if buf is not None and not torch.isfinite(buf).all():
                    n_bad = (~torch.isfinite(buf)).sum().item()
                    nan_tensors.append(f"buffer:{name}[{tuple(buf.shape)}] ({n_bad} bad)")
            for name, param in sorted(state.named_parameters()):
                if param is not None and not torch.isfinite(param.data).all():
                    n_bad = (~torch.isfinite(param.data)).sum().item()
                    nan_tensors.append(f"param:{name}[{tuple(param.data.shape)}] ({n_bad} bad)")
        raise RuntimeError(
            f"[broadcast_state] Content drift detected on rank {rank} AFTER "
            f"broadcast: content_fingerprint min={content_min.item():.6e} "
            f"max={content_max.item():.6e}. "
            f"Non-finite tensors: {nan_tensors if nan_tensors else 'none (drift is real, not NaN)'}. "
            f"Check for rank-conditional state writes between "
            f"broadcast_state() calls — those must either happen on all ranks "
            f"or be followed by an explicit re-broadcast."
        )


def sanitize_state_nan(state, step: int = -1) -> list[str]:
    """Replace NaN/Inf in all state buffers and parameters with zeros.

    Called after pass2 (rank 0 only) to prevent NaN from propagating through
    broadcast_state → all ranks on the next step. Returns the names of
    tensors that were sanitized so the caller can log them.

    Why zeros: a zeroed belief is "inactive" (radius < EPSILON), a zeroed
    edge weight is dead (collected on next sweep), and a zeroed buffer is
    neutral. This is always safer than keeping NaN, which would cascade
    into every downstream computation.
    """
    sanitized = []
    with torch.no_grad():
        for name, buf in state.named_buffers():
            if buf is None:
                continue
            mask = ~torch.isfinite(buf)
            if mask.any():
                n_bad = mask.sum().item()
                buf[mask] = 0.0
                sanitized.append(f"buffer:{name} ({n_bad} values)")
        for name, param in state.named_parameters():
            if param is None:
                continue
            mask = ~torch.isfinite(param.data)
            if mask.any():
                n_bad = mask.sum().item()
                param.data[mask] = 0.0
                sanitized.append(f"param:{name} ({n_bad} values)")
    if sanitized and step >= 0:
        import sys
        print(
            f"\n  WARNING: NaN/Inf sanitized in cognitive state at step {step}: "
            f"{sanitized}",
            file=sys.stderr,
        )
    return sanitized


def sync_ranks(world_size: int):
    """Barrier to synchronize all ranks. No-op for single GPU."""
    if world_size > 1:
        dist.barrier()


def gather_candidates(packed_candidates: torch.Tensor, rank: int, world_size: int, belief_dim: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Gather packed candidate tensors from all ranks to rank 0.

    Args:
        packed_candidates: [N_local, D+3] tensor from this rank
        rank: this process rank
        world_size: total processes
        belief_dim: belief dimension (D)
        device: CUDA device for NCCL collectives. Falls back to packed_candidates.device.

    Returns:
        On rank 0: [N_total, D+3] concatenated candidates from all ranks.
        On other ranks: empty tensor (they don't need it).
    """
    if world_size <= 1:
        return packed_candidates

    cols = belief_dim + 3

    # Ensure tensors are on the NCCL-compatible device (empty packs may arrive on CPU)
    if device is not None and packed_candidates.device != torch.device(device):
        packed_candidates = packed_candidates.to(device)

    # First, share candidate counts so we can allocate receive buffers
    local_count = torch.tensor([packed_candidates.shape[0]], device=packed_candidates.device)
    all_counts = [torch.zeros(1, device=packed_candidates.device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)

    max_count = max(c.item() for c in all_counts)

    if max_count == 0:
        return packed_candidates

    # Pad to max_count so all ranks have same shape (required by all_gather)
    padded = torch.zeros(max_count, cols, device=packed_candidates.device)
    if packed_candidates.shape[0] > 0:
        padded[:packed_candidates.shape[0]] = packed_candidates

    # Gather all padded tensors
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank == 0:
        # Unpad and concatenate
        parts = []
        for i, g in enumerate(gathered):
            n = all_counts[i].item()
            if n > 0:
                parts.append(g[:n])
        if parts:
            return torch.cat(parts, dim=0)
        return torch.zeros(0, cols, device=packed_candidates.device)
    else:
        return torch.zeros(0, cols, device=packed_candidates.device)


def gather_read_indices(read_indices: list[int], rank: int, world_size: int, device: torch.device) -> list[int]:
    """Gather read belief indices from all ranks to rank 0.

    Each rank captures which beliefs its local read path retrieved.
    For correct Hebbian learning, rank 0 needs the union of all retrievals.

    Args:
        read_indices: local read indices from this rank
        rank: this process rank
        world_size: total processes
        device: this rank's device

    Returns:
        On rank 0: deduplicated union of all ranks' read indices.
        On other ranks: empty list.
    """
    if world_size <= 1:
        return read_indices

    # Share counts
    local_t = torch.tensor(read_indices, dtype=torch.long, device=device)
    local_count = torch.tensor([len(read_indices)], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)

    max_count = max(c.item() for c in all_counts)
    if max_count == 0:
        return []

    # Pad to max_count
    padded = torch.full((max_count,), -1, dtype=torch.long, device=device)
    if len(read_indices) > 0:
        padded[:len(read_indices)] = local_t

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank == 0:
        all_indices = set()
        for i, g in enumerate(gathered):
            n = all_counts[i].item()
            if n > 0:
                all_indices.update(g[:n].tolist())
        return list(all_indices)
    else:
        return []


def setup_device() -> torch.device:
    """Get the best available device (backward compat for single-GPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
