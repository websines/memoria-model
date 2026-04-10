"""Distributed ops for cognitive state (not handled by Accelerate).

Accelerate handles DDP model wrapping, gradient sync, and device placement.
This module provides the cognitive-state-specific distributed operations:
- broadcast_state: rank 0 → all ranks before each forward pass
- gather_candidates: all ranks → rank 0 after forward pass
- gather_read_indices: all ranks → rank 0 for Hebbian learning
- sync_ranks: barrier after Pass 2

Single-GPU mode: all ops are no-ops, no overhead.
"""

import torch
import torch.distributed as dist


def broadcast_state(state, rank: int, world_size: int):
    """Broadcast cognitive state from rank 0 to all ranks.

    State is small (~4MB at max capacity) so this is cheap.
    Only needed in multi-GPU mode.
    """
    if world_size <= 1:
        return

    # Broadcast all state tensors from rank 0.
    # Uses programmatic iteration to avoid missing newly-added buffers.
    # nn.Parameters use .data to bypass autograd versioning.
    for name, buf in state._buffers.items():
        if buf is not None:
            dist.broadcast(buf, src=0)
    for name, param in state._parameters.items():
        if param is not None:
            dist.broadcast(param.data, src=0)


def sync_ranks(world_size: int):
    """Barrier to synchronize all ranks. No-op for single GPU."""
    if world_size > 1:
        dist.barrier()


def gather_candidates(packed_candidates: torch.Tensor, rank: int, world_size: int, belief_dim: int) -> torch.Tensor:
    """Gather packed candidate tensors from all ranks to rank 0.

    Args:
        packed_candidates: [N_local, D+3] tensor from this rank
        rank: this process rank
        world_size: total processes
        belief_dim: belief dimension (D)

    Returns:
        On rank 0: [N_total, D+3] concatenated candidates from all ranks.
        On other ranks: empty tensor (they don't need it).
    """
    if world_size <= 1:
        return packed_candidates

    cols = belief_dim + 3

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
