"""Distributed training setup for multi-GPU via DDP.

Architecture:
- Transformer + interface weights: replicated via DDP (gradients synced automatically)
- Cognitive state: lives on rank 0, broadcast to all ranks before each step
- Write candidates: gathered to rank 0 after forward pass
- Pass 2: runs on rank 0 only, then state is broadcast again

Single-GPU mode: all DDP ops become no-ops, no overhead.
"""

import os
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training if available.

    Returns:
        (rank, world_size, device)
    """
    if 'RANK' in os.environ and torch.cuda.device_count() > 1:
        # Launched via torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        if rank == 0:
            print(f"DDP initialized: {world_size} GPUs")

        return rank, world_size, device
    else:
        # Single GPU or CPU
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        return 0, 1, device


def cleanup_distributed():
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, device: torch.device, rank: int, world_size: int) -> tuple[nn.Module, nn.Module]:
    """Wrap model in DDP if distributed.

    The cognitive state is excluded from DDP by setting requires_grad=False
    (already done). DDP only syncs gradient-bearing parameters.

    Args:
        model: MemoriaModel on device
        device: this rank's device
        rank: this process rank
        world_size: total processes

    Returns:
        (wrapped_model, base_model) — wrapped for forward, base for state access
    """
    model = model.to(device)
    base_model = model

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True, static_graph=True)

    return model, base_model


def broadcast_state(state, rank: int, world_size: int):
    """Broadcast cognitive state from rank 0 to all ranks.

    State is small (~4MB at max capacity) so this is cheap.
    Only needed in multi-GPU mode.
    """
    if world_size <= 1:
        return

    # Broadcast all state tensors from rank 0
    dist.broadcast(state.beliefs.data, src=0)
    dist.broadcast(state.edge_src, src=0)
    dist.broadcast(state.edge_tgt, src=0)
    dist.broadcast(state.edge_relations.data, src=0)
    dist.broadcast(state.edge_weights.data, src=0)
    dist.broadcast(state.edge_active, src=0)
    dist.broadcast(state.goal_embeddings.data, src=0)
    dist.broadcast(state.goal_metadata.data, src=0)
    dist.broadcast(state.meta.data, src=0)
    dist.broadcast(state.belief_last_accessed, src=0)
    dist.broadcast(state.belief_access_count, src=0)
    dist.broadcast(state.belief_prev_surprise, src=0)
    dist.broadcast(state.edge_causal_obs, src=0)


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
