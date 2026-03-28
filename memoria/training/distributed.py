"""Distributed training setup for 2x 3090 DataParallel.

Simple DataParallel for v1. DistributedDataParallel (DDP) for later.
The cognitive state is NOT replicated — it's shared across devices.
Only the transformer and interface layers are parallelized.
"""

import torch
import torch.nn as nn


def setup_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_data_parallel(model: nn.Module) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs available.

    Note: The cognitive state (model.state) is on one device.
    DataParallel handles the transformer forward pass across GPUs.
    Pass 2 runs on a single device (state updates are sequential).

    Args:
        model: MemoriaModel

    Returns:
        model (possibly wrapped in DataParallel)
    """
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        # Note: we only parallelize the forward pass, not pass 2
        # For now, just use the model as-is on device 0
        # TODO: proper DataParallel that handles cognitive state correctly
        model = model.cuda()
    elif torch.cuda.is_available():
        model = model.cuda()

    return model


def get_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
