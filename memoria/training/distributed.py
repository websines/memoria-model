"""Distributed training setup for 2x 3090 DataParallel.

DataParallel splits the batch across GPUs for the forward pass.
The cognitive state stays on GPU 0 (not replicated).
Pass 2 runs on GPU 0 only (state updates are sequential).
"""

import torch
import torch.nn as nn


def setup_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def setup_data_parallel(model: nn.Module) -> tuple[nn.Module, bool]:
    """Wrap model in DataParallel if multiple GPUs available.

    Args:
        model: MemoriaModel

    Returns:
        (model, is_parallel) — model possibly wrapped, flag for unwrapping later
    """
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = model.cuda()
        model = nn.DataParallel(model)
        return model, True
    elif torch.cuda.is_available():
        model = model.cuda()
        return model, False
    return model, False


def unwrap_model(model: nn.Module, is_parallel: bool):
    """Get the underlying model from DataParallel wrapper."""
    if is_parallel:
        return model.module
    return model


def get_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
