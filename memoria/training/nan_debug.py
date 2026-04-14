"""NaN/Inf localization for training forward passes.

Enabled via env var MEMORIA_DEBUG_NAN=1. Registers a forward hook on every
submodule; on the first NaN/Inf observed in a module output, prints the
fully-qualified module name + tensor stats and raises to halt training.

Intentionally separate from the post-loss NaN guard in train.py:
  - train.py guard detects NaN AFTER forward completes (too late to localize)
  - this tripwire detects NaN DURING forward at the producing module

The first module to trip is the producer; ancestor modules would trip later
because they consume the NaN'd output. We raise on the first trip.
"""

import os
import torch
import torch.nn as nn


class _Tripped(RuntimeError):
    pass


def _tensor_stats(t: torch.Tensor) -> str:
    with torch.no_grad():
        finite = torch.isfinite(t)
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        if finite.any():
            ft = t[finite]
            amin = ft.min().item()
            amax = ft.max().item()
            absmax = ft.abs().max().item()
        else:
            amin = amax = absmax = float("nan")
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"nan={n_nan} inf={n_inf} "
        f"min={amin:.3e} max={amax:.3e} absmax={absmax:.3e}"
    )


def _check(tag: str, name: str, obj) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not torch.isfinite(obj).all():
            raise _Tripped(f"[NaN tripwire] {tag} of '{name}' is non-finite: {_tensor_stats(obj)}")
    elif isinstance(obj, (tuple, list)):
        for i, x in enumerate(obj):
            _check(f"{tag}[{i}]", name, x)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check(f"{tag}[{k!r}]", name, v)


def install_nan_hooks(model: nn.Module) -> list:
    """Register NaN-detecting forward hooks on every submodule.

    Returns the list of hook handles so the caller can remove them.
    The hook raises _Tripped on first non-finite output — train.py catches
    it, prints context, and re-raises.
    """
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def hook(mod, inputs, output, _name=name):
            _check("output", _name, output)

        handles.append(module.register_forward_hook(hook))
    return handles


def enabled() -> bool:
    return os.environ.get("MEMORIA_DEBUG_NAN", "0") == "1"


# Re-export the exception so callers can `except nan_debug.Tripped`
Tripped = _Tripped
