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

    The hook checks BOTH inputs and outputs so we can distinguish:
      - input clean + output NaN → this module is the producer
      - input NaN already        → bug is upstream (possibly a raw function
        call invisible to hooks, e.g. a Triton kernel)
    On trip, also dumps the module's own parameter stats so corrupted
    `weight` / `bias` is visible in the same error message.
    """
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue

        def hook(mod, inputs, output, _name=name, _mod=module):
            # Check inputs first — if they're already NaN, bug is upstream.
            try:
                _check("input", _name, inputs)
            except _Tripped as e:
                # Annotate with this module's param health for context.
                param_stats = _param_summary(_mod)
                raise _Tripped(f"{e}\n  (at module '{_name}'; params: {param_stats})")
            # Input is clean — check output.
            try:
                _check("output", _name, output)
            except _Tripped as e:
                param_stats = _param_summary(_mod)
                raise _Tripped(f"{e}\n  (producer '{_name}'; params: {param_stats})")

        handles.append(module.register_forward_hook(hook))
    return handles


def _param_summary(module: nn.Module) -> str:
    """Summarize immediate (non-recursive) parameters' finiteness + range."""
    parts = []
    for pname, p in module.named_parameters(recurse=False):
        with torch.no_grad():
            n_nan = torch.isnan(p).sum().item()
            n_inf = torch.isinf(p).sum().item()
            finite = p[torch.isfinite(p)]
            if finite.numel() > 0:
                absmax = finite.abs().max().item()
            else:
                absmax = float("nan")
        parts.append(f"{pname} nan={n_nan} inf={n_inf} absmax={absmax:.3e}")
    return "; ".join(parts) if parts else "no direct params"


def enabled() -> bool:
    return os.environ.get("MEMORIA_DEBUG_NAN", "0") == "1"


# Re-export the exception so callers can `except nan_debug.Tripped`
Tripped = _Tripped
