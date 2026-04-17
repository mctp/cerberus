"""Shared utilities for Cerberus."""

from __future__ import annotations

import importlib
from typing import Any

import torch


def import_class(name: str) -> Any:
    """Dynamically import a class from a dotted module path.

    Args:
        name: Fully qualified class name (e.g. ``'cerberus.models.bpnet.BPNet'``).

    Raises:
        ImportError: If the module or class cannot be found.
    """
    try:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{name}': {e}") from e


def resolve_device(device_arg: str | None = "auto") -> torch.device:
    """Resolve a user-provided device string or choose the best available device.

    Args:
        device_arg: Explicit device string such as ``"cpu"``, ``"cuda:0"``, or
            ``"mps"``. ``None`` and ``"auto"`` both trigger auto-detection.

    Returns:
        A concrete :class:`torch.device`.
    """
    if device_arg in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_precision_kwargs(
    precision: str,
    accelerator: str,
    devices: str | int,
    *,
    use_ddp_find_unused_parameters_false: bool = True,
) -> dict[str, Any]:
    """Return standardized trainer precision kwargs for Cerberus train tools.

    The training CLIs share the same three precision modes:

    - ``full``: float32 with highest matmul precision, no compile
    - ``mps``: fp16 mixed precision for Apple Silicon, no compile
    - ``bf16``: bf16 mixed precision for NVIDIA, with compile enabled

    For multi-GPU CUDA runs, most tools also opt into
    ``ddp_find_unused_parameters_false`` to avoid Lightning's slower default DDP
    behavior. Some scripts, such as BiasNet, intentionally keep ``strategy`` as
    ``"auto"``; callers can disable the DDP override with
    ``use_ddp_find_unused_parameters_false=False``.
    """
    if precision == "full":
        return {
            "precision": "32-true",
            "matmul_precision": "highest",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }

    if precision == "mps":
        return {
            "precision": "16-mixed",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }

    multi_gpu = (
        use_ddp_find_unused_parameters_false
        and accelerator == "gpu"
        and isinstance(devices, int)
        and devices > 1
    )
    return {
        "precision": "bf16-mixed",
        "matmul_precision": "medium",
        "accelerator": accelerator,
        "devices": devices,
        "strategy": "ddp_find_unused_parameters_false" if multi_gpu else "auto",
        "benchmark": True,
        "compile": True,
    }
