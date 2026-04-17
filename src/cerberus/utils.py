"""Shared utilities for Cerberus."""

from __future__ import annotations

import importlib
from typing import Any


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
