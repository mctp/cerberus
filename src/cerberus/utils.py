"""Shared utilities for Cerberus."""

from __future__ import annotations

import importlib
import logging
import os
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)


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


def _local_rank_cuda_device() -> torch.device | None:
    """Return ``cuda:LOCAL_RANK`` when the env var picks a valid GPU.

    DDP launchers (``torchrun``, ``srun``) set ``LOCAL_RANK`` in each
    rank's environment. Bare ``torch.device("cuda")`` resolves to
    ``cuda:0`` on every rank until ``torch.cuda.set_device`` is called
    (Lightning does this inside ``trainer.fit``), so pre-Trainer GPU
    work would otherwise pile onto card 0.

    Returns ``None`` if ``LOCAL_RANK`` is unset, non-integer, or out of
    range for the visible CUDA devices; misconfigured values log a
    warning and the caller falls back to bare ``cuda``.
    """
    raw = os.environ.get("LOCAL_RANK")
    if raw is None:
        return None
    try:
        rank = int(raw)
    except ValueError:
        logger.warning(
            "LOCAL_RANK=%r is not an integer; defaulting to cuda:0", raw,
        )
        return None
    count = torch.cuda.device_count()
    if not 0 <= rank < count:
        logger.warning(
            "LOCAL_RANK=%d is out of range for %d visible CUDA device(s); "
            "defaulting to cuda:0",
            rank, count,
        )
        return None
    device = torch.device(f"cuda:{rank}")
    logger.debug("resolve_device: LOCAL_RANK=%d → %s", rank, device)
    return device


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Auto-detection order: CUDA, MPS (Apple Silicon), CPU. Under a DDP
    launcher (``LOCAL_RANK`` set), CUDA auto-detection picks
    ``cuda:LOCAL_RANK`` so pre-Trainer GPU work spreads across cards.

    Args:
        device: Device string (e.g. ``"cuda"``, ``"cpu"``, ``"cuda:0"``,
            ``"mps"``).  ``None`` or ``"auto"`` triggers auto-detection.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return _local_rank_cuda_device() or torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_use_folds(use_folds: str | None) -> list[str] | None:
    """Parse a ``--use_folds`` CLI argument into a list of fold roles.

    Accepts comma- or plus-separated values (e.g. ``"test+val"``,
    ``"test,val"``).  The special value ``"all"`` expands to
    ``["train", "test", "val"]``.  Duplicates are removed.

    Args:
        use_folds: Raw CLI string, or ``None``.

    Returns:
        List of unique fold role strings, or ``None`` if input is ``None``.
    """
    if use_folds is None:
        return None
    folds: list[str] = []
    for part in re.split(r"[+,]", use_folds):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            folds.extend(["train", "test", "val"])
        else:
            folds.append(part)
    return list(dict.fromkeys(folds)) or None


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
    if precision not in ("full", "mps", "bf16"):
        raise ValueError(
            f"precision must be one of 'full', 'mps', 'bf16'; got {precision!r}"
        )

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
