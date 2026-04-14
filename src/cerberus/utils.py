"""Shared utilities for Cerberus."""

from __future__ import annotations

import importlib
import re
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


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Auto-detection order: CUDA, MPS (Apple Silicon), CPU.

    Args:
        device: Device string (e.g. ``"cuda"``, ``"cpu"``, ``"cuda:0"``,
            ``"mps"``).  ``None`` or ``"auto"`` triggers auto-detection.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
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
