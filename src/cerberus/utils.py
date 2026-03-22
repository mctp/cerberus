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
