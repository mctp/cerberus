"""Pretrained weight loading utilities for cerberus models."""

import logging

import torch
import torch.nn as nn

from cerberus.config import PretrainedConfig

logger = logging.getLogger(__name__)


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    """Unwrap a torch.compile'd model to get the original module."""
    return getattr(model, "_orig_mod", model)


def _extract_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Extract keys matching a prefix and strip it.

    Given a state dict and a prefix like ``"bias_model"``, returns only the
    keys starting with ``"bias_model."`` with the prefix (and dot) removed.
    This allows loading a sub-module's weights from a full model checkpoint.

    Args:
        state_dict: Source state dict.
        prefix: Sub-module name to extract (dot is appended automatically).

    Returns:
        State dict with matching keys, prefix stripped.

    Raises:
        ValueError: If no keys match the prefix.
    """
    dot_prefix = prefix + "."
    extracted = {
        k[len(dot_prefix) :]: v
        for k, v in state_dict.items()
        if k.startswith(dot_prefix)
    }
    if not extracted:
        available = sorted(state_dict.keys())[:10]
        raise ValueError(
            f"No keys found with prefix {dot_prefix!r} in state dict. "
            f"Available keys (first 10): {available}"
        )
    return extracted


def load_pretrained_weights(
    model: nn.Module,
    pretrained: list[PretrainedConfig],
) -> None:
    """Load pretrained weights into a model or its sub-modules.

    Each entry in ``pretrained`` specifies:

    - **weights_path**: Path to a ``.pt`` state dict file.
    - **source**: Which keys to extract from the file. ``None`` uses all keys.
      A string like ``"bias_model"`` extracts only keys with that prefix
      (and strips it), enabling loading a sub-module from a full-model checkpoint.
    - **target**: Where to load in the model. ``None`` loads into the whole model.
      A string like ``"bias_model"`` loads into that named sub-module.
    - **freeze**: If ``True``, set ``requires_grad=False`` on all loaded parameters.

    Common patterns::

        # biasnet.pt → Dalmatian's bias_model
        {"weights_path": "biasnet.pt", "source": None, "target": "bias_model", "freeze": True}

        # dalmatian.pt's bias_model → new Dalmatian's bias_model
        {"weights_path": "dalmatian.pt", "source": "bias_model", "target": "bias_model", "freeze": True}

        # dalmatian.pt → full Dalmatian (re-initialize everything)
        {"weights_path": "dalmatian.pt", "source": None, "target": None, "freeze": False}

    Uses ``strict=True`` so architecture mismatches raise immediately.
    Handles ``torch.compile`` transparently via ``_orig_mod`` unwrapping.

    Args:
        model: The model to load weights into (compiled or not).
        pretrained: List of pretrained weight configs.
    """
    target_root = _unwrap_compiled(model)

    for cfg in pretrained:
        weights_path = cfg.weights_path
        source = cfg.source
        target_name = cfg.target
        freeze = cfg.freeze

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Extract sub-tree from source checkpoint if specified
        if source is not None:
            state_dict = _extract_prefix(state_dict, source)

        # Resolve target module
        if target_name is not None:
            target = getattr(target_root, target_name)
        else:
            target = target_root

        target.load_state_dict(state_dict, strict=True)

        n_params = sum(p.numel() for p in target.parameters())
        label = target_name or model.__class__.__name__

        if freeze:
            for p in target.parameters():
                p.requires_grad_(False)
            logger.info(
                "Loaded and froze pretrained %s (%s params) from %s",
                label,
                f"{n_params:,}",
                weights_path,
            )
        else:
            logger.info(
                "Loaded pretrained %s (%s params) from %s",
                label,
                f"{n_params:,}",
                weights_path,
            )
