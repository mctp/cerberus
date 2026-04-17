"""Minimal helpers for integrating the upstream ``tpcav`` package with Cerberus.

This module intentionally stays thin:

- checkpoint loading remains in :mod:`cerberus.model_ensemble`
- scalar target semantics reuse :class:`cerberus.attribution.AttributionTarget`
- concept generation and TCAV scoring remain delegated to the upstream
  ``tpcav`` package

The Cerberus-owned responsibilities here are:

- restricting the MVP to supported model families
- exposing stable, Cerberus-friendly probe-layer aliases
- wrapping Cerberus model outputs into a scalar module consumable by Captum/TPCAV
"""

from __future__ import annotations

import torch.nn as nn

from cerberus.attribution import AttributionTarget
from cerberus.models.bpnet import BPNet


def build_tpcav_target_model(
    model: nn.Module,
    *,
    mode: str = "log_counts",
    channel: int = 0,
    bin_index: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> AttributionTarget:
    """Wrap a supported Cerberus model for TPCAV/Captum.

    The initial MVP intentionally supports only single-task BPNet models.
    Upstream ``tpcav`` expects a scalar-output ``nn.Module``; Cerberus models
    return structured ``ProfileCountOutput`` objects, so we reuse
    :class:`AttributionTarget` as the adapter.
    """
    if not isinstance(model, BPNet) or getattr(model, "n_output_channels", None) != 1:
        raise TypeError(
            "Minimal TPCAV integration currently supports only single-task BPNet models."
        )

    return AttributionTarget(
        model=model,
        mode=mode,
        channel=channel,
        bin_index=bin_index,
        window_start=window_start,
        window_end=window_end,
    )


def list_tpcav_probe_layers(model: nn.Module) -> dict[str, str]:
    """Return Cerberus-friendly probe-layer aliases for supported models.

    The returned values are module paths as seen by ``named_modules()``, which
    is the format expected by upstream ``tpcav.TPCAV(layer_name=...)``.
    """
    base_model, prefix = _unwrap_tpcav_model(model)
    if not isinstance(base_model, BPNet) or getattr(base_model, "n_output_channels", None) != 1:
        raise TypeError(
            "Minimal TPCAV integration currently supports only single-task BPNet models."
        )

    layers: dict[str, str] = {
        "initial_conv": f"{prefix}iconv",
        "profile_head": f"{prefix}profile_conv",
        "count_head": f"{prefix}count_dense",
    }

    for i in range(len(base_model.res_layers)):
        layers[f"tower_block_{i}"] = f"{prefix}res_layers.{i}"

    if base_model._apply_final_tower_relu:
        layers["tower_out"] = f"{prefix}final_tower_relu"
    else:
        layers["tower_out"] = f"{prefix}res_layers.{len(base_model.res_layers) - 1}"

    if base_model._activate_iconv_before_tower:
        layers["initial_activation"] = f"{prefix}iconv_act"

    return layers


def resolve_tpcav_layer_name(model: nn.Module, layer: str) -> str:
    """Resolve a user-facing alias or raw module path to a TPCAV layer name."""
    aliases = list_tpcav_probe_layers(model)
    if layer in aliases:
        return aliases[layer]

    module_names = {name for name, _ in model.named_modules()}
    if layer in module_names:
        return layer

    available = ", ".join(sorted(aliases))
    raise ValueError(
        f"Unknown TPCAV probe layer {layer!r}. Available aliases: {available}"
    )
def _unwrap_tpcav_model(model: nn.Module) -> tuple[nn.Module, str]:
    if isinstance(model, AttributionTarget):
        return model.model, "model."
    return model, ""
