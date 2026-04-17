"""Tests for the minimal Cerberus TPCAV integration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cerberus.attribution import AttributionTarget
from cerberus.model_ensemble import resolve_fold_dir
from cerberus.models.bpnet import BPNet
from cerberus.tpcav import (
    build_tpcav_target_model,
    list_tpcav_probe_layers,
    resolve_tpcav_layer_name,
)


def _make_bpnet(
    residual_architecture: str = "residual_pre-activation_conv",
) -> BPNet:
    return BPNet(
        input_len=50,
        output_len=30,
        filters=8,
        n_dilated_layers=2,
        conv_kernel_size=5,
        dil_kernel_size=3,
        profile_kernel_size=5,
        output_channels=["signal"],
        predict_total_count=True,
        residual_architecture=residual_architecture,
    )


def test_build_tpcav_target_model_wraps_bpnet_log_counts() -> None:
    model = _make_bpnet()
    target_model = build_tpcav_target_model(model)

    assert isinstance(target_model, AttributionTarget)

    x = torch.rand(2, 4, 50)
    out = target_model(x)
    assert out.shape == (2,)


def test_list_tpcav_probe_layers_uses_final_relu_for_default_bpnet() -> None:
    model = _make_bpnet("residual_pre-activation_conv")
    target_model = build_tpcav_target_model(model)

    layers = list_tpcav_probe_layers(target_model)

    assert layers["tower_out"] == "model.final_tower_relu"
    assert layers["initial_conv"] == "model.iconv"
    assert layers["tower_block_0"] == "model.res_layers.0"
    assert layers["tower_block_1"] == "model.res_layers.1"
    assert layers["profile_head"] == "model.profile_conv"
    assert layers["count_head"] == "model.count_dense"


def test_list_tpcav_probe_layers_uses_last_block_for_post_activation_bpnet() -> None:
    model = _make_bpnet("residual_post-activation_conv")
    target_model = build_tpcav_target_model(model)

    layers = list_tpcav_probe_layers(target_model)

    assert layers["tower_out"] == "model.res_layers.1"


def test_resolve_tpcav_layer_name_accepts_alias_and_raw_module_path() -> None:
    model = _make_bpnet()
    target_model = build_tpcav_target_model(model)

    assert (
        resolve_tpcav_layer_name(target_model, "tower_out")
        == "model.final_tower_relu"
    )
    assert resolve_tpcav_layer_name(target_model, "model.iconv") == "model.iconv"


def test_resolve_tpcav_layer_name_rejects_unknown_layer() -> None:
    model = _make_bpnet()
    target_model = build_tpcav_target_model(model)

    with pytest.raises(ValueError, match="Unknown TPCAV probe layer"):
        resolve_tpcav_layer_name(target_model, "not_a_real_layer")


def test_build_tpcav_target_model_rejects_non_bpnet_models() -> None:
    with pytest.raises(TypeError, match="single-task BPNet"):
        build_tpcav_target_model(torch.nn.Linear(4, 1))


def test_resolve_fold_dir_finds_nested_fold_directory(tmp_path: Path) -> None:
    fold_dir = tmp_path / "nested" / "single-fold" / "fold_0"
    fold_dir.mkdir(parents=True)
    assert resolve_fold_dir(tmp_path, 0) == fold_dir
