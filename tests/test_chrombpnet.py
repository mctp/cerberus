"""Tests for the native ChromBPNet implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cerberus.config import PretrainedConfig
from cerberus.models import ChromBPNet
from cerberus.models.chrombpnet import estimate_bias_logcount_offset
from cerberus.output import ProfileCountOutput
from cerberus.pretrained import load_pretrained_weights
from cerberus.utils import import_class


def _make_small_model() -> ChromBPNet:
    return ChromBPNet(
        input_len=128,
        output_len=64,
        output_channels=["signal"],
        accessibility_args={
            "filters": 8,
            "n_dilated_layers": 2,
            "conv_kernel_size": 11,
            "dil_kernel_size": 3,
            "profile_kernel_size": 11,
            "residual_architecture": "residual_post-activation_conv",
        },
        bias_args={
            "filters": 4,
            "n_dilated_layers": 1,
            "conv_kernel_size": 11,
            "dil_kernel_size": 3,
            "profile_kernel_size": 11,
            "residual_architecture": "residual_post-activation_conv",
        },
    )


def test_chrombpnet_forward_shape():
    model = _make_small_model()
    x = torch.randn(2, 4, 128)
    out = model(x)
    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (2, 1, 64)
    assert out.log_counts.shape == (2, 1)


def test_chrombpnet_reference_aliases():
    model = _make_small_model()
    assert model.chrombpnet_wo_bias is model.accessibility_model
    assert model.bias is model.bias_model


class _FixedBranch(nn.Module):
    def __init__(self, logits: torch.Tensor, log_counts: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)
        self.register_buffer("_log_counts", log_counts)

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        batch = x.shape[0]
        logits = self._logits.expand(batch, -1, -1)
        log_counts = self._log_counts.expand(batch, -1)
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


def test_chrombpnet_combines_logits_and_counts_exactly():
    model = _make_small_model()
    model.accessibility_model = _FixedBranch(
        logits=torch.tensor([[[1.0, 2.0, 3.0]]]),
        log_counts=torch.tensor([[0.5]]),
    )
    model.bias_model = _FixedBranch(
        logits=torch.tensor([[[0.2, -0.5, 1.1]]]),
        log_counts=torch.tensor([[1.5]]),
    )
    model.set_bias_logcount_offset(0.25)

    out = model(torch.randn(2, 4, 128))
    expected_logits = torch.tensor([[[1.2, 1.5, 4.1]]]).expand(2, -1, -1)
    expected_log_counts = torch.logaddexp(
        torch.full((2, 1), 0.5),
        torch.full((2, 1), 1.75),
    )
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)


class _ConstantCountBiasModel(nn.Module):
    def __init__(self, log_count: float):
        super().__init__()
        self.log_count = nn.Parameter(torch.tensor([log_count], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        batch = x.shape[0]
        return ProfileCountOutput(
            logits=torch.zeros(batch, 1, 4, device=x.device),
            log_counts=self.log_count.view(1, 1).expand(batch, 1),
        )


def test_estimate_bias_logcount_offset_matches_mean_residual():
    bias_model = _ConstantCountBiasModel(log_count=0.0)
    batch = {
        "inputs": torch.randn(2, 4, 32),
        "targets": torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[4.0, 3.0, 2.0, 1.0]],
            ]
        ),
    }
    delta = estimate_bias_logcount_offset(
        bias_model,
        [batch],
        count_pseudocount=1.0,
        device="cpu",
    )
    expected = torch.log(torch.tensor([11.0, 11.0])).mean().item()
    assert delta == pytest.approx(expected, rel=1e-6)


def test_pretrained_bias_loads_and_freezes(tmp_path):
    source = _make_small_model()
    with torch.no_grad():
        for p in source.bias_model.parameters():
            p.fill_(0.25)

    weights_path = tmp_path / "bias.pt"
    torch.save(source.bias_model.state_dict(), weights_path)

    target = _make_small_model()
    load_pretrained_weights(
        target,
        [
            PretrainedConfig(
                weights_path=str(weights_path),
                source=None,
                target="bias_model",
                freeze=True,
            )
        ],
    )

    for src, dst in zip(
        source.bias_model.parameters(), target.bias_model.parameters(), strict=True
    ):
        assert torch.allclose(src, dst)
        assert dst.requires_grad is False

    assert any(p.requires_grad for p in target.accessibility_model.parameters())


def test_chrombpnet_importable_from_class_path():
    cls = import_class("cerberus.models.chrombpnet.ChromBPNet")
    model = cls()
    assert isinstance(model, ChromBPNet)


def test_chrombpnet_importable_from_models_shortcut():
    from cerberus.models import ChromBPNet as Imported

    assert Imported is ChromBPNet
