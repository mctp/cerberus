"""Tests for the ChromBPNet model and its bias-count calibration helper.

Covers the architectural composition and forward math of
:class:`cerberus.models.chrombpnet.ChromBPNet` plus the standalone
calibration helper :func:`estimate_bias_logcount_offset`.

The bias-loading-and-freezing test path (loading a pre-trained bias
``state_dict`` and freezing it via ``ModelConfig.freeze``) is exercised
in the stage-2 trainer's deferred work, not here.
"""

import pytest
import torch
import torch.nn as nn

from cerberus.models import ChromBPNet
from cerberus.models.chrombpnet import estimate_bias_logcount_offset
from cerberus.output import ProfileCountOutput
from cerberus.utils import import_class


def _make_small_model() -> ChromBPNet:
    """Tiny ChromBPNet for fast tests; preserves the two-branch composition."""
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


class _FixedBranch(nn.Module):
    """Returns hardcoded logits/log_counts for math-pinning tests."""

    def __init__(self, logits: torch.Tensor, log_counts: torch.Tensor):
        super().__init__()
        # Plain attributes (not buffers); these test tensors don't need to
        # move with .to() and aren't part of any state_dict round-trip.
        self.fixed_logits = logits
        self.fixed_log_counts = log_counts

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        batch = x.shape[0]
        logits = self.fixed_logits.expand(batch, -1, -1)
        log_counts = self.fixed_log_counts.expand(batch, -1)
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


def test_chrombpnet_combines_logits_and_counts_exactly():
    """Forward math: ``acc.logits + bias.logits`` and
    ``logaddexp(acc.log_counts, bias.log_counts + offset)``."""
    model = _make_small_model()
    # Replace the real BPNet sub-modules with fixed-output stubs to pin the
    # combining math without depending on the inner BPNet's behavior.
    model.accessibility_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[1.0, 2.0, 3.0]]]),
        log_counts=torch.tensor([[0.5]]),
    )
    model.bias_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[0.2, -0.5, 1.1]]]),
        log_counts=torch.tensor([[1.5]]),
    )
    model.bias_logcount_offset.fill_(0.25)  # type: ignore[operator]

    out = model(torch.randn(2, 4, 128))
    expected_logits = torch.tensor([[[1.2, 1.5, 4.1]]]).expand(2, -1, -1)
    expected_log_counts = torch.logaddexp(
        torch.full((2, 1), 0.5),
        torch.full((2, 1), 1.75),
    )
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)


class _ConstantCountBiasModel(nn.Module):
    """Tiny bias model whose log-count output is a single learnable scalar."""

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
    """The returned scalar is the mean residual
    ``log(observed + pc) - log(predicted)``."""
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
    # Both rows sum to 10; observed log-counts are log(10 + 1).  Predicted
    # log-counts are zero (the constant log_count parameter).  Mean residual
    # = log(11) - 0 = log(11).
    expected = torch.log(torch.tensor([11.0, 11.0])).mean().item()
    assert delta == pytest.approx(expected, rel=1e-6)


def test_chrombpnet_importable_from_class_path():
    """``cerberus.utils.import_class`` round-trip (used by ModelConfig)."""
    cls = import_class("cerberus.models.chrombpnet.ChromBPNet")
    assert cls is ChromBPNet


def test_chrombpnet_importable_from_models_shortcut():
    """``from cerberus.models import ChromBPNet`` resolves to the same class."""
    from cerberus.models import ChromBPNet as Imported
    assert Imported is ChromBPNet
