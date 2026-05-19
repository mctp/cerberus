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

import cerberus.models as _cerberus_models
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
    assert _cerberus_models.ChromBPNet is ChromBPNet


# ---------------------------------------------------------------------------
# Regression guards: architecture defaults, buffer wiring, calibration helper.
# ---------------------------------------------------------------------------


def test_chrombpnet_default_constructor_uses_module_defaults():
    """``ChromBPNet()`` builds a 512-filter accessibility + 128-filter bias.

    Pins the ``_ACCESSIBILITY_DEFAULTS`` / ``_BIAS_DEFAULTS`` values that match
    the reference chrombpnet-pytorch architecture.  Accidental edits to those
    dicts surface here rather than silently shipping a different model.
    """
    model = ChromBPNet(input_len=128, output_len=64)
    # First conv filter count: (out_channels, in_channels=4, kernel_size).
    # ``iconv`` is annotated as nn.Module in BPNet (so weight_norm can replace
    # it), hence the explicit type-ignore on .weight access.
    assert model.accessibility_model.iconv.weight.shape[0] == 512  # type: ignore[union-attr]
    assert model.bias_model.iconv.weight.shape[0] == 128  # type: ignore[union-attr]
    # Tower depth.
    assert len(model.accessibility_model.res_layers) == 8
    assert len(model.bias_model.res_layers) == 1 * 4


def test_chrombpnet_user_args_override_defaults():
    """``accessibility_args`` / ``bias_args`` win over the module-level dicts."""
    model = ChromBPNet(
        input_len=128, output_len=64,
        accessibility_args={"filters": 16, "n_dilated_layers": 3},
        bias_args={"filters": 8, "n_dilated_layers": 2},
    )
    assert model.accessibility_model.iconv.weight.shape[0] == 16  # type: ignore[union-attr]
    assert model.bias_model.iconv.weight.shape[0] == 8  # type: ignore[union-attr]
    assert len(model.accessibility_model.res_layers) == 3
    assert len(model.bias_model.res_layers) == 2


def test_chrombpnet_shared_kwargs_override_user_duplicates():
    """Constructor-injected ``input_len`` / ``output_len`` win over user values.

    Without this, a user-supplied ``accessibility_args={"input_len": 999}``
    could give the two branches different input dimensions and silently
    break the forward pass at shape-time.
    """
    model = ChromBPNet(
        input_len=128, output_len=64,
        accessibility_args={"input_len": 999, "filters": 8, "n_dilated_layers": 2,
                            "conv_kernel_size": 11, "profile_kernel_size": 11},
        bias_args={"input_len": 777, "filters": 4, "n_dilated_layers": 1,
                   "conv_kernel_size": 11, "profile_kernel_size": 11},
    )
    assert model.accessibility_model.input_len == 128
    assert model.bias_model.input_len == 128
    assert model.accessibility_model.output_len == 64
    assert model.bias_model.output_len == 64


def test_chrombpnet_bias_logcount_offset_round_trips_in_state_dict():
    """The offset survives ``state_dict()`` / ``load_state_dict()``.

    Guards against the buffer being accidentally demoted to a plain attribute,
    which would silently drop the offset on any checkpoint reload.
    """
    src = _make_small_model()
    src.bias_logcount_offset.fill_(0.42)  # type: ignore[union-attr]

    dst = _make_small_model()
    dst.load_state_dict(src.state_dict())

    assert dst.bias_logcount_offset.item() == pytest.approx(0.42)  # type: ignore[union-attr]


def test_chrombpnet_bias_logcount_offset_moves_with_model_to():
    """The buffer follows ``model.to(device)`` (a Parameter or buffer would; a
    plain Python float would not)."""
    model = _make_small_model()
    assert isinstance(model.bias_logcount_offset, torch.Tensor)
    model.to(torch.device("cpu"))
    assert model.bias_logcount_offset.device.type == "cpu"
    # Buffer, not Parameter -- it must not appear in parameters().
    assert all(p is not model.bias_logcount_offset for p in model.parameters())


def test_estimate_bias_logcount_offset_restores_training_mode():
    """Helper must restore the bias model's ``training`` flag on exit."""
    bias_model = _ConstantCountBiasModel(log_count=0.0)
    bias_model.train()
    batch = {
        "inputs": torch.randn(1, 4, 32),
        "targets": torch.ones(1, 1, 4),
    }
    estimate_bias_logcount_offset(bias_model, [batch], device="cpu")
    assert bias_model.training is True


def test_estimate_bias_logcount_offset_raises_on_empty_dataloader():
    """Helper must surface the ``no batches consumed`` failure mode rather
    than returning NaN / zero."""
    bias_model = _ConstantCountBiasModel(log_count=0.0)
    with pytest.raises(ValueError, match="No batches were available"):
        estimate_bias_logcount_offset(bias_model, [], device="cpu")


def test_estimate_bias_logcount_offset_honors_max_batches():
    """``max_batches`` caps consumption -- avoids running the full epoch when
    a small calibration sample is sufficient."""
    bias_model = _ConstantCountBiasModel(log_count=0.0)
    # Two batches with different observed counts; with max_batches=1 only
    # the first contributes to the mean.
    batch1 = {
        "inputs": torch.randn(1, 4, 32),
        "targets": torch.full((1, 1, 4), 0.0),  # sum=0 -> log(0+1)=0
    }
    batch2 = {
        "inputs": torch.randn(1, 4, 32),
        "targets": torch.full((1, 1, 4), 99.0),  # sum=396 -> log(397)
    }
    delta_capped = estimate_bias_logcount_offset(
        bias_model, [batch1, batch2], device="cpu", max_batches=1,
    )
    delta_full = estimate_bias_logcount_offset(
        bias_model, [batch1, batch2], device="cpu",
    )
    assert delta_capped == pytest.approx(0.0)
    assert delta_full > delta_capped
