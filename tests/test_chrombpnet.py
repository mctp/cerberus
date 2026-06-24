"""Tests for the ChromBPNet model and its bias-count calibration helper.

Covers the architectural composition and forward math of
:class:`cerberus.models.chrombpnet.ChromBPNet` plus the standalone
calibration helper :func:`estimate_bias_logcount_offset`.

The bias-loading-and-freezing test path (loading a pre-trained bias
``state_dict`` and freezing it via ``ModelConfig.freeze``) is exercised
in the stage-2 trainer's deferred work, not here.
"""

import logging
from typing import Any

import pytest
import torch
import torch.nn as nn

import cerberus.models as _cerberus_models
from cerberus.config import FreezeSpec, PretrainedConfig
from cerberus.freeze import apply_freeze
from cerberus.loss import MSEMultinomialLoss, ProfileJSDLoss
from cerberus.models import ChromBPNet, MultitaskChromBPNet
from cerberus.models.bpnet import BPNet, MultitaskBPNetLoss
from cerberus.models.chrombpnet import (
    _resolve_branch_shapes,
    estimate_bias_logcount_offset,
)
from cerberus.output import ProfileCountOutput
from cerberus.pretrained import load_pretrained_weights
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
    # Mutating the wrapper-level buffer here mirrors the post-forward
    # arithmetic; chrombpnet-pytorch achieves the same effect by mutating
    # bias_model.linear.bias before training.
    model.bias_logcount_offset.fill_(0.25)  # type: ignore[operator]

    out = model(torch.randn(2, 4, 128))
    expected_logits = torch.tensor([[[1.2, 1.5, 4.1]]]).expand(2, -1, -1)
    expected_log_counts = torch.logaddexp(
        torch.full((2, 1), 0.5),
        torch.full((2, 1), 1.75),
    )
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)


def test_chrombpnet_profile_only_bias_keeps_accessibility_counts():
    """bpAI-TAC-style mode adds bias logits but ignores bias log-counts."""
    model = _make_small_model()
    model.bias_count_mode = "profile_only"
    model.accessibility_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[1.0, 2.0, 3.0]]]),
        log_counts=torch.tensor([[0.5]]),
    )
    model.bias_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[0.2, -0.5, 1.1]]]),
        log_counts=torch.tensor([[100.0]]),
    )
    model.bias_logcount_offset.fill_(100.0)  # type: ignore[operator]

    out = model(torch.randn(2, 4, 128))
    expected_logits = torch.tensor([[[1.2, 1.5, 4.1]]]).expand(2, -1, -1)
    expected_log_counts = torch.full((2, 1), 0.5)
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)


def test_chrombpnet_rejects_unknown_bias_count_mode():
    with pytest.raises(ValueError, match="bias_count_mode"):
        ChromBPNet(
            input_len=128,
            output_len=64,
            bias_count_mode="not-a-mode",
        )


def test_profile_jsd_loss_ignores_count_head_gradients():
    logits = torch.tensor([[[2.0, 0.0, -1.0, 1.0]]], requires_grad=True)
    log_counts = torch.tensor([[123.0]], requires_grad=True)
    targets = torch.tensor([[[8.0, 1.0, 0.0, 3.0]]])
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)

    loss = ProfileJSDLoss()(outputs, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
    assert log_counts.grad is None


def test_profile_jsd_loss_leaves_bpnet_count_head_untrained():
    model = BPNet(
        input_len=128,
        output_len=64,
        output_channels=["signal"],
        filters=4,
        n_dilated_layers=1,
        conv_kernel_size=11,
        dil_kernel_size=3,
        profile_kernel_size=11,
        residual_architecture="residual_post-activation_conv",
    )
    outputs = model(torch.randn(2, 4, 128))
    targets = torch.rand_like(outputs.logits)

    loss = ProfileJSDLoss()(outputs, targets)
    loss.backward()

    assert model.profile_conv.weight.grad is not None
    assert torch.any(model.profile_conv.weight.grad != 0)
    assert model.count_dense.weight.grad is None
    assert model.count_dense.bias.grad is None


def test_profile_jsd_loss_uses_base2_divergence():
    """JSD must be computed in base 2.

    softmax(zero logits) = [0.5, 0.5], normalized targets ≈ [1, 0], so
    P = [1, 0], Q = [0.5, 0.5], M = [0.75, 0.25]:
        KL_2(P || M) = 1 * log2(1 / 0.75) = log2(4/3) ≈ 0.41504
        KL_2(Q || M) = 0.5 * log2(0.5/0.75) + 0.5 * log2(0.5/0.25)
                     = 0.5 * log2(2/3) + 0.5 * log2(2) ≈ 0.20752
        JSD_2       = 0.5 * (KL_2(P||M) + KL_2(Q||M))    ≈ 0.31128
    A natural-log regression would land on JSD_e ≈ 0.21576 instead.
    """
    outputs = ProfileCountOutput(
        logits=torch.zeros(1, 1, 2),
        log_counts=torch.zeros(1, 1),
    )
    targets = torch.tensor([[[1.0, 0.0]]])

    loss = ProfileJSDLoss()(outputs, targets)

    assert torch.allclose(loss, torch.tensor(0.31127813), atol=1e-6)


def test_profile_jsd_loss_handles_silent_targets():
    outputs = ProfileCountOutput(
        logits=torch.zeros(2, 1, 4),
        log_counts=torch.randn(2, 1),
    )
    targets = torch.zeros(2, 1, 4)
    loss = ProfileJSDLoss()(outputs, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


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


def test_pretrained_bias_loads_and_freezes_via_freeze_spec(tmp_path):
    """End-to-end bias-load + freeze recipe for the stage-2 trainer.

    Pins the freeze migration: ``PretrainedConfig.freeze=True`` was removed,
    so the equivalent recipe is now ``load_pretrained_weights`` for the
    bias subtree followed by ``apply_freeze`` with a ``FreezeSpec`` matching
    ``"bias_model"``.  After this:

    - bias parameters carry the loaded values bit-for-bit and have
      ``requires_grad=False``;
    - accessibility parameters are untouched and remain trainable;
    - ``bias_model`` is in ``.eval()`` mode (so Dropout / BatchNorm inside
      the frozen subtree stop firing).
    """
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
            )
        ],
    )
    apply_freeze(target, [FreezeSpec(pattern="bias_model", eval_mode=True)])

    for src, dst in zip(
        source.bias_model.parameters(), target.bias_model.parameters(), strict=True,
    ):
        assert torch.allclose(src, dst)
        assert dst.requires_grad is False

    assert all(p.requires_grad for p in target.accessibility_model.parameters())
    assert target.bias_model.training is False
    assert target.accessibility_model.training is True


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


# ---------------------------------------------------------------------------
# _resolve_branch_shapes — the shared-bias broadcasting helper used by
# ChromBPNet.forward to combine a 1-channel bias output with an N-channel
# accessibility output.
# ---------------------------------------------------------------------------


def test_resolve_branch_shapes_passthrough_when_shapes_match():
    branch = torch.randn(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    out = _resolve_branch_shapes("x", branch, target)
    # Identical object, not a copy: pass-through must not allocate.
    assert out is branch


def test_resolve_branch_shapes_broadcasts_singleton_channel():
    branch = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
    target = torch.zeros(1, 3, 4)
    out = _resolve_branch_shapes("x", branch, target)
    assert out.shape == (1, 3, 4)
    # Each output channel sees the same broadcast value.
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 1], out[:, 2])
    assert torch.equal(out[0, 0], branch[0, 0])


def test_resolve_branch_shapes_broadcasts_log_count_rank():
    # log_counts has shape (B, C) -- two-rank tensors must also broadcast.
    branch = torch.tensor([[7.0]])  # (1, 1)
    target = torch.zeros(1, 3)
    out = _resolve_branch_shapes("counts", branch, target)
    assert out.shape == (1, 3)
    assert torch.all(out == 7.0)


def test_resolve_branch_shapes_raises_on_incompatible_channels():
    """A 2-channel branch can't broadcast to a 3-channel target."""
    branch = torch.zeros(1, 2, 4)
    target = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_branch_shapes("x", branch, target)


def test_resolve_branch_shapes_raises_on_rank_mismatch():
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_branch_shapes("x", torch.zeros(1, 1, 4), torch.zeros(1, 3))


# ---------------------------------------------------------------------------
# ChromBPNet(shared_bias=True) -- new constructor flag wires a 1-channel
# bias branch and the forward broadcasts it across accessibility channels.
# ---------------------------------------------------------------------------


def _multitask_args(filters: int = 4) -> dict[str, Any]:
    """Small BPNet sub-model kwargs for fast multi-task tests."""
    return {
        "filters": filters,
        "n_dilated_layers": 1,
        "conv_kernel_size": 5,
        "dil_kernel_size": 3,
        "profile_kernel_size": 5,
        "activation": "relu",
        "weight_norm": False,
        "residual_architecture": "residual_post-activation_conv",
    }


def test_chrombpnet_default_shared_bias_is_false():
    model = _make_small_model()
    assert model.shared_bias is False
    # Bias branch matches the accessibility branch's channel count.
    assert model.bias_model.n_output_channels == 1


def test_chrombpnet_shared_bias_builds_singleton_bias_branch():
    model = ChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b", "task_c"],
        shared_bias=True,
        accessibility_args=_multitask_args(filters=6),
        bias_args=_multitask_args(filters=4),
    )
    assert model.shared_bias is True
    assert model.accessibility_model.n_output_channels == 3
    # The single-channel bias contract: tools/train_chrombpnet_bias.py exports
    # a one-channel checkpoint and downstream loaders broadcast that one
    # channel across multi-task accessibility outputs.
    assert model.bias_model.n_output_channels == 1


def test_chrombpnet_shared_bias_forward_broadcasts():
    """Forward math: combined[:, k, :] == acc[:, k, :] + bias[:, 0, :]
    for every k, with bias contributing identically to every task."""
    model = ChromBPNet(
        input_len=16, output_len=4,
        output_channels=["task_a", "task_b"],
        shared_bias=True,
        accessibility_args=_multitask_args(),
        bias_args=_multitask_args(),
    )
    acc_logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
    acc_log_counts = torch.log(torch.tensor([[2.0, 4.0]]))
    bias_logits = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    bias_log_counts = torch.log(torch.tensor([[8.0]]))
    model.accessibility_model = _FixedBranch(  # type: ignore[assignment]
        logits=acc_logits, log_counts=acc_log_counts,
    )
    model.bias_model = _FixedBranch(  # type: ignore[assignment]
        logits=bias_logits, log_counts=bias_log_counts,
    )

    out = model(torch.zeros(3, 4, 16))

    assert out.logits.shape == (3, 2, 4)
    assert out.log_counts.shape == (3, 2)
    expected_logits = (acc_logits + bias_logits).expand(3, -1, -1)
    assert torch.allclose(out.logits, expected_logits)
    # Counts: logaddexp(log(2), log(8)) per row 0 channel 0, logaddexp(log(4),
    # log(8)) per row 0 channel 1.  Closed-form: log(2+8)=log(10), log(4+8)=log(12).
    expected_counts = torch.log(torch.tensor([[10.0, 12.0]])).expand(3, -1)
    assert torch.allclose(out.log_counts, expected_counts)


def test_chrombpnet_shared_bias_forward_equals_manual_broadcast():
    """Pin against a hand-built broadcast (no helper) so a future helper
    rewrite cannot silently drift from add-then-logaddexp semantics."""
    torch.manual_seed(0)
    model = ChromBPNet(
        input_len=32, output_len=16,
        output_channels=["a", "b", "c"],
        shared_bias=True,
        accessibility_args=_multitask_args(filters=6),
        bias_args=_multitask_args(filters=4),
    )
    x = torch.randn(2, 4, 32)
    out = model(x)

    with torch.no_grad():
        acc_out = model.accessibility_model(x)
        bias_out = model.bias_model(x)
        # Replicate the helper's broadcast inline.
        bias_logits_broadcast = bias_out.logits.expand_as(acc_out.logits)
        bias_log_counts_broadcast = bias_out.log_counts.expand_as(acc_out.log_counts)
        expected_logits = acc_out.logits + bias_logits_broadcast
        expected_log_counts = torch.logaddexp(
            acc_out.log_counts, bias_log_counts_broadcast,
        )
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)


def test_chrombpnet_non_shared_bias_keeps_per_channel_branches():
    """Default ChromBPNet path (shared_bias=False) keeps a per-channel
    bias branch; regression guard so the new branch doesn't silently
    collapse the old contract."""
    model = ChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=6),
        bias_args=_multitask_args(filters=4),
    )
    assert model.shared_bias is False
    assert model.bias_model.n_output_channels == 2


# ---------------------------------------------------------------------------
# MultitaskChromBPNet -- the subclass that enforces the shared-bias +
# per-channel-count contract.
# ---------------------------------------------------------------------------


def test_multitask_chrombpnet_forward_shape_and_count_modes():
    """Forward emits (B, N, L) logits + (B, N) log-counts; sub-branches
    have the pinned predict_total_count modes."""
    model = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b", "task_c"],
        accessibility_args=_multitask_args(filters=6),
        bias_args=_multitask_args(filters=4),
    )

    out = model(torch.randn(2, 4, 64))

    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (2, 3, 32)
    assert out.log_counts.shape == (2, 3)
    assert model.shared_bias is True
    assert model.accessibility_model.n_output_channels == 3
    assert model.accessibility_model.predict_total_count is False
    assert model.bias_model.n_output_channels == 1
    assert model.bias_model.predict_total_count is True


def test_multitask_chrombpnet_profile_only_bias_keeps_accessibility_counts():
    """The profile_only routing must also work through MultitaskChromBPNet's
    shared-bias broadcast: bias logits broadcast over tasks, but the final
    log_counts come purely from the per-task accessibility heads."""
    model = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b", "task_c"],
        accessibility_args=_multitask_args(filters=6),
        bias_args=_multitask_args(filters=4),
        bias_count_mode="profile_only",
    )

    # Replace branches with deterministic fixtures so the assertion does not
    # depend on randomly-initialised weights.
    model.accessibility_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]),
        log_counts=torch.tensor([[0.5, 1.5, 2.5]]),
    )
    # Shared-bias branch emits (B, 1, L) logits + (B, 1) log_counts; the
    # ChromBPNet forward broadcasts these across the accessibility tasks.
    model.bias_model = _FixedBranch(  # type: ignore[assignment]
        logits=torch.tensor([[[1.0, 1.0, 1.0]]]),
        log_counts=torch.tensor([[999.0]]),
    )
    model.bias_logcount_offset.fill_(999.0)  # type: ignore[operator]

    out = model(torch.randn(2, 4, 64))

    # Logits: bias broadcasts to (B, 3, L) and is added to accessibility logits.
    expected_logits = torch.tensor(
        [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]]]
    ).expand(2, -1, -1)
    # Counts: untouched by bias, exactly the accessibility per-task log-counts.
    expected_log_counts = torch.tensor([[0.5, 1.5, 2.5]]).expand(2, -1)
    assert torch.allclose(out.logits, expected_logits)
    assert torch.allclose(out.log_counts, expected_log_counts)
    assert model.bias_count_mode == "profile_only"


def test_multitask_chrombpnet_requires_at_least_two_tasks():
    with pytest.raises(ValueError, match="at least two output_channels"):
        MultitaskChromBPNet(output_channels=["only_task"])
    with pytest.raises(ValueError, match="at least two output_channels"):
        MultitaskChromBPNet(output_channels=None)
    with pytest.raises(ValueError, match="at least two output_channels"):
        MultitaskChromBPNet(output_channels=[])


def test_multitask_chrombpnet_warns_on_accessibility_predict_total_count_override(
    caplog,
):
    """Passing ``predict_total_count=True`` to the accessibility branch is
    incompatible with multi-task per-channel counts; the constructor warns
    and overrides rather than silently accepting it."""
    with caplog.at_level(logging.WARNING, logger="cerberus.models.chrombpnet"):
        model = MultitaskChromBPNet(
            input_len=64, output_len=32,
            output_channels=["task_a", "task_b"],
            accessibility_args={
                **_multitask_args(filters=5),
                "predict_total_count": True,
            },
            bias_args=_multitask_args(filters=4),
        )
    assert any(
        "predict_total_count" in m and "False" in m for m in caplog.messages
    ), caplog.messages
    assert model.accessibility_model.predict_total_count is False


def test_multitask_chrombpnet_warns_on_bias_predict_total_count_override(caplog):
    with caplog.at_level(logging.WARNING, logger="cerberus.models.chrombpnet"):
        model = MultitaskChromBPNet(
            input_len=64, output_len=32,
            output_channels=["task_a", "task_b"],
            accessibility_args=_multitask_args(filters=5),
            bias_args={
                **_multitask_args(filters=4),
                "predict_total_count": False,
            },
        )
    assert any(
        "bias_args['predict_total_count']" in m for m in caplog.messages
    ), caplog.messages
    assert model.bias_model.predict_total_count is True


def test_multitask_chrombpnet_accepts_matching_predict_total_count_silently(caplog):
    """When the caller passes the same values the constructor already
    pins (acc=False, bias=True), no warning fires -- the warning is for
    surprise, not for user clarity."""
    with caplog.at_level(logging.WARNING, logger="cerberus.models.chrombpnet"):
        MultitaskChromBPNet(
            input_len=64, output_len=32,
            output_channels=["task_a", "task_b"],
            accessibility_args={
                **_multitask_args(filters=5),
                "predict_total_count": False,
            },
            bias_args={
                **_multitask_args(filters=4),
                "predict_total_count": True,
            },
        )
    assert not [
        m for m in caplog.messages if "predict_total_count" in m
    ], caplog.messages


def test_multitask_chrombpnet_loss_is_per_channel_compatible():
    """Output flows through :class:`MultitaskBPNetLoss` without shape errors."""
    model = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=5),
        bias_args=_multitask_args(filters=4),
    )
    outputs = model(torch.randn(2, 4, 64))
    targets = torch.rand(2, 2, 32)
    loss = MultitaskBPNetLoss(alpha=0.1, beta=0.1)(outputs, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_non_shared_multitask_chrombpnet_fails_per_channel_loss():
    """Regression guard: a multi-output ChromBPNet WITHOUT shared_bias
    produces per-channel combined log_counts via the per-channel bias
    branch -- which is fine for absolute-counts losses but breaks the
    ``MSEMultinomialLoss(count_per_channel=True)`` contract when the
    sub-branches' log_counts have shape (B, N) instead of (B,).  This
    test pins the failure mode so a future "helpful" change can't
    silently let the wrong shape through."""
    model = ChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=5),
        bias_args=_multitask_args(filters=4),
    )
    outputs = model(torch.randn(2, 4, 64))
    targets = torch.rand(2, 2, 32)
    with pytest.raises(ValueError, match="per-channel log_counts"):
        MSEMultinomialLoss(count_per_channel=True)(outputs, targets)


def test_multitask_chrombpnet_loads_single_channel_bias_then_freezes(tmp_path):
    """End-to-end recipe for the stage-2 multitask trainer:

    1. Train a one-channel stage-1 bias BPNet (here: build a fresh BPNet and
       save its state_dict).
    2. Instantiate a MultitaskChromBPNet (which has a one-channel bias
       branch).
    3. Load the bias weights into ``bias_model`` via
       ``load_pretrained_weights`` + ``PretrainedConfig(target="bias_model")``.
    4. Freeze the bias subtree via ``apply_freeze`` + ``FreezeSpec``.
    """
    bias_args = _multitask_args(filters=4)
    single_bias = BPNet(
        input_len=64, output_len=32,
        output_channels=["signal"],
        predict_total_count=True,
        **bias_args,
    )
    with torch.no_grad():
        for p in single_bias.parameters():
            p.fill_(0.25)

    ckpt_path = tmp_path / "bias.pt"
    torch.save(single_bias.state_dict(), ckpt_path)

    model = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=5),
        bias_args=bias_args,
    )
    load_pretrained_weights(
        model,
        [
            PretrainedConfig(
                weights_path=str(ckpt_path),
                source=None,
                target="bias_model",
            )
        ],
    )
    apply_freeze(model, [FreezeSpec(pattern="bias_model", eval_mode=True)])

    # Bias weights are loaded bit-for-bit and have requires_grad=False.
    for p in model.bias_model.parameters():
        assert torch.all(p == 0.25)
        assert p.requires_grad is False
    # Accessibility branch is untouched and remains trainable.
    assert all(p.requires_grad for p in model.accessibility_model.parameters())
    assert model.bias_model.training is False


# ---------------------------------------------------------------------------
# Cross-construction equivalence -- protects against drift between the
# explicit ``ChromBPNet(shared_bias=True, ...)`` low-level path and the
# ``MultitaskChromBPNet(...)`` convenience subclass.  If a future refactor
# adds extra wiring to one path only, these regressions catch it.
# ---------------------------------------------------------------------------


def _hand_built_equivalent(
    output_channels: list[str], acc_args: dict, bias_args: dict,
) -> ChromBPNet:
    return ChromBPNet(
        input_len=64, output_len=32,
        output_channels=output_channels,
        accessibility_args={**acc_args, "predict_total_count": False},
        bias_args={**bias_args, "predict_total_count": True},
        shared_bias=True,
    )


def test_multitask_chrombpnet_matches_hand_built_state_dict_layout():
    """Both paths produce identical state_dict key sets (same architecture)."""
    acc = _multitask_args(filters=6)
    bias = _multitask_args(filters=4)
    output_channels = ["task_a", "task_b", "task_c"]

    multitask = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=output_channels,
        accessibility_args=acc,
        bias_args=bias,
    )
    hand_built = _hand_built_equivalent(output_channels, acc, bias)

    assert set(multitask.state_dict().keys()) == set(hand_built.state_dict().keys())
    # And the per-tensor shapes match exactly.
    for key, tensor in multitask.state_dict().items():
        assert tensor.shape == hand_built.state_dict()[key].shape


def test_multitask_chrombpnet_forward_matches_hand_built_under_same_weights():
    """Construct both, copy weights one-way, assert forward outputs agree
    bit-for-bit.  This is the strongest guarantee that the subclass adds
    no extra forward-time math."""
    acc = _multitask_args(filters=6)
    bias = _multitask_args(filters=4)
    output_channels = ["task_a", "task_b"]

    torch.manual_seed(7)
    multitask = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=output_channels,
        accessibility_args=acc,
        bias_args=bias,
    )
    hand_built = _hand_built_equivalent(output_channels, acc, bias)
    hand_built.load_state_dict(multitask.state_dict())

    x = torch.randn(2, 4, 64)
    multitask.eval()
    hand_built.eval()
    out_mt = multitask(x)
    out_hb = hand_built(x)
    assert torch.equal(out_mt.logits, out_hb.logits)
    assert torch.equal(out_mt.log_counts, out_hb.log_counts)


def test_multitask_chrombpnet_state_dict_round_trip_preserves_offset(tmp_path):
    """Saving and reloading a MultitaskChromBPNet preserves the
    bias_logcount_offset buffer -- key for the calibration step."""
    model = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=5),
        bias_args=_multitask_args(filters=4),
    )
    model.bias_logcount_offset.fill_(0.42)  # type: ignore[union-attr]

    ckpt = tmp_path / "multitask.pt"
    torch.save(model.state_dict(), ckpt)

    fresh = MultitaskChromBPNet(
        input_len=64, output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_multitask_args(filters=5),
        bias_args=_multitask_args(filters=4),
    )
    fresh.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    assert fresh.bias_logcount_offset.item() == pytest.approx(0.42)  # type: ignore[union-attr]


def test_multitask_chrombpnet_importable_from_class_path():
    cls = import_class("cerberus.models.chrombpnet.MultitaskChromBPNet")
    assert cls is MultitaskChromBPNet
    assert _cerberus_models.MultitaskChromBPNet is MultitaskChromBPNet
