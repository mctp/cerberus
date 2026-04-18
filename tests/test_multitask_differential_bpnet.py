"""Tests for MultitaskBPNet, MultitaskBPNetLoss, and DifferentialCountLoss.

Covers:
- MultitaskBPNet forward pass shape and per-channel count outputs
- MultitaskBPNet validation (requires ≥2 output channels, predict_total_count=False)
- MultitaskBPNetLoss fixed hyperparameter enforcement
- DifferentialCountLoss: delta derived from (B, N, L) targets
- DifferentialCountLoss validation errors
- Round-trip: MultitaskBPNet → MultitaskBPNetLoss (Phase 1)
- Round-trip: MultitaskBPNet → DifferentialCountLoss (Phase 2)
"""

import pytest
import torch

from cerberus.loss import DifferentialCountLoss
from cerberus.models.bpnet import (
    MultitaskBPNet,
    MultitaskBPNetLoss,
)
from cerberus.output import ProfileCountOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
INPUT_LEN = 1200
OUTPUT_LEN = 1000
FILTERS = 16
N_LAYERS = 2
CONDITIONS = ["ctrl", "treat"]


@pytest.fixture
def model():
    return MultitaskBPNet(
        output_channels=CONDITIONS,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        filters=FILTERS,
        n_dilated_layers=N_LAYERS,
    )


@pytest.fixture
def seq():
    return torch.zeros(BATCH, 4, INPUT_LEN)


@pytest.fixture
def targets_2cond():
    """Absolute count targets for 2 conditions, (B, N, L)."""
    return torch.rand(BATCH, len(CONDITIONS), OUTPUT_LEN) * 10


# ---------------------------------------------------------------------------
# MultitaskBPNet
# ---------------------------------------------------------------------------


def test_multitask_bpnet_forward_shape(model, seq):
    out = model(seq)
    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (BATCH, len(CONDITIONS), OUTPUT_LEN)
    assert out.log_counts.shape == (BATCH, len(CONDITIONS))


def test_multitask_bpnet_predict_total_count_is_false(model):
    """predict_total_count must always be False."""
    assert model.predict_total_count is False


@pytest.mark.parametrize("channels", [[], ["only_one"]])
def test_multitask_bpnet_requires_at_least_two_channels(channels):
    with pytest.raises(ValueError, match="at least 2"):
        MultitaskBPNet(output_channels=channels, input_len=500, output_len=300)


def test_multitask_bpnet_three_conditions():
    model3 = MultitaskBPNet(
        output_channels=["a", "b", "c"],
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        filters=FILTERS,
        n_dilated_layers=N_LAYERS,
    )
    seq = torch.zeros(BATCH, 4, INPUT_LEN)
    out = model3(seq)
    assert out.logits.shape == (BATCH, 3, OUTPUT_LEN)
    assert out.log_counts.shape == (BATCH, 3)


# ---------------------------------------------------------------------------
# MultitaskBPNetLoss
# ---------------------------------------------------------------------------


def test_multitask_bpnet_loss_fixed_params():
    loss = MultitaskBPNetLoss()
    assert loss.count_per_channel is True
    assert loss.average_channels is True
    assert loss.flatten_channels is False
    assert loss.log1p_targets is False


def test_multitask_bpnet_loss_alpha_beta():
    loss = MultitaskBPNetLoss(alpha=2.0, beta=0.5)
    assert loss.count_weight == 2.0
    assert loss.profile_weight == 0.5


@pytest.mark.parametrize(
    "override_kwarg,conflicting_value",
    [
        ("count_per_channel", False),
        ("log1p_targets", True),
        ("flatten_channels", True),
    ],
)
def test_multitask_bpnet_loss_warns_on_override(override_kwarg, conflicting_value):
    """Override a pinned kwarg → a warning record is emitted on the bpnet logger.

    Attaches a local handler to the ``cerberus.models.bpnet`` logger so the
    assertion is independent of whether ``cerberus.setup_logging`` has been
    called elsewhere in the test session (which sets propagate=False and
    breaks caplog capture of ``cerberus.*`` loggers).
    """
    import logging

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    bpnet_logger = logging.getLogger("cerberus.models.bpnet")
    bpnet_logger.addHandler(handler)
    try:
        MultitaskBPNetLoss(**{override_kwarg: conflicting_value})
    finally:
        bpnet_logger.removeHandler(handler)

    assert any(override_kwarg in r.getMessage() for r in records)


def test_multitask_bpnet_loss_phase1_roundtrip(model, seq, targets_2cond):
    loss_fn = MultitaskBPNetLoss()
    out = model(seq)
    loss = loss_fn(out, targets_2cond)
    assert loss.ndim == 0
    assert loss.item() > 0
    loss.backward()


# ---------------------------------------------------------------------------
# DifferentialCountLoss — basic
# ---------------------------------------------------------------------------


def _make_output(log_counts: torch.Tensor, output_len: int = OUTPUT_LEN) -> ProfileCountOutput:
    """Create a ProfileCountOutput with dummy logits."""
    B, N = log_counts.shape
    logits = torch.zeros(B, N, output_len)
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


def _bnl_targets_with_known_delta(
    sum_a: torch.Tensor,
    sum_b: torch.Tensor,
    n_channels: int,
    output_len: int,
) -> torch.Tensor:
    """Build a ``(B, N, L)`` target tensor whose per-channel length-sums are exact.

    Channels 0 (``A``) and 1 (``B``) get constant per-length values so their
    length-sums equal ``sum_a`` and ``sum_b`` (non-negative) exactly. All
    other channels are zero.
    """
    B = sum_a.shape[0]
    t = torch.zeros(B, n_channels, output_len)
    t[:, 0, :] = (sum_a / output_len).view(B, 1)
    t[:, 1, :] = (sum_b / output_len).view(B, 1)
    return t


def test_differential_count_loss_derives_delta_from_targets():
    """Delta is derived from (B, N, L) targets: log2((sum_B + pc) / (sum_A + pc))."""
    pc = 1.0
    loss_fn = DifferentialCountLoss(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc
    )
    sum_a = torch.tensor([3.0, 1.0, 7.0, 0.0])
    sum_b = torch.tensor([15.0, 3.0, 1.0, 0.0])
    targets = _bnl_targets_with_known_delta(
        sum_a, sum_b, n_channels=2, output_len=OUTPUT_LEN
    )
    # If the model predicts exactly log2((sum_b + pc) / (sum_a + pc)) as
    # log_counts[:, 1] - log_counts[:, 0], MSE is zero.
    expected_delta = torch.log2((sum_b + pc) / (sum_a + pc))
    log_counts = torch.zeros(4, 2)
    log_counts[:, 1] = expected_delta  # (b - a) = expected_delta since a=0
    out = _make_output(log_counts)

    loss = loss_fn(out, targets)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_differential_count_loss_nonzero_when_prediction_off():
    """Shifting the prediction by a constant moves MSE predictably."""
    pc = 1.0
    loss_fn = DifferentialCountLoss(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc
    )
    sum_a = torch.tensor([3.0, 7.0])
    sum_b = torch.tensor([15.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b, 2, OUTPUT_LEN)
    # Zero prediction → MSE equals mean(expected_delta ** 2).
    expected_delta = torch.log2((sum_b + pc) / (sum_a + pc))
    out = _make_output(torch.zeros(2, 2))
    loss = loss_fn(out, targets)
    assert loss.item() == pytest.approx((expected_delta ** 2).mean().item(), rel=1e-6)


def test_differential_count_loss_pseudocount_affects_target():
    """Changing ``count_pseudocount`` changes the derived delta target."""
    sum_a = torch.tensor([0.0])
    sum_b = torch.tensor([1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b, 2, OUTPUT_LEN)
    out = _make_output(torch.zeros(1, 2))
    loss_small = DifferentialCountLoss(count_pseudocount=0.1)(out, targets)
    loss_large = DifferentialCountLoss(count_pseudocount=100.0)(out, targets)
    # Larger pseudocount → smaller |log2 ratio| → smaller MSE
    assert loss_large.item() < loss_small.item()


def test_differential_count_loss_components_has_only_delta_loss():
    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)
    out = _make_output(torch.zeros(2, 2))
    targets = torch.zeros(2, 2, OUTPUT_LEN)
    comps = loss_fn.loss_components(out, targets)
    assert set(comps.keys()) == {"delta_loss"}


def test_differential_count_loss_same_idx_raises():
    with pytest.raises(ValueError, match="must differ"):
        DifferentialCountLoss(cond_a_idx=1, cond_b_idx=1)


def test_differential_count_loss_out_of_range_idx():
    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=5)
    out = _make_output(torch.zeros(2, 2))  # only 2 channels
    targets = torch.zeros(2, 2, OUTPUT_LEN)
    with pytest.raises(ValueError, match="out of range"):
        loss_fn(out, targets)


def test_differential_count_loss_wrong_output_type():
    loss_fn = DifferentialCountLoss()
    with pytest.raises(TypeError, match="ProfileCountOutput"):
        loss_fn("not_an_output", torch.zeros(2, 2, OUTPUT_LEN))


def test_differential_count_loss_rejects_2d_targets():
    """Targets must be (B, N, L); a (B, N) shape must raise clearly."""
    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)
    out = _make_output(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape \\(B, N, L\\)"):
        loss_fn(out, torch.zeros(2, 2))


def test_differential_count_loss_rejects_too_few_target_channels():
    """Targets with fewer channels than max(cond_a, cond_b)+1 must raise."""
    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)
    out = _make_output(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="at least 2 channels"):
        loss_fn(out, torch.zeros(2, 1, OUTPUT_LEN))


@pytest.mark.parametrize(
    "cond_a,cond_b",
    [(-1, 1), (0, -2), (-3, -1)],
)
def test_differential_count_loss_rejects_negative_idx(cond_a, cond_b):
    """Negative indices silently select the last channel — reject at init."""
    with pytest.raises(ValueError, match="must be non-negative"):
        DifferentialCountLoss(cond_a_idx=cond_a, cond_b_idx=cond_b)


# ---------------------------------------------------------------------------
# Phase 2 round-trip
# ---------------------------------------------------------------------------


def test_phase2_roundtrip_backward(model, seq, targets_2cond):
    """Phase 2: freeze trunk + profile heads, fine-tune count heads on delta."""
    # Simulate partial fine-tuning: only count_dense receives gradients
    for name, param in model.named_parameters():
        param.requires_grad = name == "count_dense.weight" or name == "count_dense.bias"

    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)
    out = model(seq)
    loss = loss_fn(out, targets_2cond)
    assert loss.ndim == 0
    loss.backward()

    # Only count_dense should have gradients
    for name, param in model.named_parameters():
        if name in ("count_dense.weight", "count_dense.bias"):
            assert param.grad is not None, f"{name} should have grad"
        else:
            assert param.grad is None, f"{name} should not have grad"


def test_phase2_end_to_end_all_weights(model, seq, targets_2cond):
    """Phase 2 end-to-end: trunk and count_dense update; profile_conv does not.

    DifferentialCountLoss sets profile loss weight to 0 (following Naqvi et al.).
    Gradients flow through: count_dense → global avg pool → res_layers → iconv.
    profile_conv is a separate branch that does not feed into log_counts and
    therefore correctly receives no gradient.
    """
    loss_fn = DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1)
    out = model(seq)
    loss = loss_fn(out, targets_2cond)
    loss.backward()

    # Trunk parameters (iconv, res_layers) and count_dense must have gradients
    trunk_and_count = {"iconv", "count_dense"}
    for name, param in model.named_parameters():
        is_trunk_or_count = any(k in name for k in trunk_and_count)
        if is_trunk_or_count:
            assert param.grad is not None, f"{name} should have gradient"

    # profile_conv has no gradient: it is a separate branch not used by count loss
    for name, param in model.named_parameters():
        if "profile_conv" in name:
            assert param.grad is None, (
                f"{name} should not have gradient when profile loss weight is 0"
            )
