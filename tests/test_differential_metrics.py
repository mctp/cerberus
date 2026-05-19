"""Tests for the per-class differential log-count metrics.

Covers :class:`DifferentialLogCountsMeanSquaredError`,
:class:`DifferentialLogCountsRootMeanSquaredError`, and
:class:`DifferentialLogCountsPearsonCorrCoef` in :mod:`cerberus.metrics`.
The ``DifferentialBPNetMetricCollection`` wrapper and the
``instantiate_metrics_and_loss`` integration are exercised separately.
"""

from __future__ import annotations

import pytest
import torch

from cerberus.metrics import (
    DifferentialLogCountsMeanSquaredError,
    DifferentialLogCountsPearsonCorrCoef,
    DifferentialLogCountsRootMeanSquaredError,
)
from cerberus.output import ProfileCountOutput


OUTPUT_LEN = 32


def _bnl_targets_with_known_delta(
    sum_a: torch.Tensor,
    sum_b: torch.Tensor,
    n_channels: int = 2,
    output_len: int = OUTPUT_LEN,
) -> torch.Tensor:
    """Build a (B, N, L) targets tensor whose channel-wise length-sums are
    ``sum_a`` and ``sum_b``.  Constant signal per position keeps the math
    transparent."""
    batch_size = sum_a.shape[0]
    targets = torch.zeros(batch_size, n_channels, output_len)
    targets[:, 0, :] = (sum_a / output_len).view(batch_size, 1)
    targets[:, 1, :] = (sum_b / output_len).view(batch_size, 1)
    return targets


def _make_output(log_counts: torch.Tensor) -> ProfileCountOutput:
    batch_size, n_channels = log_counts.shape
    logits = torch.zeros(batch_size, n_channels, OUTPUT_LEN)
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


# ---------------------------------------------------------------------------
# Construction-time validation (the small shared helper)
# ---------------------------------------------------------------------------


def test_rejects_equal_channel_indices_on_construction():
    with pytest.raises(ValueError, match="must differ"):
        DifferentialLogCountsMeanSquaredError(cond_a_idx=1, cond_b_idx=1)
    with pytest.raises(ValueError, match="must differ"):
        DifferentialLogCountsPearsonCorrCoef(cond_a_idx=2, cond_b_idx=2)


def test_rejects_negative_channel_index_on_construction():
    with pytest.raises(ValueError, match="non-negative"):
        DifferentialLogCountsMeanSquaredError(cond_a_idx=-1, cond_b_idx=0)
    with pytest.raises(ValueError, match="non-negative"):
        DifferentialLogCountsPearsonCorrCoef(cond_a_idx=0, cond_b_idx=-1)


# ---------------------------------------------------------------------------
# Update-time input validation
# ---------------------------------------------------------------------------


def test_metrics_reject_out_of_range_channel():
    metric = DifferentialLogCountsMeanSquaredError(cond_a_idx=0, cond_b_idx=5)
    with pytest.raises(ValueError, match="out of range"):
        metric.update(_make_output(torch.zeros(2, 2)), torch.zeros(2, 2, OUTPUT_LEN))


def test_metrics_reject_non_profile_count_output():
    metric = DifferentialLogCountsMeanSquaredError(cond_a_idx=0, cond_b_idx=1)
    with pytest.raises(TypeError, match="ProfileCountOutput"):
        metric.update(torch.zeros(2, 2), torch.zeros(2, 2, OUTPUT_LEN))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Math: MSE / RMSE
# ---------------------------------------------------------------------------


def test_mse_zero_when_prediction_matches_target():
    pc = 1.0
    metric = DifferentialLogCountsMeanSquaredError(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc,
    )
    sum_a = torch.tensor([3.0, 1.0, 7.0, 0.0])
    sum_b = torch.tensor([15.0, 3.0, 1.0, 0.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    expected_delta = torch.log((sum_b + pc) / (sum_a + pc))
    log_counts = torch.zeros(4, 2)
    log_counts[:, 1] = expected_delta

    metric.update(_make_output(log_counts), targets)
    assert metric.compute().item() == pytest.approx(0.0, abs=1e-6)


def test_rmse_matches_root_mse():
    pc = 1.0
    metric = DifferentialLogCountsRootMeanSquaredError(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc,
    )
    sum_a = torch.tensor([3.0, 7.0])
    sum_b = torch.tensor([15.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    expected_delta = torch.log((sum_b + pc) / (sum_a + pc))

    metric.update(_make_output(torch.zeros(2, 2)), targets)
    expected_rmse = torch.sqrt((expected_delta ** 2).mean())
    assert metric.compute().item() == pytest.approx(expected_rmse.item(), rel=1e-6)


def test_mse_accumulates_across_batches():
    """Two updates with different per-batch errors → running mean is correct."""
    pc = 1.0
    metric = DifferentialLogCountsMeanSquaredError(count_pseudocount=pc)

    # Batch 1: zero error.
    sum_a1 = torch.tensor([2.0])
    sum_b1 = torch.tensor([8.0])
    targets1 = _bnl_targets_with_known_delta(sum_a1, sum_b1)
    delta1 = torch.log((sum_b1 + pc) / (sum_a1 + pc))
    log_counts1 = torch.zeros(1, 2)
    log_counts1[:, 1] = delta1
    metric.update(_make_output(log_counts1), targets1)

    # Batch 2: predict zero → squared error == delta**2.
    sum_a2 = torch.tensor([1.0, 4.0])
    sum_b2 = torch.tensor([7.0, 1.0])
    targets2 = _bnl_targets_with_known_delta(sum_a2, sum_b2)
    delta2 = torch.log((sum_b2 + pc) / (sum_a2 + pc))
    metric.update(_make_output(torch.zeros(2, 2)), targets2)

    # Mean over 3 examples: (0 + delta2[0]**2 + delta2[1]**2) / 3
    expected = (delta2 ** 2).sum() / 3
    assert metric.compute().item() == pytest.approx(expected.item(), rel=1e-6)


def test_mse_pseudocount_affects_target_delta():
    """Same counts, two pseudocount values → different target → different MSE."""
    sum_a = torch.tensor([1.0])
    sum_b = torch.tensor([10.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    out = _make_output(torch.zeros(1, 2))  # always predicts zero

    metric_small = DifferentialLogCountsMeanSquaredError(count_pseudocount=0.1)
    metric_large = DifferentialLogCountsMeanSquaredError(count_pseudocount=100.0)
    metric_small.update(out, targets)
    metric_large.update(out, targets)
    # Larger pc shrinks the log-ratio toward 0, so MSE shrinks too.
    assert metric_large.compute().item() < metric_small.compute().item()


def test_mse_sign_convention_swap_is_invariant():
    """Swapping (cond_a, cond_b) negates target and pred; MSE squares away."""
    pc = 1.0
    sum_a = torch.tensor([3.0, 7.0])
    sum_b = torch.tensor([15.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    log_counts = torch.tensor([[0.5, 1.2], [2.0, 0.3]])

    ab = DifferentialLogCountsMeanSquaredError(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc,
    )
    ba = DifferentialLogCountsMeanSquaredError(
        cond_a_idx=1, cond_b_idx=0, count_pseudocount=pc,
    )
    ab.update(_make_output(log_counts), targets)
    ba.update(_make_output(log_counts), targets)
    assert ab.compute().item() == pytest.approx(ba.compute().item(), rel=1e-6)


# ---------------------------------------------------------------------------
# Math: Pearson
# ---------------------------------------------------------------------------


def test_pearson_perfect_correlation_returns_one():
    pc = 1.0
    metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)
    sum_a = torch.tensor([1.0, 2.0, 4.0, 8.0])
    sum_b = torch.tensor([8.0, 4.0, 2.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    true_delta = torch.log((sum_b + pc) / (sum_a + pc))

    log_counts = torch.zeros(4, 2)
    log_counts[:, 1] = true_delta
    metric.update(_make_output(log_counts), targets)
    assert metric.compute().item() == pytest.approx(1.0, abs=1e-6)


def test_pearson_invariant_to_positive_affine_prediction():
    """``2.5 * true_delta + 1.75`` is still perfectly correlated with truth."""
    pc = 1.0
    metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)
    sum_a = torch.tensor([1.0, 2.0, 4.0, 8.0])
    sum_b = torch.tensor([8.0, 4.0, 2.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    true_delta = torch.log((sum_b + pc) / (sum_a + pc))

    log_counts = torch.zeros(4, 2)
    log_counts[:, 1] = 2.5 * true_delta + 1.75
    metric.update(_make_output(log_counts), targets)
    assert metric.compute().item() == pytest.approx(1.0, abs=1e-6)


def test_pearson_empty_returns_nan():
    metric = DifferentialLogCountsPearsonCorrCoef()
    result = metric.compute()
    assert torch.isnan(result)


def test_pearson_constant_prediction_returns_nan():
    """Zero predicted variance → denom under threshold → NaN."""
    pc = 1.0
    metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)
    sum_a = torch.tensor([1.0, 2.0, 4.0])
    sum_b = torch.tensor([8.0, 4.0, 2.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    # All-zero log_counts → pred_delta is constant zero → zero variance.
    metric.update(_make_output(torch.zeros(3, 2)), targets)
    assert torch.isnan(metric.compute())


def test_pearson_accumulates_across_batches():
    """Two updates with perfectly-correlated halves → r=1.0 on the union."""
    pc = 1.0
    metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)

    for sum_a, sum_b in [
        (torch.tensor([1.0, 2.0]), torch.tensor([8.0, 4.0])),
        (torch.tensor([4.0, 8.0]), torch.tensor([2.0, 1.0])),
    ]:
        targets = _bnl_targets_with_known_delta(sum_a, sum_b)
        true_delta = torch.log((sum_b + pc) / (sum_a + pc))
        log_counts = torch.zeros(2, 2)
        log_counts[:, 1] = true_delta
        metric.update(_make_output(log_counts), targets)

    assert metric.compute().item() == pytest.approx(1.0, abs=1e-6)
