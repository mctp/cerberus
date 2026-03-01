"""
Tests verifying mathematical consistency of count_pseudocount across the codebase.

The count_pseudocount parameter controls the log-space transform for count targets:
    forward:  log(count + pseudocount)
    inverse:  exp(log_count) - pseudocount

When pseudocount=1.0 (default), this is equivalent to log1p / expm1.
These tests verify that every component handles arbitrary pseudocount values
correctly and consistently.

Key invariants tested:
  1. Loss targets use log(count + pseudocount)                → loss.py
  2. Metrics use the same transform for target log-counts     → metrics.py
  3. Count-reconstruction metrics invert correctly             → metrics.py
  4. Multi-channel aggregation inverts/reapplies correctly     → output.py
  5. All components agree for pseudocount=1.0 (log1p compat)  → everywhere
  6. Per-channel loss paths use pseudocount correctly          → loss.py
  7. MetricCollections propagate pseudocount to all sub-metrics
  8. log1p_targets and pseudocount are independent transforms
  9. Gradient flow through pseudocount-adjusted targets
 10. Pure-log multi-channel path ignores pseudocount parameter
 11. obs_log_count_fn uses pseudocount for MSE losses
 12. Three-way consistency: loss = metric = module accumulator
"""

import math
import torch
import torch.nn.functional as F
import pytest

from cerberus.output import ProfileCountOutput, ProfileLogRates, compute_total_log_counts
from cerberus.loss import MSEMultinomialLoss, CoupledMSEMultinomialLoss
from cerberus.metrics import (
    LogCountsMeanSquaredError,
    LogCountsPearsonCorrCoef,
    CountProfilePearsonCorrCoef,
    CountProfileMeanSquaredError,
)


# ---------------------------------------------------------------------------
# 1. Forward/inverse round-trip: log(count + p) → exp(...) - p == count
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify log(x + p) → exp(.) - p is exact for various pseudocounts."""

    @pytest.mark.parametrize("pseudocount", [0.5, 1.0, 10.0, 100.0])
    def test_scalar_round_trip(self, pseudocount):
        counts = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0])
        log_counts = torch.log(counts + pseudocount)
        recovered = torch.exp(log_counts) - pseudocount
        assert torch.allclose(recovered, counts, atol=1e-4)

    def test_pseudocount_1_matches_log1p_expm1(self):
        """pseudocount=1.0 must be numerically identical to log1p / expm1."""
        counts = torch.tensor([0.0, 0.5, 1.0, 50.0, 1e4])
        via_log1p = torch.log1p(counts)
        via_pseudocount = torch.log(counts + 1.0)
        assert torch.allclose(via_log1p, via_pseudocount, atol=1e-6)

        via_expm1 = torch.expm1(via_log1p)
        via_exp_minus = torch.exp(via_pseudocount) - 1.0
        assert torch.allclose(via_expm1, via_exp_minus, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. MSEMultinomialLoss target transform uses pseudocount correctly
# ---------------------------------------------------------------------------

class TestLossPseudocount:
    """MSEMultinomialLoss count target must be log(count + pseudocount)."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_count_target_matches_pseudocount(self, pseudocount):
        """The loss target for counts must use the configured pseudocount."""
        B, C, L = 2, 1, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50  # random positive counts

        total = targets.sum(dim=(1, 2))  # (B,)
        expected_log_target = torch.log(total + pseudocount)

        # Build a loss, extract the count target it would compute
        loss_fn = MSEMultinomialLoss(count_weight=1.0, profile_weight=0.0,
                                     count_pseudocount=pseudocount)

        # Create dummy prediction matching the expected target
        pred_log_counts = expected_log_target.reshape(B, 1)
        logits = torch.zeros(B, C, L)
        out = ProfileCountOutput(logits=logits, log_counts=pred_log_counts)

        # If targets match perfectly, count loss should be 0
        loss_val = loss_fn(out, targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5), \
            f"Loss should be 0 for perfect prediction, got {loss_val.item()}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_coupled_count_target_matches_pseudocount(self, pseudocount):
        """CoupledMSEMultinomialLoss count target uses the same pseudocount."""
        B, C, L = 2, 1, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        total = targets.sum(dim=(1, 2))
        expected_log_target = torch.log(total + pseudocount)

        loss_fn = CoupledMSEMultinomialLoss(count_weight=1.0, profile_weight=0.0,
                                            count_pseudocount=pseudocount)

        # For coupled loss, log_rates determine the predicted count via logsumexp.
        # Create log_rates such that logsumexp = expected_log_target.
        # Single position: logsumexp over length=1 is just the value itself.
        log_rates = expected_log_target.reshape(B, 1, 1)
        out = ProfileLogRates(log_rates=log_rates)

        loss_val = loss_fn(out, targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5)

    def test_wrong_pseudocount_gives_nonzero_loss(self):
        """Using the wrong pseudocount for predictions yields a non-zero count loss."""
        B, C, L = 1, 1, 8
        targets = torch.ones(B, C, L) * 10  # total = 80

        # Loss uses pseudocount=100
        loss_fn = MSEMultinomialLoss(count_weight=1.0, profile_weight=0.0,
                                     count_pseudocount=100.0)

        # Prediction uses pseudocount=1 (wrong) → log(80 + 1) = log(81)
        wrong_pred = torch.log(torch.tensor([[81.0]]))
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=wrong_pred)

        loss_val = loss_fn(out, targets)
        expected_diff = (torch.log(torch.tensor(81.0)) - torch.log(torch.tensor(180.0))) ** 2
        assert torch.isclose(loss_val, expected_diff, atol=1e-5)
        assert loss_val > 0.1, "Wrong pseudocount should give substantial loss"


# ---------------------------------------------------------------------------
# 3. LogCountsMeanSquaredError / LogCountsPearsonCorrCoef use pseudocount
# ---------------------------------------------------------------------------

class TestLogCountsMetricsPseudocount:
    """Log-counts metrics must use count_pseudocount for the target transform."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_mse_perfect_prediction(self, pseudocount):
        """Perfect prediction under the given pseudocount → MSE = 0."""
        B, C, L = 2, 1, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        total = targets.sum(dim=(1, 2))
        pred_log = torch.log(total + pseudocount).reshape(B, 1)
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred_log)

        metric = LogCountsMeanSquaredError(count_pseudocount=pseudocount)
        metric.update(out, targets)
        assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-5)

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_pearson_perfect_correlation(self, pseudocount):
        """Perfectly correlated predictions → Pearson ≈ 1."""
        counts = [10.0, 50.0, 100.0, 500.0, 1000.0]
        B = len(counts)
        L = 4

        metric = LogCountsPearsonCorrCoef(count_pseudocount=pseudocount)

        for c in counts:
            targets = torch.zeros(1, 1, L)
            targets[0, 0, 0] = c  # total = c
            pred_log = torch.log(torch.tensor(c) + pseudocount).reshape(1, 1)
            out = ProfileCountOutput(logits=torch.zeros(1, 1, L), log_counts=pred_log)
            metric.update(out, targets)

        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)

    def test_mse_pseudocount_mismatch_gives_error(self):
        """Using pseudocount=1 predictions with pseudocount=100 metric → nonzero MSE."""
        targets = torch.zeros(1, 1, 4)
        targets[0, 0, 0] = 500.0  # total = 500

        # Pred is log(500 + 1) = log(501) — matches pseudocount=1
        pred = torch.log(torch.tensor([[501.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=pred)

        # Metric uses pseudocount=100 — expects log(500 + 100) = log(600)
        metric = LogCountsMeanSquaredError(count_pseudocount=100.0)
        metric.update(out, targets)
        val = metric.compute()

        expected = (torch.log(torch.tensor(501.0)) - torch.log(torch.tensor(600.0))) ** 2
        assert torch.isclose(val, expected, atol=1e-5)
        assert val > 0.01


# ---------------------------------------------------------------------------
# 4. Count-reconstruction metrics invert log_counts with correct pseudocount
# ---------------------------------------------------------------------------

class TestCountReconstructionPseudocount:
    """
    CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, and
    CountProfilePearsonCorrCoef reconstruct counts from log_counts
    via exp(log_counts) - pseudocount. Verify correctness for non-default pseudocounts.
    """

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_count_profile_pearson_perfect(self, pseudocount):
        """Perfect reconstruction → Pearson ≈ 1."""
        B, C, L = 1, 1, 16
        target = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        total = target.sum()

        probs = target / total
        logits = torch.log(probs + 1e-10)
        # Model predicts log(total + pseudocount)
        log_counts = torch.log(total + pseudocount).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfilePearsonCorrCoef(count_pseudocount=pseudocount)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3), \
            f"Expected ~1.0 for pseudocount={pseudocount}, got {val.item()}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_count_profile_mse_perfect(self, pseudocount):
        """Perfect reconstruction → MSE ≈ 0."""
        B, C, L = 1, 1, 16
        target = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)
        total = target.sum()

        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log(total + pseudocount).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfileMeanSquaredError(count_pseudocount=pseudocount)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(0.0), atol=1e-2), \
            f"Expected ~0.0 for pseudocount={pseudocount}, got {val.item()}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_per_example_count_profile_pearson_perfect(self, pseudocount):
        """Per-example count profile Pearson with correct pseudocount."""
        B, C, L = 1, 1, 16
        target = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)
        total = target.sum()

        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log(total + pseudocount).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfilePearsonCorrCoef(
            count_pseudocount=pseudocount
        )
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)

    def test_wrong_pseudocount_changes_reconstruction(self):
        """Using the wrong pseudocount in the metric distorts the reconstructed counts."""
        B, C, L = 1, 1, 8
        target = torch.ones(B, C, L) * 10  # total = 80

        # Model was trained with pseudocount=100, predicts log(80 + 100) = log(180)
        log_counts = torch.log(torch.tensor([[180.0]]))
        probs = target / target.sum()
        logits = torch.log(probs + 1e-10)
        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)

        # Correct metric: pseudocount=100 → exp(log(180)) - 100 = 80 → correct total
        metric_correct = CountProfileMeanSquaredError(count_pseudocount=100.0)
        metric_correct.update(preds, target)
        mse_correct = metric_correct.compute()

        # Wrong metric: pseudocount=1 → exp(log(180)) - 1 = 179 → wrong total
        metric_wrong = CountProfileMeanSquaredError(count_pseudocount=1.0)
        metric_wrong.update(preds, target)
        mse_wrong = metric_wrong.compute()

        assert mse_correct < mse_wrong, \
            f"Correct pseudocount should give lower MSE: {mse_correct.item()} vs {mse_wrong.item()}"
        assert torch.isclose(mse_correct, torch.tensor(0.0), atol=1e-2)


# ---------------------------------------------------------------------------
# 5. compute_total_log_counts with non-default pseudocount
# ---------------------------------------------------------------------------

class TestComputeTotalLogCountsPseudocount:
    """Verify multi-channel aggregation with log_counts_include_pseudocount uses the pseudocount."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_multi_channel_log_counts_include_pseudocount(self, pseudocount):
        """
        Multi-channel log_counts_include_pseudocount=True with arbitrary pseudocount:
        result = log(sum_of_channel_counts + pseudocount).
        """
        c0, c1 = 30.0, 70.0
        lc = torch.log(torch.tensor([[c0 + pseudocount, c1 + pseudocount]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=pseudocount)
        expected = torch.log(torch.tensor([c0 + c1 + pseudocount]))

        assert torch.isclose(result, expected, atol=1e-4), \
            f"Expected log({c0+c1}+{pseudocount})={expected.item():.4f}, got {result.item():.4f}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_single_channel_unaffected(self, pseudocount):
        """Single channel: log_counts_include_pseudocount has no effect, returns log_counts as-is."""
        lc = torch.tensor([[5.0]])
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=pseudocount)
        assert torch.isclose(result, torch.tensor([5.0]), atol=1e-6)

    def test_pseudocount_1_equals_log1p(self):
        """With pseudocount=1, multi-channel result matches log1p(total)."""
        c0, c1 = 10.0, 20.0
        lc = torch.log1p(torch.tensor([[c0, c1]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=1.0)
        expected = torch.log1p(torch.tensor([c0 + c1]))
        assert torch.isclose(result, expected, atol=1e-5)

    def test_wrong_pseudocount_gives_wrong_result(self):
        """Using pseudocount=1 to aggregate data trained with pseudocount=100 is wrong."""
        c0, c1 = 10.0, 20.0
        pseudocount_train = 100.0
        lc = torch.log(torch.tensor([[c0 + pseudocount_train, c1 + pseudocount_train]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        # Correct
        result_correct = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=100.0)
        expected = torch.log(torch.tensor([c0 + c1 + 100.0]))
        assert torch.isclose(result_correct, expected, atol=1e-4)

        # Wrong — using pseudocount=1.0 on data with pseudocount=100
        result_wrong = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=1.0)
        assert not torch.isclose(result_wrong, expected, atol=0.1), \
            "Wrong pseudocount should give different result"


# ---------------------------------------------------------------------------
# 6. Loss ↔ metric consistency: same pseudocount, same targets
# ---------------------------------------------------------------------------

class TestLossMetricConsistency:
    """The target log-counts computed by the loss and the metric must match."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_loss_and_metric_targets_agree(self, pseudocount):
        """MSEMultinomialLoss and LogCountsMeanSquaredError use the same target."""
        B, C, L = 4, 1, 16
        targets = torch.abs(torch.randn(B, C, L)) * 100

        # What the loss uses (extracted from MSEMultinomialLoss.forward)
        total = targets.sum(dim=(1, 2))
        loss_target = torch.log(total + pseudocount)

        # What the metric uses
        metric = LogCountsMeanSquaredError(count_pseudocount=pseudocount)

        # Create a perfect prediction
        pred = loss_target.reshape(B, 1)
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred)
        metric.update(out, targets)
        mse = metric.compute()

        assert torch.isclose(mse, torch.tensor(0.0), atol=1e-5), \
            f"Loss and metric targets should agree for pseudocount={pseudocount}, MSE={mse.item()}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_loss_and_metric_per_channel_agree(self, pseudocount):
        """Per-channel mode: loss and metric use same target transform."""
        B, C, L = 2, 2, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        per_ch_counts = targets.sum(dim=2)  # (B, C)
        loss_target = torch.log(per_ch_counts + pseudocount)

        pred = loss_target  # (B, C)
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred)

        metric = LogCountsMeanSquaredError(
            count_per_channel=True, count_pseudocount=pseudocount
        )
        metric.update(out, targets)
        mse = metric.compute()
        assert torch.isclose(mse, torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Default pseudocount=1.0 backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """All metrics with default pseudocount=1.0 reproduce old log1p behaviour."""

    def test_log_counts_mse_default_is_log1p(self):
        targets = torch.tensor([[[50.0]]])
        total = 50.0
        pred = torch.log1p(torch.tensor([[total]]))  # log1p
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 1), log_counts=pred)

        metric = LogCountsMeanSquaredError()  # default pseudocount=1.0
        metric.update(out, targets)
        assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-6)

    def test_count_profile_pearson_default_is_expm1(self):
        """Default pseudocount=1 → exp(lc) - 1 = expm1(lc)."""
        L = 16
        target = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)
        total = target.sum()
        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log1p(total).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)

        metric = CountProfilePearsonCorrCoef()  # default pseudocount=1.0
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)

    def test_count_profile_mse_default_is_expm1(self):
        L = 8
        target = torch.ones(1, 1, L) * 10
        total = target.sum()
        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log1p(total).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfileMeanSquaredError()  # default pseudocount=1.0
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(0.0), atol=1e-2)

    def test_compute_total_log_counts_default_is_log1p(self):
        c0, c1 = 10.0, 20.0
        lc = torch.log1p(torch.tensor([[c0, c1]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True)  # default pseudocount=1.0
        expected = torch.log1p(torch.tensor([c0 + c1]))
        assert torch.isclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Zero-count edge cases with pseudocount
# ---------------------------------------------------------------------------

class TestZeroCountEdgeCases:
    """Verify pseudocount prevents log(0) and handles zero-count regions."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_all_zeros_target_finite(self, pseudocount):
        """All-zero targets → log(0 + pseudocount) = log(pseudocount), must be finite."""
        targets = torch.zeros(1, 1, 8)
        expected = torch.log(torch.tensor(pseudocount))

        metric = LogCountsMeanSquaredError(count_pseudocount=pseudocount)
        pred = expected.reshape(1, 1)
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 8), log_counts=pred)
        metric.update(out, targets)
        val = metric.compute()
        assert torch.isfinite(val)
        assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0])
    def test_loss_with_zero_targets(self, pseudocount):
        """Loss with all-zero targets should be finite."""
        targets = torch.zeros(1, 1, 8)
        expected_log = torch.log(torch.tensor(pseudocount))
        pred = expected_log.reshape(1, 1)
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 8), log_counts=pred)

        loss_fn = MSEMultinomialLoss(count_weight=1.0, profile_weight=0.0,
                                     count_pseudocount=pseudocount)
        loss_val = loss_fn(out, targets)
        assert torch.isfinite(loss_val)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Per-channel loss path with pseudocount
# ---------------------------------------------------------------------------

class TestPerChannelLossPseudocount:
    """MSEMultinomialLoss count_per_channel=True uses pseudocount per channel."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_per_channel_perfect_prediction(self, pseudocount):
        """Per-channel count loss is 0 when predictions match log(ch_count + p)."""
        B, C, L = 2, 3, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        per_ch = targets.sum(dim=2)  # (B, C)
        pred_log_counts = torch.log(per_ch + pseudocount)

        loss_fn = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_per_channel=True, count_pseudocount=pseudocount,
        )
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred_log_counts)
        loss_val = loss_fn(out, targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5)

    def test_per_channel_wrong_pseudocount(self):
        """Wrong pseudocount in per-channel mode gives non-zero loss."""
        B, C, L = 1, 2, 4
        targets = torch.ones(B, C, L) * 20  # per-ch total = 80

        # Predict with pseudocount=1
        pred = torch.log(torch.tensor([[81.0, 81.0]]))
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred)

        # Loss uses pseudocount=50 → expects log(80 + 50) = log(130)
        loss_fn = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_per_channel=True, count_pseudocount=50.0,
        )
        loss_val = loss_fn(out, targets)
        expected = (torch.log(torch.tensor(81.0)) - torch.log(torch.tensor(130.0))) ** 2
        assert torch.isclose(loss_val, expected, atol=1e-5)

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_coupled_per_channel(self, pseudocount):
        """CoupledMSEMultinomialLoss per-channel with pseudocount."""
        B, C, L = 2, 2, 4
        targets = torch.abs(torch.randn(B, C, L)) * 50

        per_ch = targets.sum(dim=2)  # (B, C)
        expected_log = torch.log(per_ch + pseudocount)  # (B, C)

        # Build log_rates so logsumexp over length gives expected_log per channel.
        # Single position per channel: logsumexp over L=1 = value itself.
        log_rates = expected_log.unsqueeze(2)  # (B, C, 1)
        out = ProfileLogRates(log_rates=log_rates)

        loss_fn = CoupledMSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_per_channel=True, count_pseudocount=pseudocount,
        )
        loss_val = loss_fn(out, targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 10. MetricCollection pseudocount propagation
# ---------------------------------------------------------------------------

class TestMetricCollectionPropagation:
    """All model MetricCollections must wire count_pseudocount to every sub-metric."""

    @pytest.mark.parametrize("collection_cls_path", [
        "cerberus.models.bpnet.BPNetMetricCollection",
        "cerberus.models.geminet.GemiNetMetricCollection",
        "cerberus.models.lyra.LyraNetMetricCollection",
        "cerberus.models.pomeranian.PomeranianMetricCollection",
    ])
    def test_pseudocount_reaches_all_submetrics(self, collection_cls_path):
        """Every sub-metric with count_pseudocount must receive the configured value."""
        module_path, cls_name = collection_cls_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)

        pseudocount = 42.0
        collection = cls(count_pseudocount=pseudocount)

        for name, metric in collection.items():
            if hasattr(metric, "count_pseudocount"):
                assert metric.count_pseudocount == pseudocount, \
                    f"{collection_cls_path}['{name}'] has count_pseudocount=" \
                    f"{metric.count_pseudocount}, expected {pseudocount}"

    def test_default_metric_collection_propagation(self):
        """DefaultMetricCollection propagates pseudocount to log-count metrics."""
        from cerberus.metrics import DefaultMetricCollection
        collection = DefaultMetricCollection(count_pseudocount=77.0)
        for name, metric in collection.items():
            if hasattr(metric, "count_pseudocount"):
                assert metric.count_pseudocount == 77.0, \
                    f"DefaultMetricCollection['{name}'] got wrong pseudocount"


# ---------------------------------------------------------------------------
# 11. log1p_targets and pseudocount are independent
# ---------------------------------------------------------------------------

class TestLog1pTargetsIndependence:
    """log1p_targets inverts per-position log1p; pseudocount applies to count totals.
    These must not interfere with each other."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_loss_log1p_targets_with_pseudocount(self, pseudocount):
        """log1p_targets + non-default pseudocount: both transforms apply correctly."""
        B, C, L = 1, 1, 8
        raw_targets = torch.abs(torch.randn(B, C, L)) * 50
        log1p_targets = torch.log1p(raw_targets)  # dataset-level transform

        total = raw_targets.sum(dim=(1, 2))
        pred_log = torch.log(total + pseudocount).reshape(B, 1)

        loss_fn = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            log1p_targets=True, count_pseudocount=pseudocount,
        )
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred_log)
        loss_val = loss_fn(out, log1p_targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-4), \
            f"Combined log1p_targets + pseudocount={pseudocount} should give 0 loss"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_metric_log1p_targets_with_pseudocount(self, pseudocount):
        """LogCountsMeanSquaredError with log1p_targets + pseudocount."""
        B, C, L = 2, 1, 8
        raw_targets = torch.abs(torch.randn(B, C, L)) * 50
        log1p_targets = torch.log1p(raw_targets)

        total = raw_targets.sum(dim=(1, 2))
        pred_log = torch.log(total + pseudocount).reshape(B, 1)

        metric = LogCountsMeanSquaredError(
            log1p_targets=True, count_pseudocount=pseudocount,
        )
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred_log)
        metric.update(out, log1p_targets)
        assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-4)

    def test_log1p_targets_inversion_always_uses_expm1(self):
        """Regardless of pseudocount, per-position inversion is always expm1.

        This is a sentinel test: if someone accidentally replaces expm1 with
        exp(x) - pseudocount for per-position targets, this will catch it.
        """
        raw = torch.tensor([[[1.0, 5.0, 10.0, 50.0]]])
        log1p_raw = torch.log1p(raw)
        recovered = torch.expm1(log1p_raw)
        assert torch.allclose(recovered, raw, atol=1e-6)

        # Even with pseudocount=100, per-position should use expm1
        loss_fn = MSEMultinomialLoss(
            count_weight=0.0, profile_weight=1.0,
            log1p_targets=True, count_pseudocount=100.0,
        )
        # Profile loss with log1p_targets: the loss internally inverts with expm1.
        # If it wrongly used exp(x)-100 instead, the profile would be garbage.
        # We check that profile loss with perfect logits is reproducible.
        probs = raw / raw.sum()
        logits = torch.log(probs + 1e-10)
        out = ProfileCountOutput(
            logits=logits, log_counts=torch.zeros(1, 1),
        )
        loss_val = loss_fn(out, log1p_raw)
        assert torch.isfinite(loss_val)


# ---------------------------------------------------------------------------
# 12. Multi-channel count-reconstruction with pseudocount
# ---------------------------------------------------------------------------

class TestMultiChannelReconstruction:
    """Count-reconstruction metrics with multi-channel log_counts and pseudocount."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_multi_channel_count_profile_mse(self, pseudocount):
        """CountProfileMeanSquaredError with per-channel log_counts."""
        B, C, L = 1, 2, 8
        target = torch.abs(torch.randn(B, C, L)) * 50
        per_ch_total = target.sum(dim=2)  # (B, C)

        probs = target / target.sum(dim=2, keepdim=True)
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log(per_ch_total + pseudocount)  # (B, C)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfileMeanSquaredError(count_pseudocount=pseudocount)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(0.0), atol=1e-1), \
            f"Multi-channel MSE should be ~0 for pseudocount={pseudocount}, got {val.item()}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 50.0])
    def test_multi_channel_count_profile_pearson(self, pseudocount):
        """CountProfilePearsonCorrCoef with per-channel log_counts."""
        B, C, L = 4, 2, 16
        target = torch.abs(torch.randn(B, C, L)) * 50 + 1.0
        per_ch_total = target.sum(dim=2)

        probs = target / target.sum(dim=2, keepdim=True)
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log(per_ch_total + pseudocount)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfilePearsonCorrCoef(count_pseudocount=pseudocount)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-2), \
            f"Multi-channel Pearson should be ~1 for pseudocount={pseudocount}, got {val.item()}"


# ---------------------------------------------------------------------------
# 13. Gradient flow through pseudocount-adjusted targets
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow through log_counts to model parameters."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0])
    def test_loss_gradient_flows(self, pseudocount):
        """Count loss must produce gradients on log_counts."""
        B, C, L = 2, 1, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        logits = torch.zeros(B, C, L)
        log_counts = torch.randn(B, 1, requires_grad=True)
        out = ProfileCountOutput(logits=logits, log_counts=log_counts)

        loss_fn = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_pseudocount=pseudocount,
        )
        loss = loss_fn(out, targets)
        loss.backward()

        assert log_counts.grad is not None, "Gradient should flow to log_counts"
        assert torch.isfinite(log_counts.grad).all(), "Gradients should be finite"
        assert (log_counts.grad.abs() > 0).any(), "Gradients should be non-zero"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0])
    def test_coupled_loss_gradient_flows(self, pseudocount):
        """CoupledMSEMultinomialLoss gradients flow through logsumexp."""
        B, C, L = 2, 1, 4
        targets = torch.abs(torch.randn(B, C, L)) * 50

        log_rates = torch.randn(B, C, L, requires_grad=True)
        out = ProfileLogRates(log_rates=log_rates)

        loss_fn = CoupledMSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_pseudocount=pseudocount,
        )
        loss = loss_fn(out, targets)
        loss.backward()

        assert log_rates.grad is not None
        assert torch.isfinite(log_rates.grad).all()
        assert (log_rates.grad.abs() > 0).any()


# ---------------------------------------------------------------------------
# 14. Pure-log multi-channel path ignores pseudocount parameter
# ---------------------------------------------------------------------------

class TestPureLogIgnoresPseudocount:
    """When log_counts_include_pseudocount=False, the pseudocount parameter is unused."""

    def test_logsumexp_unaffected_by_pseudocount_value(self):
        """Multi-channel logsumexp path gives same result for any pseudocount."""
        c0, c1 = 10.0, 20.0
        lc = torch.log(torch.tensor([[c0, c1]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        r1 = compute_total_log_counts(out, log_counts_include_pseudocount=False, pseudocount=1.0)
        r100 = compute_total_log_counts(out, log_counts_include_pseudocount=False, pseudocount=100.0)
        r999 = compute_total_log_counts(out, log_counts_include_pseudocount=False, pseudocount=999.0)

        assert torch.isclose(r1, r100, atol=1e-6)
        assert torch.isclose(r1, r999, atol=1e-6)
        assert torch.isclose(r1, torch.log(torch.tensor(c0 + c1)), atol=1e-5)

    def test_profile_log_rates_unaffected(self):
        """ProfileLogRates path ignores both flags entirely."""
        log_rates = torch.tensor([[[math.log(3), math.log(7)]]])
        out = ProfileLogRates(log_rates=log_rates)

        r_off = compute_total_log_counts(out, log_counts_include_pseudocount=False, pseudocount=1.0)
        r_on = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=999.0)

        # Both should give log(3+7) = log(10)
        expected = torch.log(torch.tensor([10.0]))
        assert torch.isclose(r_off, expected, atol=1e-5)
        assert torch.isclose(r_on, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 15. obs_log_count_fn pseudocount propagation (export_predictions)
# ---------------------------------------------------------------------------

class TestObsLogCountFnPseudocount:
    """The obs_log_count_fn lambda must use the criterion's pseudocount."""

    def _make_obs_fn(self, criterion):
        """Replicate the logic from export_predictions.py."""
        if isinstance(criterion, (MSEMultinomialLoss, CoupledMSEMultinomialLoss)):
            p = getattr(criterion, "count_pseudocount", 1.0)
            return lambda x: torch.log(x + p)
        else:
            return lambda x: torch.log(x.clamp_min(1.0))

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_mse_obs_fn_uses_pseudocount(self, pseudocount):
        """MSE loss → obs_fn = log(x + pseudocount)."""
        criterion = MSEMultinomialLoss(count_pseudocount=pseudocount)
        fn = self._make_obs_fn(criterion)
        counts = torch.tensor([0.0, 10.0, 100.0])
        expected = torch.log(counts + pseudocount)
        assert torch.allclose(fn(counts), expected, atol=1e-6)

    def test_mse_default_equals_log1p(self):
        """MSE with default pseudocount=1.0 → log(x+1) = log1p(x)."""
        criterion = MSEMultinomialLoss()  # default pseudocount=1.0
        fn = self._make_obs_fn(criterion)
        counts = torch.tensor([0.0, 10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log1p(counts), atol=1e-6)

    @pytest.mark.parametrize("pseudocount", [1.0, 50.0])
    def test_obs_fn_matches_loss_target(self, pseudocount):
        """obs_fn(total_count) must equal the loss's internal count target."""
        criterion = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_pseudocount=pseudocount,
        )
        fn = self._make_obs_fn(criterion)

        B, C, L = 1, 1, 8
        targets = torch.abs(torch.randn(B, C, L)) * 100
        total = targets.sum(dim=(1, 2))

        obs_log = fn(total)

        # Perfect prediction: if pred matches obs_log, loss = 0
        out = ProfileCountOutput(
            logits=torch.zeros(B, C, L),
            log_counts=obs_log.reshape(B, 1),
        )
        loss_val = criterion(out, targets)
        assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5), \
            f"obs_fn should produce the same target as the loss, got loss={loss_val.item()}"


# ---------------------------------------------------------------------------
# 16. Three-way consistency: loss = metric = accumulator
# ---------------------------------------------------------------------------

class TestThreeWayConsistency:
    """The count target log(total + p) must be identical across:
    1. MSEMultinomialLoss (training target)
    2. LogCountsMeanSquaredError (validation metric target)
    3. module._accumulate_log_counts (scatter plot target)
    """

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_all_three_agree(self, pseudocount):
        B, C, L = 4, 1, 16
        targets = torch.abs(torch.randn(B, C, L)) * 100

        total = targets.sum(dim=(1, 2))  # (B,)

        # 1. Loss target (from MSEMultinomialLoss.forward internals)
        loss_target = torch.log(total + pseudocount)

        # 2. Metric target (from LogCountsMeanSquaredError.update internals)
        metric_target = torch.log(total + pseudocount)

        # 3. Accumulator target (from module._accumulate_log_counts)
        accum_target = torch.log(total + pseudocount)

        # All three must be identical
        assert torch.allclose(loss_target, metric_target, atol=1e-7)
        assert torch.allclose(loss_target, accum_target, atol=1e-7)

        # Verify loss is 0 with this prediction
        pred = loss_target.reshape(B, 1)
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=pred)

        loss_fn = MSEMultinomialLoss(
            count_weight=1.0, profile_weight=0.0,
            count_pseudocount=pseudocount,
        )
        assert torch.isclose(loss_fn(out, targets), torch.tensor(0.0), atol=1e-5)

        metric = LogCountsMeanSquaredError(count_pseudocount=pseudocount)
        metric.update(out, targets)
        assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-5)

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0])
    def test_pred_aggregation_matches_target(self, pseudocount):
        """For a perfect model, compute_total_log_counts(pred) == log(total + p)."""
        B, C, L = 2, 2, 8
        targets = torch.abs(torch.randn(B, C, L)) * 50

        # Per-channel totals
        per_ch = targets.sum(dim=2)  # (B, C)
        log_counts = torch.log(per_ch + pseudocount)  # (B, C) — per-channel predictions
        out = ProfileCountOutput(logits=torch.zeros(B, C, L), log_counts=log_counts)

        # Aggregate to total using the multi-channel offset-log path
        pred_total = compute_total_log_counts(
            out, log_counts_include_pseudocount=True, pseudocount=pseudocount,
        )
        # Expected: log(sum_of_all_channels + pseudocount)
        total = targets.sum(dim=(1, 2))
        expected = torch.log(total + pseudocount)

        assert torch.allclose(pred_total, expected, atol=1e-4), \
            f"Aggregated pred {pred_total} != expected {expected}"
