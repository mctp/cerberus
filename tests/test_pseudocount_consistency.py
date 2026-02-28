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
    PerExampleCountProfilePearsonCorrCoef,
    PerExampleLogCountsPearsonCorrCoef,
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

        metric = PerExampleLogCountsPearsonCorrCoef(count_pseudocount=pseudocount)

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
    PerExampleCountProfilePearsonCorrCoef reconstruct counts from log_counts
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
        metric = CountProfilePearsonCorrCoef(num_channels=C, count_pseudocount=pseudocount)
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
        metric = PerExampleCountProfilePearsonCorrCoef(
            num_channels=C, count_pseudocount=pseudocount
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
    """Verify multi-channel aggregation with implicit_log uses the pseudocount."""

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_multi_channel_implicit_log(self, pseudocount):
        """
        Multi-channel implicit_log=True with arbitrary pseudocount:
        result = log(sum_of_channel_counts + pseudocount).
        """
        c0, c1 = 30.0, 70.0
        lc = torch.log(torch.tensor([[c0 + pseudocount, c1 + pseudocount]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        result = compute_total_log_counts(out, implicit_log=True, pseudocount=pseudocount)
        expected = torch.log(torch.tensor([c0 + c1 + pseudocount]))

        assert torch.isclose(result, expected, atol=1e-4), \
            f"Expected log({c0+c1}+{pseudocount})={expected.item():.4f}, got {result.item():.4f}"

    @pytest.mark.parametrize("pseudocount", [1.0, 10.0, 100.0])
    def test_single_channel_unaffected(self, pseudocount):
        """Single channel: implicit_log has no effect, returns log_counts as-is."""
        lc = torch.tensor([[5.0]])
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        result = compute_total_log_counts(out, implicit_log=True, pseudocount=pseudocount)
        assert torch.isclose(result, torch.tensor([5.0]), atol=1e-6)

    def test_pseudocount_1_equals_log1p(self):
        """With pseudocount=1, multi-channel result matches log1p(total)."""
        c0, c1 = 10.0, 20.0
        lc = torch.log1p(torch.tensor([[c0, c1]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        result = compute_total_log_counts(out, implicit_log=True, pseudocount=1.0)
        expected = torch.log1p(torch.tensor([c0 + c1]))
        assert torch.isclose(result, expected, atol=1e-5)

    def test_wrong_pseudocount_gives_wrong_result(self):
        """Using pseudocount=1 to aggregate data trained with pseudocount=100 is wrong."""
        c0, c1 = 10.0, 20.0
        pseudocount_train = 100.0
        lc = torch.log(torch.tensor([[c0 + pseudocount_train, c1 + pseudocount_train]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        # Correct
        result_correct = compute_total_log_counts(out, implicit_log=True, pseudocount=100.0)
        expected = torch.log(torch.tensor([c0 + c1 + 100.0]))
        assert torch.isclose(result_correct, expected, atol=1e-4)

        # Wrong — using pseudocount=1.0 on data with pseudocount=100
        result_wrong = compute_total_log_counts(out, implicit_log=True, pseudocount=1.0)
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

        metric = CountProfilePearsonCorrCoef(num_channels=1)  # default pseudocount=1.0
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
        result = compute_total_log_counts(out, implicit_log=True)  # default pseudocount=1.0
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
