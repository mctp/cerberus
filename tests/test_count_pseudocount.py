"""
Comprehensive tests for count_pseudocount behaviour across loss, metrics, output
utilities, and config injection.

Background
----------
Coverage counts are sums of per-bp BigWig signal over an output window.  A single
sequencing read (~100 bp) contributes its overlap length to this sum, so the minimum
observable non-zero count is approximately ``read_length``, not 1.  Using ``log1p``
(i.e. ``log(x + 1)``) therefore compresses the biologically meaningful dynamic range.

The ``count_pseudocount`` parameter replaces the hard-coded ``+1`` with a configurable
offset set once in ``DataConfig`` and automatically propagated (after scaling by
``target_scale``) into ``loss_args`` and ``metrics_args`` by ``propagate_pseudocount``
(called from ``instantiate()`` where the loss and metrics are constructed).
"""

import math

import pytest
import torch

from cerberus.loss import CoupledMSEMultinomialLoss, MSEMultinomialLoss
from cerberus.metrics import (
    DefaultMetricCollection,
    LogCountsMeanSquaredError,
    LogCountsPearsonCorrCoef,
)
from cerberus.models.bpnet import BPNetLoss, BPNetMetricCollection
from cerberus.models.pomeranian import PomeranianMetricCollection
from cerberus.output import (
    ProfileCountOutput,
    ProfileLogRates,
    compute_total_log_counts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_count_output(batch=1, channels=1, length=10, log_counts=None):
    """Return a ProfileCountOutput with flat logits and given log_counts."""
    logits = torch.zeros(batch, channels, length)
    if log_counts is None:
        log_counts = torch.zeros(batch, channels)
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


def _make_targets(batch=1, channels=1, length=10, total_per_channel=100.0):
    """Return targets with ``total_per_channel`` concentrated in the first bin."""
    targets = torch.zeros(batch, channels, length)
    targets[:, :, 0] = total_per_channel
    return targets


# ===========================================================================
# 1. MSEMultinomialLoss — forward path
# ===========================================================================

class TestMSEMultinomialLossForward:
    """count_pseudocount affects the count-loss target in the forward pass.

    Use profile_weight=0.0 to isolate count loss and avoid confounding
    Multinomial NLL when targets are not uniform.
    """

    def test_perfect_prediction_pseudocount_1(self):
        """Default pseudocount=1 → log(count+1); perfect pred gives count_loss=0."""
        total = 500.0
        loss_fn = MSEMultinomialLoss(count_pseudocount=1.0, profile_weight=0.0)
        targets = _make_targets(total_per_channel=total)
        pred = torch.log(torch.tensor([[total + 1.0]]))
        out = _make_count_output(log_counts=pred)
        assert loss_fn(out, targets).item() == pytest.approx(0.0, abs=1e-5)

    def test_perfect_prediction_pseudocount_100(self):
        """pseudocount=100 → log(count+100); perfect pred gives count_loss=0."""
        total = 500.0
        loss_fn = MSEMultinomialLoss(count_pseudocount=100.0, profile_weight=0.0)
        targets = _make_targets(total_per_channel=total)
        pred = torch.log(torch.tensor([[total + 100.0]]))
        out = _make_count_output(log_counts=pred)
        assert loss_fn(out, targets).item() == pytest.approx(0.0, abs=1e-5)

    def test_wrong_pseudocount_gives_nonzero_loss(self):
        """A prediction calibrated for pseudocount=1 gives non-zero loss when pseudocount=100."""
        total = 500.0
        loss_fn = MSEMultinomialLoss(count_pseudocount=100.0, profile_weight=0.0)
        targets = _make_targets(total_per_channel=total)
        # Prediction calibrated for pseudocount=1 (log(501)), not log(600)
        pred = torch.log(torch.tensor([[total + 1.0]]))
        out = _make_count_output(log_counts=pred)
        assert loss_fn(out, targets).item() > 1e-3

    def test_zero_count_target_finite(self):
        """With total=0 and pseudocount=100, target is log(100) — finite, not -inf."""
        loss_fn = MSEMultinomialLoss(count_pseudocount=100.0, profile_weight=0.0)
        targets = torch.zeros(1, 1, 10)  # all zero coverage
        pred = torch.log(torch.tensor([[100.0]]))  # perfect pred
        out = _make_count_output(log_counts=pred)
        val = loss_fn(out, targets)
        assert torch.isfinite(val)
        assert val.item() == pytest.approx(0.0, abs=1e-5)

    def test_per_channel_pseudocount_100(self):
        """count_per_channel=True with pseudocount=100 sets target per channel."""
        loss_fn = MSEMultinomialLoss(count_pseudocount=100.0, count_per_channel=True, profile_weight=0.0)
        total_ch0, total_ch1 = 200.0, 300.0
        targets = torch.zeros(1, 2, 10)
        targets[0, 0, 0] = total_ch0
        targets[0, 1, 0] = total_ch1
        # Perfect per-channel predictions
        pred = torch.log(torch.tensor([[total_ch0 + 100.0, total_ch1 + 100.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred)
        assert loss_fn(out, targets).item() == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# 2. CoupledMSEMultinomialLoss — forward path (inherits count_pseudocount)
# ===========================================================================

class TestCoupledMSEMultinomialLossForward:
    """CoupledMSEMultinomialLoss inherits __init__ from MSEMultinomialLoss."""

    def test_default_pseudocount_stored(self):
        loss_fn = CoupledMSEMultinomialLoss()
        assert loss_fn.count_pseudocount == 1.0

    def test_custom_pseudocount_stored(self):
        loss_fn = CoupledMSEMultinomialLoss(count_pseudocount=100.0)
        assert loss_fn.count_pseudocount == 100.0

    def test_perfect_prediction_pseudocount_100(self):
        """pseudocount=100; perfect logsumexp pred gives count loss ≈ 0."""
        total = 500.0
        loss_fn = CoupledMSEMultinomialLoss(count_pseudocount=100.0)
        # log_rates: single channel, one bin with log(total) so logsumexp = log(total)
        log_rates = torch.log(torch.tensor([[[total]]]))  # (1, 1, 1)
        out = ProfileLogRates(log_rates=log_rates)
        targets = torch.zeros(1, 1, 1)
        targets[0, 0, 0] = total
        # target_log_count = log(total + 100) = log(600)
        # pred_log_count = logsumexp([[log(500)]]) = log(500)
        expected_count_loss = (math.log(total) - math.log(total + 100.0)) ** 2
        val = loss_fn(out, targets).item()
        # There's also profile loss; count and profile weights both = 1
        # Profile loss from Multinomial NLL with single bin = 0 (logit=log(500), target=500)
        # Actually for multinomial NLL with a single bin, softmax is 1.0, log_prob = 0.
        # So val ≈ count_loss ≈ expected_count_loss.
        # The profile component is 0 for a single-bin case.
        assert val == pytest.approx(expected_count_loss, rel=1e-4)

    def test_zero_count_target_finite(self):
        """zero coverage + pseudocount=100 → target log(100); not -inf."""
        loss_fn = CoupledMSEMultinomialLoss(count_pseudocount=100.0)
        log_rates = torch.tensor([[[math.log(100.0)]]])  # pred = log(100)
        out = ProfileLogRates(log_rates=log_rates)
        targets = torch.zeros(1, 1, 1)
        val = loss_fn(out, targets)
        assert torch.isfinite(val)


# ===========================================================================
# 3. BPNetLoss — count_pseudocount flows through **kwargs
# ===========================================================================

class TestBPNetLoss:
    """BPNetLoss passes unknown kwargs to MSEMultinomialLoss.__init__."""

    def test_default_pseudocount_stored(self):
        loss_fn = BPNetLoss()
        assert loss_fn.count_pseudocount == 1.0

    def test_custom_pseudocount_stored(self):
        loss_fn = BPNetLoss(count_pseudocount=100.0)
        assert loss_fn.count_pseudocount == 100.0

    def test_perfect_prediction_pseudocount_100(self):
        """count_pseudocount=100 from BPNetLoss.count_pseudocount used in forward."""
        total = 500.0
        # BPNetLoss uses beta= for profile weight; beta=0 isolates count loss
        loss_fn = BPNetLoss(count_pseudocount=100.0, beta=0.0)
        targets = _make_targets(total_per_channel=total)
        pred = torch.log(torch.tensor([[total + 100.0]]))
        out = _make_count_output(log_counts=pred)
        assert loss_fn(out, targets).item() == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 4. LogCountsPearsonCorrCoef — pseudocount
# ===========================================================================

class TestLogCountsPearsonCorrCoef:
    """LogCountsPearsonCorrCoef collects scalar pairs per example."""

    def test_default_pseudocount_stored(self):
        m = LogCountsPearsonCorrCoef()
        assert m.count_pseudocount == 1.0

    def test_custom_pseudocount_stored(self):
        m = LogCountsPearsonCorrCoef(count_pseudocount=100.0)
        assert m.count_pseudocount == 100.0

    def test_correlation_is_one_with_perfect_predictions(self):
        """Pearson r=1 when all pred/target log-count pairs are perfectly aligned."""
        m = LogCountsPearsonCorrCoef(count_pseudocount=100.0)
        totals = [100.0, 300.0, 600.0]
        for total in totals:
            target = torch.zeros(1, 1, 10)
            target[0, 0, 0] = total
            pred_lc = torch.log(torch.tensor([[total + 100.0]]))
            out = _make_count_output(log_counts=pred_lc)
            m.update(out, target)
        val = m.compute()
        assert torch.isfinite(val)
        assert val.item() == pytest.approx(1.0, abs=1e-4)

    def test_per_channel_pseudocount(self):
        """count_per_channel=True collects per-channel log-count pairs."""
        m = LogCountsPearsonCorrCoef(count_pseudocount=50.0, count_per_channel=True)
        totals_ch = [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]
        for totals in totals_ch:
            target = torch.zeros(1, 2, 10)
            target[0, 0, 0] = totals[0]
            target[0, 1, 0] = totals[1]
            pred_lc = torch.log(torch.tensor([[totals[0] + 50.0, totals[1] + 50.0]]))
            out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_lc)
            m.update(out, target)
        val = m.compute()
        assert torch.isfinite(val)


# ===========================================================================
# 6. LogCountsMeanSquaredError — count_per_channel + custom pseudocount
# ===========================================================================

class TestLogCountsMSEPerChannel:
    """count_per_channel=True with custom count_pseudocount."""

    def test_per_channel_perfect_prediction(self):
        m = LogCountsMeanSquaredError(count_per_channel=True, count_pseudocount=100.0)
        targets = torch.zeros(1, 2, 10)
        targets[0, 0, 0] = 200.0
        targets[0, 1, 0] = 400.0
        pred_lc = torch.log(torch.tensor([[300.0, 500.0]]))  # +100 each
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_lc)
        m.update(out, targets)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_per_channel_nonzero_error(self):
        m = LogCountsMeanSquaredError(count_per_channel=True, count_pseudocount=100.0)
        targets = torch.zeros(1, 2, 10)
        targets[0, 0, 0] = 200.0
        targets[0, 1, 0] = 400.0
        # Wrong: use pseudocount=1 instead of 100
        pred_lc = torch.log(torch.tensor([[201.0, 401.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_lc)
        m.update(out, targets)
        val = m.compute()
        expected_ch0 = (math.log(201.0) - math.log(300.0)) ** 2
        expected_ch1 = (math.log(401.0) - math.log(500.0)) ** 2
        expected = (expected_ch0 + expected_ch1) / 2
        assert val.item() == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# 7. MetricCollections — count_pseudocount propagated to inner metrics
# ===========================================================================

class TestMetricCollectionPropagation:
    """Verify count_pseudocount is stored on each inner LogCounts* metric."""

    @pytest.mark.parametrize("cls,kwargs", [
        (DefaultMetricCollection, {}),
        (BPNetMetricCollection, {}),
        (PomeranianMetricCollection, {}),
    ])
    def test_default_pseudocount(self, cls, kwargs):
        mc = cls(**kwargs)
        for name, metric in mc.items():
            if hasattr(metric, "count_pseudocount"):
                assert metric.count_pseudocount == 1.0, (
                    f"{cls.__name__}.{name} has wrong default count_pseudocount"
                )

    @pytest.mark.parametrize("cls,kwargs", [
        (DefaultMetricCollection, {}),
        (BPNetMetricCollection, {}),
        (PomeranianMetricCollection, {}),
    ])
    def test_custom_pseudocount_propagated(self, cls, kwargs):
        mc = cls(**kwargs, count_pseudocount=100.0)
        for name, metric in mc.items():
            if hasattr(metric, "count_pseudocount"):
                assert metric.count_pseudocount == 100.0, (
                    f"{cls.__name__}.{name}: expected count_pseudocount=100.0"
                )


# ===========================================================================
# 8. compute_total_log_counts — log_counts_include_pseudocount path
# ===========================================================================

class TestComputeTotalLogCountsPseudocount:
    """Multi-channel inversion using the pseudocount offset."""

    def test_single_channel_passthrough(self):
        """Single-channel: log_counts returned directly (no aggregation)."""
        lc = torch.tensor([[math.log(600.0)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=100.0)
        assert result.item() == pytest.approx(math.log(600.0), abs=1e-5)

    def test_multi_channel_correct_aggregation(self):
        """Two channels: invert per channel, sum, re-log — not logsumexp."""
        # Channel 0: count=200, channel 1: count=400 → total=600
        # log_counts stores log(count + 100):
        lc = torch.tensor([[math.log(300.0), math.log(500.0)]])  # (1, 2)
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=100.0)
        expected = math.log(600.0 + 100.0)  # log(700)
        assert result.item() == pytest.approx(expected, abs=1e-4)

    def test_multi_channel_logsumexp_default_is_wrong(self):
        """Without the flag, logsumexp gives incorrect answer for pseudocount space."""
        lc = torch.tensor([[math.log(300.0), math.log(500.0)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=lc)
        logsumexp_result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        correct_result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=100.0)
        # They should differ
        assert abs(logsumexp_result.item() - correct_result.item()) > 0.1

    def test_zero_counts_clamp(self):
        """exp(log_count) - pseudocount can go negative; clamp_min(0) prevents negative totals."""
        # log_count = log(50) meaning count = 50 - 100 = -50 → clamped to 0
        lc = torch.tensor([[math.log(50.0), math.log(200.0)]])  # ch0 underflows, ch1=100
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=100.0)
        expected = math.log(0.0 + 100.0 + 100.0)  # ch0 clamped to 0; ch1=100; total=100; +100=200
        assert result.item() == pytest.approx(expected, abs=1e-4)

    def test_pseudocount_1_matches_log1p_behaviour(self):
        """pseudocount=1 reproduces original log1p / expm1 round-trip."""
        # Original code: total = expm1(lc).sum(); return log1p(total)
        # New code: total = (exp(lc) - 1).clamp_min(0).sum(); return log(total + 1)
        counts = [200.0, 400.0]
        lc = torch.tensor([[math.log(counts[0] + 1.0), math.log(counts[1] + 1.0)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True, pseudocount=1.0)
        expected = math.log1p(sum(counts))
        assert result.item() == pytest.approx(expected, abs=1e-4)


# ===========================================================================
# 9. Config injection — count_pseudocount * target_scale ends up in loss/metrics
# ===========================================================================

class TestConfigInjection:
    """Verify the setdefault injection logic used in propagate_pseudocount.

    The full instantiate() requires model classes and many config sections.
    These tests exercise the injection logic in isolation by simulating the
    relevant part of propagate_pseudocount.
    """

    def _inject(self, count_pseudocount, target_scale, loss_args=None, metrics_args=None):
        """Reproduce the injection done by propagate_pseudocount."""
        if loss_args is None:
            loss_args = {}
        if metrics_args is None:
            metrics_args = {}
        scaled = count_pseudocount * target_scale
        loss_args.setdefault("count_pseudocount", scaled)
        metrics_args.setdefault("count_pseudocount", scaled)
        return loss_args, metrics_args

    def test_pseudocount_injected_into_loss(self):
        """count_pseudocount * target_scale ends up in loss_args when absent."""
        loss_args, _ = self._inject(100.0, 1.0)
        assert loss_args["count_pseudocount"] == pytest.approx(100.0)

    def test_pseudocount_injected_into_metrics(self):
        """count_pseudocount * target_scale ends up in metrics_args when absent."""
        _, metrics_args = self._inject(100.0, 1.0)
        assert metrics_args["count_pseudocount"] == pytest.approx(100.0)

    def test_pseudocount_scaled_by_target_scale(self):
        """When target_scale=0.5, injected pseudocount = count_pseudocount * 0.5."""
        loss_args, metrics_args = self._inject(100.0, 0.5)
        assert loss_args["count_pseudocount"] == pytest.approx(50.0)
        assert metrics_args["count_pseudocount"] == pytest.approx(50.0)

    def test_explicit_loss_arg_not_overridden(self):
        """An explicit count_pseudocount in loss_args takes precedence over injection."""
        loss_args, metrics_args = self._inject(
            100.0, 1.0,
            loss_args={"count_pseudocount": 999.0},
        )
        # loss_args had explicit value — must not be overwritten
        assert loss_args["count_pseudocount"] == pytest.approx(999.0)
        # metrics_args had no explicit value — gets injected value
        assert metrics_args["count_pseudocount"] == pytest.approx(100.0)

    def test_injected_pseudocount_instantiates_correct_loss(self):
        """Injected count_pseudocount flows into loss class at instantiation."""
        loss_args, _ = self._inject(100.0, 2.0)  # scaled = 200.0
        loss_fn = MSEMultinomialLoss(**loss_args)
        assert loss_fn.count_pseudocount == pytest.approx(200.0)

    def test_injected_pseudocount_instantiates_correct_metric(self):
        """Injected count_pseudocount flows into metric class at instantiation."""
        _, metrics_args = self._inject(100.0, 2.0)  # scaled = 200.0
        m = LogCountsMeanSquaredError(**metrics_args)
        assert m.count_pseudocount == pytest.approx(200.0)


# ===========================================================================
# 10. Zero-count edge cases — all metric/loss classes
# ===========================================================================

class TestZeroCountEdgeCases:
    """With large pseudocount, zero-coverage targets produce finite log-counts."""

    def test_mse_multinomial_zero_target(self):
        """log(0 + 100) = log(100) is finite; loss is finite and correct."""
        loss_fn = MSEMultinomialLoss(count_pseudocount=100.0)
        targets = torch.zeros(1, 1, 10)
        pred = torch.log(torch.tensor([[100.0]]))
        out = _make_count_output(log_counts=pred)
        val = loss_fn(out, targets)
        assert torch.isfinite(val)
        assert val.item() == pytest.approx(0.0, abs=1e-5)

    def test_log_counts_mse_zero_target(self):
        """LogCountsMeanSquaredError: log(0 + 100) is the target."""
        m = LogCountsMeanSquaredError(count_pseudocount=100.0)
        targets = torch.zeros(1, 1, 10)
        pred_lc = torch.log(torch.tensor([[100.0]]))
        out = _make_count_output(log_counts=pred_lc)
        m.update(out, targets)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_per_example_log_counts_pearson_zero_target(self):
        """LogCountsPearsonCorrCoef handles all-zero targets gracefully."""
        m = LogCountsPearsonCorrCoef(count_pseudocount=100.0)
        for total in [0.0, 200.0, 500.0]:
            target = torch.zeros(1, 1, 10)
            target[0, 0, 0] = total
            pred_lc = torch.log(torch.tensor([[total + 100.0]]))
            out = _make_count_output(log_counts=pred_lc)
            m.update(out, target)
        val = m.compute()
        assert torch.isfinite(val)
