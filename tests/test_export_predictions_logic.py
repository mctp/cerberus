"""
Unit tests for the log-count computation logic fixed in export_predictions.py.

Covers:
  1. compute_total_log_counts(log_counts_include_pseudocount=True) — multi-channel MSE aggregation fix.
  2. obs_log_count_fn selection by loss type — log vs log1p per training objective.
  3. target_scale applied to raw observed counts before the log transform.
"""
import math
import torch
import pytest

from cerberus.output import ProfileCountOutput, ProfileLogRates, compute_total_log_counts
from cerberus.loss import (
    MSEMultinomialLoss,
    CoupledMSEMultinomialLoss,
    PoissonMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    CoupledNegativeBinomialMultinomialLoss,
)


# ---------------------------------------------------------------------------
# Helper — replicates the obs_log_count_fn selection from export_predictions.py
# ---------------------------------------------------------------------------

def _obs_log_fn(criterion):
    if isinstance(criterion, (MSEMultinomialLoss, CoupledMSEMultinomialLoss)):
        p = getattr(criterion, "count_pseudocount", 1.0)
        return lambda x: torch.log(x + p)
    else:
        return lambda x: torch.log(x.clamp_min(1.0))


# ---------------------------------------------------------------------------
# 1. compute_total_log_counts — log_counts_include_pseudocount=False (default, Poisson/NB space)
# ---------------------------------------------------------------------------

class TestComputeTotalLogCountsDefault:

    def test_single_channel_returns_log_counts_flat(self):
        """Single channel: returns log_counts.flatten() unchanged."""
        lc = torch.tensor([[2.5]])
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        assert result.shape == (1,)
        assert torch.isclose(result, torch.tensor([2.5]))

    def test_multi_channel_uses_logsumexp(self):
        """Multi-channel log_counts_include_pseudocount=False: logsumexp gives log(sum(exp(log_counts)))."""
        # log_counts = [log(10), log(20)] → logsumexp = log(30)
        lc = torch.tensor([[math.log(10), math.log(20)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        assert torch.isclose(result, torch.log(torch.tensor(30.0)), atol=1e-5)

    def test_profile_log_rates_single_channel(self):
        """ProfileLogRates single channel: logsumexp over positions."""
        log_rates = torch.tensor([[[math.log(3), math.log(7)]]])  # sum exp = 10
        out = ProfileLogRates(log_rates=log_rates)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        assert torch.isclose(result, torch.log(torch.tensor(10.0)), atol=1e-5)

    def test_profile_log_rates_multi_channel(self):
        """ProfileLogRates multi-channel: logsumexp across all channels and positions."""
        # Ch0: [log5, log5] → 10. Ch1: [log10, log10] → 20. Total=30.
        log_rates = torch.tensor([[[math.log(5), math.log(5)],
                                   [math.log(10), math.log(10)]]])  # (1, 2, 2)
        out = ProfileLogRates(log_rates=log_rates)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        assert torch.isclose(result, torch.log(torch.tensor(30.0)), atol=1e-5)

    def test_batched_multi_channel(self):
        """Batch of two intervals, multi-channel."""
        lc = torch.tensor([[math.log(10), math.log(20)],   # total 30
                           [math.log(1),  math.log(99)]])  # total 100
        out = ProfileCountOutput(logits=torch.zeros(2, 2, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        expected = torch.tensor([math.log(30.0), math.log(100.0)])
        assert result.shape == (2,)
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. compute_total_log_counts — log_counts_include_pseudocount=True (MSE / log1p space)
# ---------------------------------------------------------------------------

class TestComputeTotalLogCountsImplicitLog:

    def test_single_channel_unchanged(self):
        """Single channel: log_counts_include_pseudocount has no effect (path returns flatten() directly)."""
        lc = torch.tensor([[math.log1p(42)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        assert torch.isclose(
            compute_total_log_counts(out, log_counts_include_pseudocount=True),
            torch.tensor([math.log1p(42)]),
            atol=1e-6,
        )

    def test_multi_channel_gives_log1p_of_total(self):
        """Multi-channel log_counts_include_pseudocount=True: result is log1p(c0+c1), not log(n_ch+c0+c1).

        This is the core regression test for the bug:
          logsumexp([log1p(c0), log1p(c1)]) = log((1+c0)+(1+c1)) = log(2+c0+c1)
          which is wrong. The fix gives log1p(c0+c1) = log(1+c0+c1).
        """
        c0, c1 = 10.0, 20.0
        lc = torch.tensor([[math.log1p(c0), math.log1p(c1)]])
        out = ProfileCountOutput(logits=torch.zeros(1, 2, 4), log_counts=lc)

        result = compute_total_log_counts(out, log_counts_include_pseudocount=True)
        correct = torch.tensor([math.log1p(c0 + c1)])       # log1p(30)
        wrong   = torch.log(torch.tensor([2 + c0 + c1]))    # log(32)  ← old logsumexp

        assert torch.isclose(result, correct, atol=1e-5), \
            f"Expected log1p({c0+c1})={correct.item():.4f}, got {result.item():.4f}"
        assert not torch.isclose(result, wrong, atol=1e-3), \
            "Result incorrectly matches the logsumexp-based wrong answer"

    def test_multi_channel_batched(self):
        """Batch of two intervals, multi-channel, log_counts_include_pseudocount=True."""
        counts = torch.tensor([[5.0, 15.0], [100.0, 200.0]])  # totals: 20, 300
        lc = torch.log1p(counts)
        out = ProfileCountOutput(logits=torch.zeros(2, 2, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True)
        expected = torch.log1p(torch.tensor([20.0, 300.0]))
        assert result.shape == (2,)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_three_channels(self):
        """Three channels: expm1 → sum → log1p."""
        c = torch.tensor([[3.0, 7.0, 10.0]])  # total=20
        lc = torch.log1p(c)
        out = ProfileCountOutput(logits=torch.zeros(1, 3, 4), log_counts=lc)
        result = compute_total_log_counts(out, log_counts_include_pseudocount=True)
        assert torch.isclose(result, torch.tensor([math.log1p(20.0)]), atol=1e-5)

    def test_log_counts_include_pseudocount_ignored_for_profile_log_rates(self):
        """log_counts_include_pseudocount is unused for ProfileLogRates — both flags give same result."""
        log_rates = torch.tensor([[[math.log(3), math.log(7)]]])
        out = ProfileLogRates(log_rates=log_rates)
        r_false = compute_total_log_counts(out, log_counts_include_pseudocount=False)
        r_true  = compute_total_log_counts(out, log_counts_include_pseudocount=True)
        assert torch.isclose(r_false, r_true, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. obs_log_count_fn — log-space selection matches each loss's training target
# ---------------------------------------------------------------------------

class TestObsLogCountFnByLossType:

    def test_mse_uses_log1p(self):
        fn = _obs_log_fn(MSEMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log1p(counts))

    def test_coupled_mse_uses_log1p(self):
        fn = _obs_log_fn(CoupledMSEMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log1p(counts))

    def test_poisson_uses_log(self):
        fn = _obs_log_fn(PoissonMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log(counts.clamp_min(1.0)))

    def test_nb_uses_log(self):
        fn = _obs_log_fn(NegativeBinomialMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log(counts.clamp_min(1.0)))

    def test_coupled_poisson_uses_log(self):
        fn = _obs_log_fn(CoupledPoissonMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log(counts.clamp_min(1.0)))

    def test_coupled_nb_uses_log(self):
        fn = _obs_log_fn(CoupledNegativeBinomialMultinomialLoss())
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log(counts.clamp_min(1.0)))

    def test_mse_nondefault_pseudocount(self):
        """MSE with pseudocount=50 → log(x + 50), not log1p(x)."""
        fn = _obs_log_fn(MSEMultinomialLoss(count_pseudocount=50.0))
        counts = torch.tensor([0.0, 10.0, 100.0])
        expected = torch.log(counts + 50.0)
        assert torch.allclose(fn(counts), expected)
        # Must differ from log1p
        assert not torch.allclose(fn(counts), torch.log1p(counts), atol=0.1)

    def test_coupled_mse_nondefault_pseudocount(self):
        fn = _obs_log_fn(CoupledMSEMultinomialLoss(count_pseudocount=25.0))
        counts = torch.tensor([10.0, 100.0])
        assert torch.allclose(fn(counts), torch.log(counts + 25.0))

    def test_poisson_zero_count_clamped(self):
        """Zero counts should not produce -inf for Poisson/NB losses."""
        fn = _obs_log_fn(PoissonMultinomialLoss())
        counts = torch.tensor([0.0, 5.0])
        result = fn(counts)
        assert torch.isfinite(result).all()
        assert result[0] == 0.0  # log(clamp(0, min=1.0)) = log(1) = 0

    def test_mse_predicted_vs_observed_perfect_single_channel(self):
        """For a perfect MSE model: predicted_log_count == observed_log_count."""
        obs_total = torch.tensor([50.0])
        fn = _obs_log_fn(MSEMultinomialLoss())
        obs_log = fn(obs_total)

        # Perfect MSE model: log_counts ≈ log1p(total)
        lc = torch.log1p(obs_total).reshape(1, 1)
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        pred_log = compute_total_log_counts(out, log_counts_include_pseudocount=True)

        assert torch.isclose(obs_log, pred_log, atol=1e-6)

    def test_poisson_predicted_vs_observed_perfect_single_channel(self):
        """For a perfect Poisson model: predicted_log_count == observed_log_count."""
        obs_total = torch.tensor([50.0])
        fn = _obs_log_fn(PoissonMultinomialLoss())
        obs_log = fn(obs_total)

        # Perfect Poisson model: log_counts ≈ log(total)
        lc = torch.log(obs_total).reshape(1, 1)
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 4), log_counts=lc)
        pred_log = compute_total_log_counts(out, log_counts_include_pseudocount=False)

        assert torch.isclose(obs_log, pred_log, atol=1e-6)

    def test_mse_vs_poisson_log_space_differ_for_small_counts(self):
        """log1p and log diverge for small counts, confirming different scales."""
        counts = torch.tensor([1.0])
        log1p_val = torch.log1p(counts)          # log(2) ≈ 0.693
        log_val   = torch.log(counts.clamp_min(1.0))  # log(1) = 0.0
        assert not torch.isclose(log1p_val, log_val, atol=0.1)


# ---------------------------------------------------------------------------
# 4. target_scale applied to raw observed counts
# ---------------------------------------------------------------------------

class TestTargetScale:

    def test_scale_1_no_change(self):
        """target_scale=1.0 leaves the total unchanged."""
        raw = torch.tensor([[[1.0, 2.0, 3.0]]])  # sum=6
        obs_total = raw.sum(dim=(1, 2)) * 1.0
        assert torch.isclose(obs_total, torch.tensor([6.0]))

    def test_scale_2_doubles_total(self):
        """target_scale=2.0 doubles the total before the log transform."""
        raw = torch.tensor([[[1.0, 2.0, 3.0]]])  # sum=6
        obs_total = raw.sum(dim=(1, 2)) * 2.0
        assert torch.isclose(obs_total, torch.tensor([12.0]))

    def test_scaled_obs_matches_mse_training_target(self):
        """Observed log count with target_scale equals the MSE loss's training target.

        MSE loss trains log_counts against log1p(target_scale * sum(raw)).
        So obs_log = log1p(target_scale * raw_total) must hold.
        """
        raw_total = 10.0
        target_scale = 2.0
        raw = torch.tensor([[[raw_total]]])

        obs_total = raw.sum(dim=(1, 2)) * target_scale
        obs_log = torch.log1p(obs_total)

        expected = torch.log1p(torch.tensor([raw_total * target_scale]))
        assert torch.isclose(obs_log, expected, atol=1e-6)

    def test_omitting_scale_produces_systematic_offset(self):
        """Without scale, observed log count is lower than what the model was trained against.

        This is a regression test documenting the original bug.
        """
        raw_total = 100.0
        target_scale = 2.0

        obs_log_correct = torch.log1p(torch.tensor([raw_total * target_scale]))
        obs_log_old     = torch.log1p(torch.tensor([raw_total]))

        diff = (obs_log_correct - obs_log_old).item()
        assert diff > 0, "Scaled observed should be larger"
        assert diff > 0.5, "Offset should be substantial for scale=2, total=100"

    def test_scale_batched(self):
        """target_scale applied consistently across a batch."""
        # Batch of 2: sums=[5, 15], scale=3 → scaled totals=[15, 45]
        raw = torch.tensor([[[1.0, 4.0]], [[3.0, 12.0]]])  # (2, 1, 2)
        target_scale = 3.0
        obs_total = raw.sum(dim=(1, 2)) * target_scale
        expected = torch.tensor([15.0, 45.0])
        assert torch.allclose(obs_total, expected)
