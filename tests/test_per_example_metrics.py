import torch
import torch.nn.functional as F
import pytest
from cerberus.metrics import (
    _per_example_pearson,
    PerExampleProfilePearsonCorrCoef,
    PerExampleCountProfilePearsonCorrCoef,
    PerExampleLogCountsPearsonCorrCoef,
)
from cerberus.output import ProfileCountOutput, ProfileLogRates, ProfileLogits


# ---------------------------------------------------------------------------
# _per_example_pearson
# ---------------------------------------------------------------------------

class TestPerExamplePearson:
    def test_perfect_correlation(self):
        """Identical tensors should give correlation = 1.0."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
        corr = _per_example_pearson(x, x)
        assert corr.shape == (1, 1)
        assert torch.isclose(corr[0, 0], torch.tensor(1.0), atol=1e-6)

    def test_negative_correlation(self):
        """Reversed tensor should give correlation = -1.0."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        y = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
        corr = _per_example_pearson(x, y)
        assert torch.isclose(corr[0, 0], torch.tensor(-1.0), atol=1e-6)

    def test_zero_variance_returns_nan(self):
        """Constant input (zero variance) should return NaN."""
        x = torch.tensor([[[5.0, 5.0, 5.0, 5.0]]])
        y = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        corr = _per_example_pearson(x, y)
        assert torch.isnan(corr[0, 0])

    def test_batch_and_channels(self):
        """Verify correct shape and independent computation across B and C."""
        B, C, L = 3, 2, 8
        x = torch.randn(B, C, L)
        y = torch.randn(B, C, L)
        corr = _per_example_pearson(x, y)
        assert corr.shape == (B, C)

    def test_scale_invariance(self):
        """Pearson correlation should be invariant to scale."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        y = x * 100.0 + 42.0
        corr = _per_example_pearson(x, y)
        assert torch.isclose(corr[0, 0], torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# PerExampleProfilePearsonCorrCoef
# ---------------------------------------------------------------------------

class TestPerExampleProfilePearsonCorrCoef:
    def test_perfect_profile_correlation(self):
        """If softmax(logits) matches target profile shape, correlation ~ 1."""
        L = 16
        logits = torch.arange(float(L)).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        probs = F.softmax(logits, dim=-1)
        target = probs.clone()  # exact match

        metric = PerExampleProfilePearsonCorrCoef(num_channels=1)
        metric.update(ProfileLogRates(log_rates=logits), target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)

    def test_accepts_profile_logits(self):
        """Should accept ProfileLogits as well as ProfileLogRates."""
        L = 16
        logits = torch.randn(2, 1, L)
        target = F.softmax(logits, dim=-1)

        metric = PerExampleProfilePearsonCorrCoef(num_channels=1)
        metric.update(ProfileLogits(logits=logits), target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)

    def test_rejects_invalid_type(self):
        metric = PerExampleProfilePearsonCorrCoef(num_channels=1)
        with pytest.raises(TypeError):
            metric.update(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))  # type: ignore

    def test_multi_channel(self):
        """Multi-channel inputs should work and average across channels."""
        B, C, L = 4, 3, 32
        logits = torch.randn(B, C, L)
        probs = F.softmax(logits, dim=-1)

        metric = PerExampleProfilePearsonCorrCoef(num_channels=C)
        metric.update(ProfileLogRates(log_rates=logits), probs)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-4)

    def test_implicit_log_targets(self):
        """With implicit_log_targets=True, targets should be expm1'd."""
        L = 16
        logits = torch.randn(2, 1, L)
        raw_target = F.softmax(logits, dim=-1) * 10
        log_target = torch.log1p(raw_target)

        metric = PerExampleProfilePearsonCorrCoef(num_channels=1, implicit_log_targets=True)
        metric.update(ProfileLogRates(log_rates=logits), log_target)
        val = metric.compute()
        # Pearson is scale-invariant, so softmax vs softmax*10 gives same result
        assert not torch.isnan(val)

    def test_multi_batch_accumulation(self):
        """Accumulating across multiple update() calls should work."""
        L = 16
        metric = PerExampleProfilePearsonCorrCoef(num_channels=1)
        for _ in range(5):
            logits = torch.randn(4, 1, L)
            target = F.softmax(logits, dim=-1)
            metric.update(ProfileLogRates(log_rates=logits), target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-4)

    def test_zero_variance_examples_skipped(self):
        """Examples with zero-variance predictions should be skipped, not crash."""
        L = 16
        logits = torch.zeros(2, 1, L)  # uniform softmax → near-zero variance after mean subtraction
        # but softmax of zeros is 1/L everywhere, which IS zero variance
        target = torch.randn(2, 1, L).abs()

        metric = PerExampleProfilePearsonCorrCoef(num_channels=1)
        metric.update(ProfileLogRates(log_rates=logits), target)
        val = metric.compute()
        # Should be NaN (all examples skipped) — but should NOT crash
        assert torch.isnan(val)


# ---------------------------------------------------------------------------
# PerExampleCountProfilePearsonCorrCoef
# ---------------------------------------------------------------------------

class TestPerExampleCountProfilePearsonCorrCoef:
    def test_perfect_count_correlation(self):
        """If reconstructed counts match targets, correlation ~ 1."""
        L = 16
        target = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        total = target.sum()
        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log1p(total).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = PerExampleCountProfilePearsonCorrCoef(num_channels=1)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-4)

    def test_rejects_invalid_type(self):
        metric = PerExampleCountProfilePearsonCorrCoef(num_channels=1)
        with pytest.raises(TypeError):
            metric.update(
                ProfileLogRates(log_rates=torch.zeros(1, 1, 4)),  # type: ignore[arg-type]
                torch.zeros(1, 1, 4),
            )

    def test_zero_counts_skipped(self):
        """When log_counts → expm1 ≤ 0 (clamped to 0), predictions are all zero.
        These zero-variance examples should be skipped gracefully."""
        L = 16
        logits = torch.randn(2, 1, L)
        log_counts = torch.tensor([[-1.0], [-1.0]])  # expm1(-1) < 0, clamped to 0
        target = torch.randn(2, 1, L).abs()

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = PerExampleCountProfilePearsonCorrCoef(num_channels=1)
        metric.update(preds, target)
        val = metric.compute()
        # All examples have zero preds → NaN, all skipped
        assert torch.isnan(val)

    def test_multi_channel(self):
        B, C, L = 4, 2, 32
        target = torch.randn(B, C, L).abs() + 0.1
        total = target.sum(dim=-1, keepdim=True)
        probs = target / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log1p(total.squeeze(-1))  # (B, C)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = PerExampleCountProfilePearsonCorrCoef(num_channels=C)
        metric.update(preds, target)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)

    def test_implicit_log_targets(self):
        L = 16
        target_raw = torch.arange(1, float(L + 1)).unsqueeze(0).unsqueeze(0)
        target_log = torch.log1p(target_raw)
        total = target_raw.sum()
        probs = target_raw / total
        logits = torch.log(probs + 1e-10)
        log_counts = torch.log1p(total).reshape(1, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = PerExampleCountProfilePearsonCorrCoef(num_channels=1, implicit_log_targets=True)
        metric.update(preds, target_log)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# PerExampleLogCountsPearsonCorrCoef
# ---------------------------------------------------------------------------

class TestPerExampleLogCountsPearsonCorrCoef:
    def _make_batch(self, counts, L=10):
        """Helper to create ProfileCountOutput from a list of total counts."""
        B = len(counts)
        counts_t = torch.tensor(counts, dtype=torch.float32)
        log_counts = torch.log1p(counts_t).unsqueeze(1)  # (B, 1)
        logits = torch.zeros(B, 1, L)
        targets = torch.zeros(B, 1, L)
        for i, c in enumerate(counts):
            targets[i, 0, 0] = c  # put all counts in first position
        return ProfileCountOutput(logits=logits, log_counts=log_counts), targets

    def test_perfect_correlation(self):
        """When pred log_counts == log1p(target_sum), correlation should be 1."""
        preds, targets = self._make_batch([10, 20, 30, 40, 50])
        metric = PerExampleLogCountsPearsonCorrCoef()
        metric.update(preds, targets)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)

    def test_multi_batch_accumulation(self):
        """Correlation computed correctly across multiple update() calls."""
        metric = PerExampleLogCountsPearsonCorrCoef()
        preds1, targets1 = self._make_batch([10, 20, 30])
        preds2, targets2 = self._make_batch([40, 50, 60])
        metric.update(preds1, targets1)
        metric.update(preds2, targets2)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)

    def test_too_few_samples(self):
        """With < 2 samples, should return NaN."""
        preds, targets = self._make_batch([10])
        metric = PerExampleLogCountsPearsonCorrCoef()
        metric.update(preds, targets)
        val = metric.compute()
        assert torch.isnan(val)

    def test_constant_counts_nan(self):
        """Zero variance in counts → NaN."""
        preds, targets = self._make_batch([10, 10, 10, 10])
        metric = PerExampleLogCountsPearsonCorrCoef()
        metric.update(preds, targets)
        val = metric.compute()
        assert torch.isnan(val)

    def test_from_log_rates(self):
        """Should also accept ProfileLogRates input."""
        B, L = 5, 10
        counts = [10.0, 20.0, 30.0, 40.0, 50.0]
        targets = torch.zeros(B, 1, L)
        for i, c in enumerate(counts):
            targets[i, 0, 0] = c

        # log_rates where sum of rates = total count
        log_rates = torch.zeros(B, 1, L)
        for i, c in enumerate(counts):
            log_rates[i, 0, :] = torch.log(torch.tensor(c / L))

        preds = ProfileLogRates(log_rates=log_rates)
        metric = PerExampleLogCountsPearsonCorrCoef()
        metric.update(preds, targets)
        val = metric.compute()
        assert not torch.isnan(val)
        assert val > 0.9  # should be high correlation

    def test_rejects_invalid_type(self):
        metric = PerExampleLogCountsPearsonCorrCoef()
        with pytest.raises(TypeError):
            metric.update(
                ProfileLogits(logits=torch.zeros(1, 1, 4)),  # type: ignore[arg-type]
                torch.zeros(1, 1, 4),
            )

    def test_implicit_log_targets(self):
        """With implicit_log_targets, targets are expm1'd before summing."""
        counts = [10.0, 20.0, 30.0, 40.0, 50.0]
        B, L = len(counts), 10

        targets_raw = torch.zeros(B, 1, L)
        for i, c in enumerate(counts):
            targets_raw[i, 0, 0] = c
        targets_log = torch.log1p(targets_raw)

        counts_t = torch.tensor(counts, dtype=torch.float32)
        log_counts = torch.log1p(counts_t).unsqueeze(1)
        logits = torch.zeros(B, 1, L)
        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)

        metric = PerExampleLogCountsPearsonCorrCoef(implicit_log_targets=True)
        metric.update(preds, targets_log)
        val = metric.compute()
        assert torch.isclose(val, torch.tensor(1.0), atol=1e-4)

    def test_reset_clears_state(self):
        """After reset(), metric should start fresh."""
        import warnings
        preds, targets = self._make_batch([10, 20, 30, 40, 50])
        metric = PerExampleLogCountsPearsonCorrCoef()
        metric.update(preds, targets)
        metric.reset()
        # After reset with no updates, compute() before update() is expected here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            val = metric.compute()
        assert torch.isnan(val)
