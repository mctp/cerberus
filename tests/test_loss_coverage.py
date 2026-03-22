"""Coverage tests for cerberus.loss — untested code paths."""

import pytest
import torch

from cerberus.loss import (
    CoupledMSEMultinomialLoss,
    CoupledNegativeBinomialMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    MSEMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    PoissonMultinomialLoss,
)
from cerberus.output import ProfileCountOutput, ProfileLogRates


@pytest.fixture
def targets_2ch():
    """(B=2, C=2, L=8) positive integer-valued targets (needed for NB distribution)."""
    torch.manual_seed(42)
    return (torch.rand(2, 2, 8) * 10 + 1).floor()


@pytest.fixture
def logits_2ch():
    """(B=2, C=2, L=8) logits."""
    torch.manual_seed(43)
    return torch.randn(2, 2, 8)


@pytest.fixture
def log_counts_per_channel():
    """(B=2, C=2) per-channel log counts."""
    torch.manual_seed(44)
    return torch.randn(2, 2)


# ---------------------------------------------------------------------------
# NegativeBinomialMultinomialLoss — count_per_channel
# ---------------------------------------------------------------------------


class TestNBMultinomialCountPerChannel:
    def test_count_per_channel_true(self, logits_2ch, targets_2ch):
        loss_fn = NegativeBinomialMultinomialLoss(
            total_count=10.0, count_per_channel=True
        )
        # NB needs reasonable log_counts (positive mu) to avoid NaN
        log_counts = torch.tensor([[5.0, 4.0], [4.5, 3.5]])
        output = ProfileCountOutput(logits=logits_2ch, log_counts=log_counts)
        loss = loss_fn(output, targets_2ch)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_count_per_channel_false(self, logits_2ch, targets_2ch):
        loss_fn = NegativeBinomialMultinomialLoss(
            total_count=10.0, count_per_channel=False
        )
        log_counts = torch.tensor([[5.0], [4.5]])
        output = ProfileCountOutput(logits=logits_2ch, log_counts=log_counts)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# CoupledNegativeBinomialMultinomialLoss — count_per_channel and TypeError
# ---------------------------------------------------------------------------


class TestCoupledNBMultinomial:
    def test_count_per_channel_true(self, targets_2ch):
        loss_fn = CoupledNegativeBinomialMultinomialLoss(
            total_count=10.0, count_per_channel=True
        )
        # Use positive log_rates to keep NB stable
        log_rates = torch.ones(2, 2, 8) * 2.0
        output = ProfileLogRates(log_rates=log_rates)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_count_per_channel_false(self, targets_2ch):
        loss_fn = CoupledNegativeBinomialMultinomialLoss(
            total_count=10.0, count_per_channel=False
        )
        log_rates = torch.ones(2, 2, 8) * 2.0
        output = ProfileLogRates(log_rates=log_rates)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_rejects_profile_count_output(
        self, logits_2ch, targets_2ch, log_counts_per_channel
    ):
        loss_fn = CoupledNegativeBinomialMultinomialLoss(total_count=10.0)
        output = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        with pytest.raises(TypeError, match="does not accept ProfileCountOutput"):
            loss_fn(output, targets_2ch)


# ---------------------------------------------------------------------------
# CoupledMSEMultinomialLoss — count_per_channel
# ---------------------------------------------------------------------------


class TestCoupledMSEMultinomial:
    def test_count_per_channel_true(self, logits_2ch, targets_2ch):
        loss_fn = CoupledMSEMultinomialLoss(count_per_channel=True)
        output = ProfileLogRates(log_rates=logits_2ch)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_count_per_channel_false(self, logits_2ch, targets_2ch):
        loss_fn = CoupledMSEMultinomialLoss(count_per_channel=False)
        output = ProfileLogRates(log_rates=logits_2ch)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_rejects_profile_count_output(
        self, logits_2ch, targets_2ch, log_counts_per_channel
    ):
        loss_fn = CoupledMSEMultinomialLoss()
        output = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        with pytest.raises(TypeError, match="does not accept ProfileCountOutput"):
            loss_fn(output, targets_2ch)


# ---------------------------------------------------------------------------
# CoupledPoissonMultinomialLoss — TypeError
# ---------------------------------------------------------------------------


class TestCoupledPoissonMultinomial:
    def test_rejects_profile_count_output(
        self, logits_2ch, targets_2ch, log_counts_per_channel
    ):
        loss_fn = CoupledPoissonMultinomialLoss()
        output = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        with pytest.raises(TypeError, match="does not accept ProfileCountOutput"):
            loss_fn(output, targets_2ch)

    def test_count_per_channel_true(self, logits_2ch, targets_2ch):
        loss_fn = CoupledPoissonMultinomialLoss(count_per_channel=True)
        output = ProfileLogRates(log_rates=logits_2ch)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_count_per_channel_false(self, logits_2ch, targets_2ch):
        loss_fn = CoupledPoissonMultinomialLoss(count_per_channel=False)
        output = ProfileLogRates(log_rates=logits_2ch)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# flatten_channels + average_channels combo
# ---------------------------------------------------------------------------


class TestFlattenAndAverageChannels:
    def test_mse_multinomial_flatten_true_average_true(self, logits_2ch, targets_2ch):
        """flatten_channels=True ignores average_channels."""
        loss_fn = MSEMultinomialLoss(flatten_channels=True, average_channels=True)
        # Global count: log_counts should be (B, 1) or (B,)
        log_counts = torch.randn(2, 1)
        output = ProfileCountOutput(logits=logits_2ch, log_counts=log_counts)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_poisson_multinomial_flatten_true_average_true(
        self, logits_2ch, targets_2ch
    ):
        loss_fn = PoissonMultinomialLoss(flatten_channels=True, average_channels=True)
        log_counts = torch.randn(2, 1)
        output = ProfileCountOutput(logits=logits_2ch, log_counts=log_counts)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# MSEMultinomialLoss._compute_profile_loss with average_channels
# ---------------------------------------------------------------------------


class TestMSEProfileLossAverageChannels:
    def test_average_channels_true_gives_different_result(
        self, logits_2ch, targets_2ch
    ):
        """average_channels=True vs False should give different magnitudes."""
        loss_avg = MSEMultinomialLoss(average_channels=True)
        loss_sum = MSEMultinomialLoss(average_channels=False)

        # Compute profile losses directly
        pl_avg = loss_avg._compute_profile_loss(logits_2ch, targets_2ch)
        pl_sum = loss_sum._compute_profile_loss(logits_2ch, targets_2ch)

        assert torch.isfinite(pl_avg)
        assert torch.isfinite(pl_sum)
        # With 2 channels, sum should generally be larger than mean
        # (unless values are negative, but multinomial NLL is non-negative)
        # Just verify they differ
        assert not torch.allclose(pl_avg, pl_sum)


# ---------------------------------------------------------------------------
# MSEMultinomialLoss count_per_channel=True
# ---------------------------------------------------------------------------


class TestMSEMultinomialCountPerChannel:
    def test_count_per_channel(self, logits_2ch, targets_2ch, log_counts_per_channel):
        loss_fn = MSEMultinomialLoss(count_per_channel=True)
        output = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_count_per_channel_loss_differs_from_global(
        self, logits_2ch, targets_2ch, log_counts_per_channel
    ):
        loss_per = MSEMultinomialLoss(count_per_channel=True)
        loss_global = MSEMultinomialLoss(count_per_channel=False)

        output_per = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        output_global = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel[:, :1]
        )

        l_per = loss_per(output_per, targets_2ch)
        l_global = loss_global(output_global, targets_2ch)
        # They should be different values (different count computation)
        assert not torch.allclose(l_per, l_global)


# ---------------------------------------------------------------------------
# PoissonMultinomialLoss count_per_channel
# ---------------------------------------------------------------------------


class TestPoissonMultinomialCountPerChannel:
    def test_count_per_channel_true(
        self, logits_2ch, targets_2ch, log_counts_per_channel
    ):
        loss_fn = PoissonMultinomialLoss(count_per_channel=True)
        output = ProfileCountOutput(
            logits=logits_2ch, log_counts=log_counts_per_channel
        )
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)

    def test_average_channels_false(self, logits_2ch, targets_2ch):
        loss_fn = PoissonMultinomialLoss(average_channels=False)
        log_counts = torch.randn(2, 1)
        output = ProfileCountOutput(logits=logits_2ch, log_counts=log_counts)
        loss = loss_fn(output, targets_2ch)
        assert torch.isfinite(loss)
