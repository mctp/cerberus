"""Coverage tests for cerberus.metrics — untested code paths."""

import warnings

import pytest
import torch

from cerberus.metrics import (
    LogCountsMeanSquaredError,
    LogCountsPearsonCorrCoef,
    ProfileMeanSquaredError,
    ProfilePearsonCorrCoef,
)
from cerberus.output import ProfileCountOutput, ProfileLogRates


@pytest.fixture
def targets_2ch():
    """(B=4, C=2, L=8) positive targets."""
    torch.manual_seed(50)
    return torch.rand(4, 2, 8) * 10 + 1


@pytest.fixture
def log_rates_2ch():
    """(B=4, C=2, L=8) log rates."""
    torch.manual_seed(51)
    return torch.randn(4, 2, 8)


# ---------------------------------------------------------------------------
# LogCountsMeanSquaredError with count_per_channel=True
# ---------------------------------------------------------------------------


class TestLogCountsMSEPerChannel:
    def test_count_per_channel_profile_count_output(self, targets_2ch):
        metric = LogCountsMeanSquaredError(count_per_channel=True)
        log_counts = torch.randn(4, 2)
        output = ProfileCountOutput(logits=torch.randn(4, 2, 8), log_counts=log_counts)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert torch.isfinite(result)

    def test_count_per_channel_profile_log_rates(self, targets_2ch, log_rates_2ch):
        metric = LogCountsMeanSquaredError(count_per_channel=True)
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# LogCountsPearsonCorrCoef with count_per_channel=True
# ---------------------------------------------------------------------------


class TestLogCountsPearsonPerChannel:
    def test_count_per_channel_profile_count_output(self, targets_2ch):
        metric = LogCountsPearsonCorrCoef(count_per_channel=True)
        log_counts = torch.randn(4, 2)
        output = ProfileCountOutput(logits=torch.randn(4, 2, 8), log_counts=log_counts)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert result.ndim == 0

    def test_count_per_channel_profile_log_rates(self, targets_2ch, log_rates_2ch):
        metric = LogCountsPearsonCorrCoef(count_per_channel=True)
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert result.ndim == 0

    def test_empty_state_returns_nan(self):
        metric = LogCountsPearsonCorrCoef()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = metric.compute()
        assert torch.isnan(result)

    def test_constant_predictions_returns_nan(self, targets_2ch):
        """When all predictions are the same, denom < eps -> NaN."""
        metric = LogCountsPearsonCorrCoef()
        # Constant log_rates -> constant logsumexp
        log_rates = torch.ones(4, 1, 8) * 5.0
        output = ProfileLogRates(log_rates=log_rates)
        metric.update(output, targets_2ch[:, :1, :])
        result = metric.compute()
        assert torch.isnan(result)

    def test_single_element_returns_nan(self):
        """Less than 2 elements -> NaN."""
        metric = LogCountsPearsonCorrCoef()
        log_rates = torch.randn(1, 1, 8)
        target = torch.rand(1, 1, 8)
        output = ProfileLogRates(log_rates=log_rates)
        metric.update(output, target)
        result = metric.compute()
        assert torch.isnan(result)


# ---------------------------------------------------------------------------
# Metrics with ProfileLogRates inputs (not just ProfileLogits)
# ---------------------------------------------------------------------------


class TestMetricsWithProfileLogRates:
    def test_profile_pearson_with_log_rates(self, targets_2ch, log_rates_2ch):
        metric = ProfilePearsonCorrCoef()
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert torch.isfinite(result) or torch.isnan(result)

    def test_profile_mse_with_log_rates(self, targets_2ch, log_rates_2ch):
        metric = ProfileMeanSquaredError()
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert torch.isfinite(result)

    def test_log_counts_mse_with_log_rates(self, targets_2ch, log_rates_2ch):
        metric = LogCountsMeanSquaredError()
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert torch.isfinite(result)

    def test_log_counts_pearson_with_log_rates(self, targets_2ch, log_rates_2ch):
        metric = LogCountsPearsonCorrCoef()
        output = ProfileLogRates(log_rates=log_rates_2ch)
        metric.update(output, targets_2ch)
        result = metric.compute()
        assert result.ndim == 0
