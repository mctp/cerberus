"""
Tests that metric compute() methods return tensors on the correct device.

Regression tests for the DDP bug where compute() returned torch.tensor(float("nan"))
without specifying device=, causing RuntimeError: No backend type associated with
device type cpu when NCCL tried to all_reduce a CPU tensor during DDP sync.
"""

import pytest
import torch

from cerberus.metrics import (
    CountProfilePearsonCorrCoef,
    LogCountsPearsonCorrCoef,
    ProfilePearsonCorrCoef,
)
from cerberus.output import ProfileCountOutput, ProfileLogRates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile_count_output(batch=2, channels=1, length=10, device="cpu"):
    logits = torch.randn(batch, channels, length, device=device)
    log_counts = torch.log1p(torch.rand(batch, channels, device=device) * 10 + 1)
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


def _make_target(batch=2, channels=1, length=10, device="cpu"):
    return torch.rand(batch, channels, length, device=device) * 10


def _make_log_rates_output(batch=2, channels=1, length=10, device="cpu"):
    log_rates = torch.randn(batch, channels, length, device=device)
    return ProfileLogRates(log_rates=log_rates)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# ProfilePearsonCorrCoef
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:The ``compute`` method of metric.*was called before the ``update`` method"
)
@pytest.mark.parametrize("device", DEVICES)
def test_per_example_profile_pearson_empty_compute_device(device):
    """compute() on a fresh (empty) metric returns nan on the correct device."""
    metric = ProfilePearsonCorrCoef().to(device)
    result = metric.compute()
    assert result.device.type == device
    assert torch.isnan(result)


@pytest.mark.parametrize("device", DEVICES)
def test_per_example_profile_pearson_normal_compute_device(device):
    """compute() after updates returns a value on the correct device."""
    metric = ProfilePearsonCorrCoef().to(device)
    preds = _make_log_rates_output(device=device)
    target = _make_target(device=device)
    metric.update(preds, target)
    result = metric.compute()
    assert result.device.type == device


# ---------------------------------------------------------------------------
# CountProfilePearsonCorrCoef
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:The ``compute`` method of metric.*was called before the ``update`` method"
)
@pytest.mark.parametrize("device", DEVICES)
def test_per_example_count_profile_pearson_empty_compute_device(device):
    """compute() on a fresh (empty) metric returns nan on the correct device."""
    metric = CountProfilePearsonCorrCoef().to(device)
    result = metric.compute()
    assert result.device.type == device
    assert torch.isnan(result)


@pytest.mark.parametrize("device", DEVICES)
def test_per_example_count_profile_pearson_normal_compute_device(device):
    """compute() after updates returns a value on the correct device."""
    metric = CountProfilePearsonCorrCoef().to(device)
    preds = _make_profile_count_output(device=device)
    target = _make_target(device=device)
    metric.update(preds, target)
    result = metric.compute()
    assert result.device.type == device


# ---------------------------------------------------------------------------
# LogCountsPearsonCorrCoef
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:The ``compute`` method of metric.*was called before the ``update`` method"
)
@pytest.mark.parametrize("device", DEVICES)
def test_per_example_log_counts_pearson_empty_compute_device(device):
    """compute() on a fresh (empty) metric returns nan on the correct device."""
    metric = LogCountsPearsonCorrCoef().to(device)
    result = metric.compute()
    assert result.device.type == device
    assert torch.isnan(result)


@pytest.mark.parametrize("device", DEVICES)
def test_per_example_log_counts_pearson_single_example_device(device):
    """compute() with only one example (numel < 2) returns nan on the correct device."""
    metric = LogCountsPearsonCorrCoef().to(device)
    preds = _make_profile_count_output(batch=1, device=device)
    target = _make_target(batch=1, device=device)
    metric.update(preds, target)
    result = metric.compute()
    assert result.device.type == device
    # numel() == 1 < 2, so result should be nan
    assert torch.isnan(result)


@pytest.mark.parametrize("device", DEVICES)
def test_per_example_log_counts_pearson_constant_preds_device(device):
    """compute() when denom < 1e-8 (constant preds) returns nan on the correct device."""
    metric = LogCountsPearsonCorrCoef().to(device)
    # All examples have identical log_counts -> zero variance -> denom == 0
    log_counts = torch.log1p(torch.tensor([[5.0], [5.0]], device=device))
    logits = torch.zeros(2, 1, 10, device=device)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    target = _make_target(batch=2, device=device)
    metric.update(preds, target)
    result = metric.compute()
    assert result.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_per_example_log_counts_pearson_normal_compute_device(device):
    """compute() with varying log counts returns a value on the correct device."""
    metric = LogCountsPearsonCorrCoef().to(device)
    # Use different log_counts per example to ensure non-zero variance
    log_counts = torch.tensor([[1.0], [3.0]], device=device)
    logits = torch.zeros(2, 1, 10, device=device)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    target = _make_target(batch=2, device=device)
    metric.update(preds, target)
    result = metric.compute()
    assert result.device.type == device


# ---------------------------------------------------------------------------
# BPNetMetricCollection smoke test
# ---------------------------------------------------------------------------


def test_bpnet_metric_collection_compute_device():
    """BPNetMetricCollection.compute() returns all metrics on CPU without error."""
    from cerberus.models.bpnet import BPNetMetricCollection

    metrics = BPNetMetricCollection()
    preds = _make_profile_count_output(batch=4)
    target = _make_target(batch=4)
    metrics.update(preds, target)
    results = metrics.compute()
    for name, val in results.items():
        assert val.device.type == "cpu", f"{name} not on cpu: {val.device}"
