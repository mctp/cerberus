"""
Tests for adaptive count loss weight computation.

Covers:
  - compute_counts_loss_weight()
  - CerberusDataModule.compute_median_counts()
  - resolve_adaptive_loss_args()
"""

import pytest
import torch
from typing import cast
from unittest.mock import MagicMock, patch

from cerberus.train import compute_counts_loss_weight, resolve_adaptive_loss_args
from cerberus.config import ModelConfig


# ---------------------------------------------------------------------------
# compute_counts_loss_weight
# ---------------------------------------------------------------------------

def test_compute_counts_loss_weight_basic():
    assert compute_counts_loss_weight(500.0) == pytest.approx(50.0)


def test_compute_counts_loss_weight_custom_scale():
    assert compute_counts_loss_weight(500.0, scale=5.0) == pytest.approx(100.0)


def test_compute_counts_loss_weight_linear_in_depth():
    # Doubling median_counts should double the weight (linear relationship)
    w1 = compute_counts_loss_weight(200.0)
    w2 = compute_counts_loss_weight(400.0)
    assert w2 == pytest.approx(2 * w1)


def test_compute_counts_loss_weight_zero_raises():
    with pytest.raises(ValueError, match="median_counts must be positive"):
        compute_counts_loss_weight(0.0)


def test_compute_counts_loss_weight_negative_raises():
    with pytest.raises(ValueError, match="median_counts must be positive"):
        compute_counts_loss_weight(-10.0)


# ---------------------------------------------------------------------------
# resolve_adaptive_loss_args
# ---------------------------------------------------------------------------

def _model_config(loss_args: dict) -> ModelConfig:
    return cast(ModelConfig, {
        "name": "Test",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.models.bpnet.BPNetLoss",
        "loss_args": loss_args,
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
        "metrics_args": {},
        "model_args": {},
    })


def _mock_datamodule(median_counts: float) -> MagicMock:
    dm = MagicMock()
    dm.compute_median_counts.return_value = median_counts
    return dm


def test_resolve_adaptive_no_sentinel_returns_same():
    config = _model_config({"alpha": 1.0})
    dm = _mock_datamodule(500.0)
    result = resolve_adaptive_loss_args(config, dm)
    # No "adaptive" sentinel — should return the same object unchanged
    assert result is config
    dm.compute_median_counts.assert_not_called()


def test_resolve_adaptive_alpha_key():
    config = _model_config({"alpha": "adaptive"})
    dm = _mock_datamodule(500.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result["loss_args"]["alpha"] == pytest.approx(50.0)


def test_resolve_adaptive_count_weight_key():
    config = _model_config({"count_weight": "adaptive"})
    dm = _mock_datamodule(300.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result["loss_args"]["count_weight"] == pytest.approx(30.0)


def test_resolve_adaptive_preserves_other_keys():
    config = _model_config({"alpha": "adaptive", "beta": 1.0})
    dm = _mock_datamodule(200.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result["loss_args"]["alpha"] == pytest.approx(20.0)
    assert result["loss_args"]["beta"] == pytest.approx(1.0)


def test_resolve_adaptive_does_not_mutate_input():
    config = _model_config({"alpha": "adaptive"})
    dm = _mock_datamodule(400.0)
    resolve_adaptive_loss_args(config, dm)
    # Original config must be unchanged so train_multi can reuse it across folds
    assert config["loss_args"]["alpha"] == "adaptive"


def test_resolve_adaptive_calls_compute_once():
    config = _model_config({"alpha": "adaptive", "count_weight": "adaptive"})
    dm = _mock_datamodule(600.0)
    resolve_adaptive_loss_args(config, dm)
    # compute_median_counts should be called exactly once regardless of how many
    # "adaptive" keys exist
    dm.compute_median_counts.assert_called_once()


def test_resolve_adaptive_multiple_keys_same_weight():
    # Both adaptive keys should get the same computed weight
    config = _model_config({"alpha": "adaptive", "count_weight": "adaptive"})
    dm = _mock_datamodule(500.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result["loss_args"]["alpha"] == result["loss_args"]["count_weight"]


# ---------------------------------------------------------------------------
# CerberusDataModule.compute_median_counts
# ---------------------------------------------------------------------------

def test_compute_median_counts_requires_setup():
    from cerberus.datamodule import CerberusDataModule
    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = False
    dm.train_dataset = None
    # Call the real method on a mock that mimics an un-setup datamodule
    with pytest.raises(RuntimeError, match="setup"):
        CerberusDataModule.compute_median_counts(dm)


def test_compute_median_counts_applies_target_scale():
    from cerberus.datamodule import CerberusDataModule

    # Build a mock datamodule that looks initialized
    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = {"target_scale": 2.0}

    # Mock training dataset: 5 intervals each with a tensor summing to 100
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=5)
    mock_dataset.sampler = MagicMock()
    mock_dataset.sampler.__getitem__ = MagicMock(return_value="interval")
    mock_dataset.get_raw_targets = MagicMock(
        return_value=torch.ones(1, 100)  # sum = 100 per interval
    )
    dm.train_dataset = mock_dataset

    with patch("random.sample", return_value=list(range(5))):
        result = CerberusDataModule.compute_median_counts(dm)

    # raw_median = 100, target_scale = 2.0 → scaled_median = 200
    assert result == pytest.approx(200.0)


def test_compute_median_counts_uses_median_not_mean():
    from cerberus.datamodule import CerberusDataModule
    import numpy as np

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = {"target_scale": 1.0}

    raw_counts = [10.0, 20.0, 1000.0]  # mean=343, median=20

    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=3)
    mock_dataset.sampler = MagicMock()
    mock_dataset.sampler.__getitem__ = MagicMock(return_value="interval")

    side_effects = [torch.tensor([[c]]) for c in raw_counts]
    mock_dataset.get_raw_targets = MagicMock(side_effect=side_effects)
    dm.train_dataset = mock_dataset

    with patch("random.sample", return_value=[0, 1, 2]):
        result = CerberusDataModule.compute_median_counts(dm)

    assert result == pytest.approx(float(np.median(raw_counts)))
