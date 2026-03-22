"""
Tests for adaptive count loss weight computation.

Covers:
  - compute_counts_loss_weight()
  - CerberusDataModule.compute_median_counts()
  - resolve_adaptive_loss_args()
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from cerberus.config import DataConfig, ModelConfig
from cerberus.train import compute_counts_loss_weight, resolve_adaptive_loss_args

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
    return ModelConfig(
        name="Test",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.models.bpnet.BPNetLoss",
        loss_args=loss_args,
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args={},
        pretrained=[],
    )


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
    assert result.loss_args["alpha"] == pytest.approx(50.0)


def test_resolve_adaptive_count_weight_key():
    config = _model_config({"count_weight": "adaptive"})
    dm = _mock_datamodule(300.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result.loss_args["count_weight"] == pytest.approx(30.0)


def test_resolve_adaptive_preserves_other_keys():
    config = _model_config({"alpha": "adaptive", "beta": 1.0})
    dm = _mock_datamodule(200.0)
    result = resolve_adaptive_loss_args(config, dm)
    assert result.loss_args["alpha"] == pytest.approx(20.0)
    assert result.loss_args["beta"] == pytest.approx(1.0)


def test_resolve_adaptive_does_not_mutate_input():
    config = _model_config({"alpha": "adaptive"})
    dm = _mock_datamodule(400.0)
    resolve_adaptive_loss_args(config, dm)
    # Original config must be unchanged so train_multi can reuse it across folds
    assert config.loss_args["alpha"] == "adaptive"


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
    assert result.loss_args["alpha"] == result.loss_args["count_weight"]


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


def _make_mock_dataset(data_config: SimpleNamespace, n: int = 5):
    """Helper: mock CerberusDataset for compute_median_counts tests."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=n)
    mock_dataset.sampler = MagicMock()
    mock_dataset.sampler.__getitem__ = MagicMock(return_value="interval")
    mock_dataset.data_config = data_config
    # target_signal_extractor starts with no open handles (mimics reality)
    mock_dataset.target_signal_extractor = MagicMock()
    mock_dataset.target_signal_extractor._bigwig_files = None
    return mock_dataset


def test_compute_median_counts_applies_target_scale():
    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = DataConfig.model_construct(target_scale=2.0)

    data_config = SimpleNamespace(
        targets={"sig": "sig.bw"},
        input_len=100,
        output_len=100,
    )
    mock_dataset = _make_mock_dataset(data_config, n=5)
    dm.train_dataset = mock_dataset

    # Patch UniversalExtractor so no real files are opened
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = torch.ones(1, 100)  # sum = 100

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        with patch("random.sample", return_value=list(range(5))):
            result = CerberusDataModule.compute_median_counts(dm)

    # raw_median = 100, target_scale = 2.0 → scaled_median = 200
    assert result == pytest.approx(200.0)


def test_compute_median_counts_uses_median_not_mean():
    import numpy as np

    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = DataConfig.model_construct(target_scale=1.0)

    raw_counts = [10.0, 20.0, 1000.0]  # mean=343, median=20

    data_config = SimpleNamespace(
        targets={"sig": "sig.bw"},
        input_len=100,
        output_len=100,
    )
    mock_dataset = _make_mock_dataset(data_config, n=3)
    dm.train_dataset = mock_dataset

    mock_extractor = MagicMock()
    side_effects = [torch.tensor([[c]]) for c in raw_counts]
    mock_extractor.extract.side_effect = side_effects

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        with patch("random.sample", return_value=[0, 1, 2]):
            result = CerberusDataModule.compute_median_counts(dm)

    assert result == pytest.approx(float(np.median(raw_counts)))


def test_compute_median_counts_does_not_open_dataset_extractor():
    """Fork-safety: compute_median_counts must not open handles on the dataset's
    own target_signal_extractor.  If it did, forked DataLoader workers would
    inherit the shared file descriptor and produce BadData panics."""
    from pathlib import Path

    from cerberus.datamodule import CerberusDataModule
    from cerberus.signal import SignalExtractor

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = DataConfig.model_construct(target_scale=1.0)

    data_config = SimpleNamespace(
        targets={"sig": "sig.bw"},
        input_len=100,
        output_len=100,
    )
    mock_dataset = _make_mock_dataset(data_config, n=3)
    # Give the dataset a real SignalExtractor with no handles open
    real_extractor = SignalExtractor({"sig": Path("sig.bw")})
    mock_dataset.target_signal_extractor = real_extractor
    dm.train_dataset = mock_dataset

    mock_tmp = MagicMock()
    mock_tmp.extract.return_value = torch.ones(1, 100)

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_tmp):
        with patch("random.sample", return_value=[0, 1, 2]):
            CerberusDataModule.compute_median_counts(dm)

    # The dataset's own extractor must never have been loaded
    assert real_extractor._bigwig_files is None
