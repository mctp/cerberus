"""Tests for CerberusDataModule data-distribution statistics helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cerberus.config import DataConfig
from cerberus.datamodule import CerberusDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_dataset(data_config: SimpleNamespace, n: int):
    """Mock CerberusDataset for compute_count_quantile_samples tests."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=n)
    mock_dataset.sampler = MagicMock()
    # Return the index back as the "interval" so the mocked extractor can key
    # its returned tensor off it (used by the determinism test).
    mock_dataset.sampler.__getitem__ = MagicMock(side_effect=lambda i: i)
    mock_dataset.data_config = data_config
    return mock_dataset


def _make_dm(target_scale: float, dataset_size: int, data_config: SimpleNamespace):
    """Bare DataModule mock pre-configured for compute_count_quantile_samples."""
    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.data_config = DataConfig.model_construct(target_scale=target_scale)
    dm.train_dataset = _make_mock_dataset(data_config, n=dataset_size)
    return dm


# ---------------------------------------------------------------------------
# compute_count_quantile_samples
# ---------------------------------------------------------------------------


def test_compute_count_quantile_samples_requires_setup():
    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = False
    dm.train_dataset = None
    with pytest.raises(RuntimeError, match="setup"):
        CerberusDataModule.compute_count_quantile_samples(dm)


def test_compute_count_quantile_samples_applies_target_scale():
    # Each extract returns a (2, 100) tensor of 1s -> per-channel length-sum
    # is 100; with target_scale=3.0 every value in the result is 300.
    data_config = SimpleNamespace(
        targets={"a": "a.bw", "b": "b.bw"}, input_len=100, output_len=100,
    )
    dm = _make_dm(target_scale=3.0, dataset_size=5, data_config=data_config)

    # extract() is typed to return torch.Tensor; the mock matches that shape.
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = torch.ones(2, 100)

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        result = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=5, per_channel=True, seed=0,
        )
    assert result.shape == (5, 2)
    assert np.allclose(result, 300.0)


def test_compute_count_quantile_samples_shapes():
    # per_channel=True -> (N, C); per_channel=False -> (N,) with channels summed.
    data_config = SimpleNamespace(
        targets={"a": "a.bw", "b": "b.bw"}, input_len=50, output_len=50,
    )
    dm = _make_dm(target_scale=1.0, dataset_size=10, data_config=data_config)

    # Channel 0 sums to 10, channel 1 sums to 20 per region.
    raw = torch.stack([torch.full((50,), 0.2), torch.full((50,), 0.4)], dim=0)
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = raw

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        per_ch = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=4, per_channel=True, seed=0,
        )
        summed = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=4, per_channel=False, seed=0,
        )

    assert per_ch.shape == (4, 2)
    assert np.allclose(per_ch[:, 0], 10.0)
    assert np.allclose(per_ch[:, 1], 20.0)
    assert summed.shape == (4,)
    assert np.allclose(summed, 30.0)  # channels summed per region: 10 + 20


def test_compute_count_quantile_samples_clamps_n_samples_to_dataset_size():
    # Request more samples than the dataset has — must clamp to dataset size
    # (otherwise random.Random.sample raises ValueError: Sample larger than
    # population).
    data_config = SimpleNamespace(
        targets={"a": "a.bw"}, input_len=10, output_len=10,
    )
    dm = _make_dm(target_scale=1.0, dataset_size=3, data_config=data_config)

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = torch.ones(1, 10)

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        result = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=10, per_channel=True, seed=0,
        )

    assert result.shape == (3, 1)


def test_compute_count_quantile_samples_crops_input_to_output_len():
    # Mock extractor returns a (1, 200) tensor whose 100 centre positions hold
    # a distinguishable value: cropping should sum *only* those positions.
    # input_len=200, output_len=100 -> crop_start=50, crop_end=150.
    raw = torch.cat(
        [
            torch.full((1, 50), 1.0),
            torch.full((1, 100), 7.0),   # the only segment that should be kept
            torch.full((1, 50), 1.0),
        ],
        dim=1,
    )
    assert raw.shape == (1, 200)

    data_config = SimpleNamespace(
        targets={"a": "a.bw"}, input_len=200, output_len=100,
    )
    dm = _make_dm(target_scale=1.0, dataset_size=4, data_config=data_config)

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = raw

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        result = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=4, per_channel=True, seed=0,
        )

    # Cropped centre sum = 100 positions * 7.0 = 700. Uncropped would be
    # 50 + 700 + 50 = 800, so 700 unambiguously asserts the slice fired.
    assert result.shape == (4, 1)
    assert np.allclose(result, 700.0)


def test_compute_count_quantile_samples_seed_determinism():
    # Mock extractor returns a per-interval-keyed tensor so different index
    # samples produce different value sets.  Dataset size > n_samples ensures
    # the seed actually selects a subset.
    data_config = SimpleNamespace(
        targets={"a": "a.bw"}, input_len=10, output_len=10,
    )
    dm = _make_dm(target_scale=1.0, dataset_size=100, data_config=data_config)

    def extract_side_effect(interval: int) -> torch.Tensor:
        # interval is the integer index returned by sampler.__getitem__.
        return torch.full((1, 10), float(interval), dtype=torch.float64)

    mock_extractor = MagicMock()
    mock_extractor.extract.side_effect = extract_side_effect

    with patch("cerberus.datamodule.UniversalExtractor", return_value=mock_extractor):
        run_a1 = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=10, per_channel=True, seed=42,
        )
        run_a2 = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=10, per_channel=True, seed=42,
        )
        run_b = CerberusDataModule.compute_count_quantile_samples(
            dm, n_samples=10, per_channel=True, seed=43,
        )

    # Same seed -> bitwise-identical output.
    assert np.array_equal(run_a1, run_a2)
    # Different seed -> at least one differing element (high-probability given
    # 100 choose 10 ~= 1.7e13; an exact-equality collision is astronomically
    # unlikely).
    assert not np.array_equal(run_a1, run_b)
