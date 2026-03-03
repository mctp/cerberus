"""
Tests for prepare_data() complexity metrics caching.

This test file grows incrementally across commits:
- Commit 1: Sampler + dataset plumbing (prepare_cache passthrough)
- Commit 2: Cache dir resolution and loading helpers
- Commit 3: prepare_data() + setup() integration
"""

import numpy as np
import pytest
from typing import cast
from unittest.mock import MagicMock, patch

from cerberus.interval import Interval
from cerberus.samplers import (
    ComplexityMatchedSampler,
    IntervalSampler,
    ListSampler,
    PeakSampler,
    RandomSampler,
    create_sampler,
)
from cerberus.config import SamplerConfig
from interlap import InterLap


# --- Commit 1: Sampler + dataset plumbing tests ---


@pytest.fixture
def mock_compute():
    """Mock compute_intervals_complexity to avoid real FASTA reads."""
    with patch("cerberus.samplers.compute_intervals_complexity") as m:
        def side_effect(intervals, fasta, metrics):
            return np.random.rand(len(intervals), len(metrics)).astype(np.float32)
        m.side_effect = side_effect
        yield m


class TestCreateSamplerPrepareCache:
    """Tests that create_sampler() passes prepare_cache through to ComplexityMatchedSampler."""

    def test_complexity_matched_receives_cache(self, mock_compute):
        """create_sampler() with complexity_matched type passes prepare_cache as metrics_cache."""
        cache: dict[str, np.ndarray] = {
            "chr1:0-100(+)": np.array([0.5], dtype=np.float32),
        }

        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "complexity_matched",
            "padded_size": 100,
            "sampler_args": {
                "target_sampler": {
                    "type": "random",
                    "args": {"num_intervals": 5},
                },
                "candidate_sampler": {
                    "type": "random",
                    "args": {"num_intervals": 10},
                },
                "bins": 10,
                "candidate_ratio": 1.0,
                "metrics": ["gc"],
            },
        })
        chrom_sizes = {"chr1": 10000}

        sampler = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=42,
            prepare_cache=cache,
        )

        assert isinstance(sampler, ComplexityMatchedSampler)
        assert sampler.metrics_cache is cache

    def test_complexity_matched_without_cache(self, mock_compute):
        """create_sampler() without prepare_cache lets sampler create its own internal dict."""
        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "complexity_matched",
            "padded_size": 100,
            "sampler_args": {
                "target_sampler": {
                    "type": "random",
                    "args": {"num_intervals": 5},
                },
                "candidate_sampler": {
                    "type": "random",
                    "args": {"num_intervals": 10},
                },
                "bins": 10,
                "candidate_ratio": 1.0,
                "metrics": ["gc"],
            },
        })
        chrom_sizes = {"chr1": 10000}

        sampler = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=42,
        )

        assert isinstance(sampler, ComplexityMatchedSampler)
        assert isinstance(sampler.metrics_cache, dict)


class TestPeakSamplerPrepareCache:
    """Tests that PeakSampler passes prepare_cache to its internal ComplexityMatchedSampler."""

    @pytest.fixture
    def mock_peak_deps(self):
        """Mock dependencies for PeakSampler to avoid real I/O."""
        with patch("cerberus.samplers.IntervalSampler") as mock_interval, \
             patch("cerberus.samplers.RandomSampler") as mock_random, \
             patch("cerberus.samplers.ComplexityMatchedSampler") as mock_cms:

            mock_interval_inst = MagicMock(spec=IntervalSampler)
            mock_interval_inst.__len__.return_value = 50
            intervals = [Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(50)]
            mock_interval_inst.__iter__.return_value = iter(intervals)
            mock_interval_inst.__getitem__.side_effect = lambda i: intervals[i]
            mock_interval.return_value = mock_interval_inst

            mock_random.return_value = MagicMock(spec=RandomSampler)
            mock_cms.return_value = MagicMock(spec=ComplexityMatchedSampler)

            yield {
                "interval_sampler": mock_interval,
                "random_sampler": mock_random,
                "complexity_matched_sampler": mock_cms,
            }

    def test_peak_sampler_passes_cache(self, mock_peak_deps):
        """PeakSampler passes prepare_cache as metrics_cache to ComplexityMatchedSampler."""
        cache: dict[str, np.ndarray] = {"chr1:0-100(+)": np.array([0.5], dtype=np.float32)}

        PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
            prepare_cache=cache,
        )

        mock_cms = mock_peak_deps["complexity_matched_sampler"]
        mock_cms.assert_called_once()
        assert mock_cms.call_args.kwargs["metrics_cache"] is cache

    def test_peak_sampler_without_cache(self, mock_peak_deps):
        """PeakSampler without prepare_cache passes None as metrics_cache."""
        PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
        )

        mock_cms = mock_peak_deps["complexity_matched_sampler"]
        mock_cms.assert_called_once()
        assert mock_cms.call_args.kwargs.get("metrics_cache") is None

    def test_peak_sampler_no_background_ignores_cache(self, mock_peak_deps):
        """PeakSampler with background_ratio=0 doesn't create ComplexityMatchedSampler."""
        cache: dict[str, np.ndarray] = {"chr1:0-100(+)": np.array([0.5], dtype=np.float32)}

        sampler = PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
            background_ratio=0,
            prepare_cache=cache,
        )

        mock_peak_deps["complexity_matched_sampler"].assert_not_called()
        assert sampler.negatives is None


class TestCreateSamplerCacheIgnoredForSimpleSamplers:
    """Tests that prepare_cache is harmlessly ignored for non-complexity samplers."""

    def test_random_sampler_ignores_cache(self):
        """create_sampler() with random type ignores prepare_cache."""
        cache: dict[str, np.ndarray] = {"chr1:0-100(+)": np.array([0.5], dtype=np.float32)}

        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "random",
            "padded_size": 100,
            "sampler_args": {"num_intervals": 10},
        })

        sampler = create_sampler(
            config,
            chrom_sizes={"chr1": 10000},
            folds=[],
            exclude_intervals={},
            prepare_cache=cache,
        )

        assert not hasattr(sampler, "prepare_cache")


class TestDatasetPrepareCache:
    """Tests that CerberusDataset threads prepare_cache through to create_sampler."""

    @pytest.fixture
    def _mock_dataset_init(self):
        """Mock all heavy CerberusDataset init dependencies to isolate prepare_cache plumbing."""
        genome_config = {
            "fasta_path": "mock.fa",
            "chrom_sizes": {"chr1": 10000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2},
            "exclude_intervals": [],
        }
        data_config = {
            "use_sequence": False,
            "encoding": "one_hot",
            "inputs": {"sig": "mock.bw"},
            "targets": {},
            "input_length": 1000,
            "output_length": 1000,
            "target_scale": 1.0,
            "reverse_complement": False,
            "random_shift": 0,
        }
        sampler_config = {
            "sampler_type": "random",
            "padded_size": 1000,
            "sampler_args": {"num_intervals": 10},
        }

        patches = [
            patch("cerberus.dataset.validate_genome_config", return_value=genome_config),
            patch("cerberus.dataset.validate_data_config", return_value=data_config),
            patch("cerberus.dataset.validate_sampler_config", return_value=sampler_config),
            patch("cerberus.dataset.validate_data_and_sampler_compatibility"),
            patch("cerberus.dataset.create_genome_folds", return_value=[]),
            patch("cerberus.dataset.get_exclude_intervals", return_value={}),
            patch("cerberus.dataset.create_sampler"),
            patch("cerberus.dataset.UniversalExtractor"),
            patch("cerberus.dataset.create_default_transforms", return_value=[]),
        ]

        mocks = [p.start() for p in patches]
        mock_create_ref = mocks[6]  # create_sampler mock
        mock_create_ref.return_value = MagicMock()

        yield {
            "genome_config": genome_config,
            "data_config": data_config,
            "sampler_config": sampler_config,
            "create_sampler": mock_create_ref,
        }

        for p in patches:
            p.stop()

    def test_dataset_passes_cache_to_sampler(self, _mock_dataset_init):
        """CerberusDataset(prepare_cache=cache) passes it through _initialize_sampler."""
        from cerberus.dataset import CerberusDataset

        m = _mock_dataset_init
        cache: dict[str, np.ndarray] = {
            "chr1:0-100(+)": np.array([0.5, 0.3, 0.7], dtype=np.float32),
        }

        CerberusDataset(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            prepare_cache=cache,
        )

        m["create_sampler"].assert_called_once()
        assert m["create_sampler"].call_args.kwargs["prepare_cache"] is cache

    def test_dataset_without_cache(self, _mock_dataset_init):
        """CerberusDataset without prepare_cache passes None to create_sampler."""
        from cerberus.dataset import CerberusDataset

        m = _mock_dataset_init

        CerberusDataset(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
        )

        m["create_sampler"].assert_called_once()
        assert m["create_sampler"].call_args.kwargs["prepare_cache"] is None
