"""
Tests for prepare_data() complexity metrics caching.

This test file grows incrementally across commits:
- Commit 1: Sampler + dataset plumbing (prepare_cache passthrough)
- Commit 2: Cache dir resolution and loading helpers
- Commit 3: prepare_data() + setup() integration
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.interval import Interval
from cerberus.samplers import (
    ComplexityMatchedSampler,
    IntervalSampler,
    PeakSampler,
    RandomSampler,
    create_sampler,
)


def _cm_config(target_n: int = 5, candidate_n: int = 10) -> SamplerConfig:
    """Helper: build a complexity_matched SamplerConfig for tests."""
    return SamplerConfig.model_construct(
        sampler_type="complexity_matched",
        padded_size=100,
        sampler_args={
            "target_sampler": SamplerConfig.model_construct(
                sampler_type="random",
                padded_size=100,
                sampler_args={"num_intervals": target_n},
            ),
            "candidate_sampler": SamplerConfig.model_construct(
                sampler_type="random",
                padded_size=100,
                sampler_args={"num_intervals": candidate_n},
            ),
            "bins": 10,
            "candidate_ratio": 1.0,
            "metrics": ["gc"],
        },
    )


# --- Commit 1: Sampler + dataset plumbing tests ---


@pytest.fixture
def mock_compute():
    """Mock compute_intervals_complexity to avoid real FASTA reads."""
    with patch("cerberus.samplers.compute_intervals_complexity") as m:

        def side_effect(intervals, fasta, metrics, center_size=None):
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

        config = _cm_config()
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
        config = _cm_config()
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
        with (
            patch("cerberus.samplers.IntervalSampler") as mock_interval,
            patch("cerberus.samplers.RandomSampler") as mock_random,
            patch("cerberus.samplers.ComplexityMatchedSampler") as mock_cms,
        ):
            mock_interval_inst = MagicMock(spec=IntervalSampler)
            mock_interval_inst.__len__.return_value = 50
            intervals = [
                Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(50)
            ]
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
        cache: dict[str, np.ndarray] = {
            "chr1:0-100(+)": np.array([0.5], dtype=np.float32)
        }

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
        cache: dict[str, np.ndarray] = {
            "chr1:0-100(+)": np.array([0.5], dtype=np.float32)
        }

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
        cache: dict[str, np.ndarray] = {
            "chr1:0-100(+)": np.array([0.5], dtype=np.float32)
        }

        config = SamplerConfig.model_construct(
            sampler_type="random",
            padded_size=100,
            sampler_args={"num_intervals": 10},
        )

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
        genome_config = GenomeConfig.model_construct(
            name="test",
            fasta_path="mock.fa",
            chrom_sizes={"chr1": 10000},
            allowed_chroms=["chr1"],
            exclude_intervals={},
            fold_type="chrom_partition",
            fold_args={"k": 2, "test_fold": None, "val_fold": None},
        )
        data_config = DataConfig.model_construct(
            inputs={"sig": "mock.bw"},
            targets={},
            input_len=1000,
            output_len=1000,
            output_bin_size=1,
            max_jitter=0,
            encoding="one_hot",
            log_transform=False,
            reverse_complement=False,
            target_scale=1.0,
            use_sequence=False,
        )
        sampler_config = SamplerConfig.model_construct(
            sampler_type="random",
            padded_size=1000,
            sampler_args={"num_intervals": 10},
        )

        patches = [
            patch("cerberus.dataset.create_genome_folds", return_value=[]),
            patch("cerberus.dataset.get_exclude_intervals", return_value={}),
            patch("cerberus.dataset.create_sampler"),
            patch("cerberus.dataset.UniversalExtractor"),
            patch("cerberus.dataset.create_default_transforms", return_value=[]),
        ]

        mocks = [p.start() for p in patches]
        mock_create_ref = mocks[2]  # create_sampler mock
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


# --- Seed determinism verification ---


class TestSeedDeterminism:
    """Verify that explicit seeds produce deterministic sampler intervals."""

    def test_different_seeds_produce_different_candidates(self, mock_compute):
        """Two create_sampler() calls with different seeds produce different candidate intervals."""
        config = _cm_config(target_n=5, candidate_n=50)
        chrom_sizes = {"chr1": 1_000_000}

        sampler1 = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=42,
        )
        assert isinstance(sampler1, ComplexityMatchedSampler)
        intervals1 = {str(iv) for iv in sampler1.candidate_sampler}

        sampler2 = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=123,
        )
        assert isinstance(sampler2, ComplexityMatchedSampler)
        intervals2 = {str(iv) for iv in sampler2.candidate_sampler}

        # Different seeds should produce different intervals
        overlap = intervals1 & intervals2
        assert len(overlap) < len(intervals1) * 0.5, (
            f"Expected mostly different intervals with different seeds, "
            f"but got {len(overlap)}/{len(intervals1)} overlap"
        )

    def test_explicit_seed_produces_identical_candidates(self, mock_compute):
        """Two create_sampler() calls with seed=42 produce identical candidate intervals."""
        config = _cm_config(target_n=5, candidate_n=50)
        chrom_sizes = {"chr1": 1_000_000}

        sampler1 = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=42,
        )
        assert isinstance(sampler1, ComplexityMatchedSampler)
        intervals1 = [str(iv) for iv in sampler1.candidate_sampler]

        sampler2 = create_sampler(
            config,
            chrom_sizes=chrom_sizes,
            folds=[],
            exclude_intervals={},
            fasta_path="mock.fa",
            seed=42,
        )
        assert isinstance(sampler2, ComplexityMatchedSampler)
        intervals2 = [str(iv) for iv in sampler2.candidate_sampler]

        assert intervals1 == intervals2


# --- Commit 2: Cache utilities tests ---

from pathlib import Path

from cerberus.cache import (
    get_default_cache_dir,
    load_prepare_cache,
    resolve_cache_dir,
    save_prepare_cache,
)


class TestGetDefaultCacheDir:
    """Tests for default cache directory resolution."""

    def test_uses_xdg_cache_home_if_set(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/custom_xdg")
        result = get_default_cache_dir()
        assert result == Path("/tmp/custom_xdg/cerberus")

    def test_falls_back_to_home_dot_cache(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = get_default_cache_dir()
        assert result == Path.home() / ".cache" / "cerberus"


class TestResolveCacheDir:
    """Tests for deterministic cache directory computation."""

    def test_deterministic_same_inputs(self, tmp_path):
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        config = SamplerConfig.model_construct(
            sampler_type="peak",
            padded_size=100,
            sampler_args={
                "intervals_path": Path("peaks.bed"),
                "background_ratio": 1.0,
                "complexity_center_size": None,
            },
        )
        chrom_sizes = {"chr1": 10000}

        dir1 = resolve_cache_dir(
            tmp_path, fasta, config, seed=42, chrom_sizes=chrom_sizes
        )
        dir2 = resolve_cache_dir(
            tmp_path, fasta, config, seed=42, chrom_sizes=chrom_sizes
        )
        assert dir1 == dir2

    def test_different_seed_different_dir(self, tmp_path):
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        config = SamplerConfig.model_construct(
            sampler_type="peak",
            padded_size=100,
            sampler_args={
                "intervals_path": Path("peaks.bed"),
                "background_ratio": 1.0,
                "complexity_center_size": None,
            },
        )
        chrom_sizes = {"chr1": 10000}

        dir1 = resolve_cache_dir(
            tmp_path, fasta, config, seed=42, chrom_sizes=chrom_sizes
        )
        dir2 = resolve_cache_dir(
            tmp_path, fasta, config, seed=99, chrom_sizes=chrom_sizes
        )
        assert dir1 != dir2

    def test_different_config_different_dir(self, tmp_path):
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        config_a = SamplerConfig.model_construct(
            sampler_type="peak",
            padded_size=100,
            sampler_args={
                "intervals_path": Path("peaks.bed"),
                "background_ratio": 1.0,
                "complexity_center_size": None,
            },
        )
        config_b = SamplerConfig.model_construct(
            sampler_type="peak",
            padded_size=100,
            sampler_args={
                "intervals_path": Path("peaks.bed"),
                "background_ratio": 2.0,
                "complexity_center_size": None,
            },
        )
        chrom_sizes = {"chr1": 10000}

        dir1 = resolve_cache_dir(
            tmp_path, fasta, config_a, seed=42, chrom_sizes=chrom_sizes
        )
        dir2 = resolve_cache_dir(
            tmp_path, fasta, config_b, seed=42, chrom_sizes=chrom_sizes
        )
        assert dir1 != dir2

    def test_is_subdirectory_of_cache_dir(self, tmp_path):
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        config = SamplerConfig.model_construct(
            sampler_type="peak",
            padded_size=100,
            sampler_args={
                "intervals_path": Path("peaks.bed"),
                "background_ratio": 1.0,
                "complexity_center_size": None,
            },
        )
        result = resolve_cache_dir(
            tmp_path / "my_cache", fasta, config, seed=0, chrom_sizes={}
        )
        assert result.parent == tmp_path / "my_cache"


class TestSaveLoadPrepareCache:
    """Tests for cache serialization round-trip."""

    def test_round_trip(self, tmp_path):
        cache = {
            "chr1:100-200(+)": np.array([0.5, 0.3, 0.7], dtype=np.float32),
            "chr1:300-400(+)": np.array([0.1, 0.9, 0.4], dtype=np.float32),
        }

        save_prepare_cache(tmp_path, cache)

        assert (tmp_path / "metrics_cache.npz").exists()
        assert (tmp_path / "ready").exists()
        assert not (tmp_path / "seed").exists()

        loaded = load_prepare_cache(tmp_path)
        assert loaded is not None
        assert set(loaded.keys()) == set(cache.keys())
        for k in cache:
            np.testing.assert_array_almost_equal(loaded[k], cache[k])

    def test_load_returns_none_when_no_ready(self, tmp_path):
        np.savez_compressed(tmp_path / "metrics_cache.npz", keys=[], values=[])
        assert load_prepare_cache(tmp_path) is None

    def test_load_returns_none_when_no_npz(self, tmp_path):
        (tmp_path / "ready").touch()
        assert load_prepare_cache(tmp_path) is None

    def test_load_returns_none_when_dir_missing(self, tmp_path):
        assert load_prepare_cache(tmp_path / "nonexistent") is None


# --- Commit 3: prepare_data() + setup() integration tests ---


class TestPrepareData:
    """Tests for CerberusDataModule.prepare_data() and setup() integration."""

    @pytest.fixture
    def _mock_datamodule_deps(self, tmp_path):
        """Mock dependencies for CerberusDataModule to isolate prepare_data logic."""
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        genome_config = GenomeConfig.model_construct(
            name="test",
            fasta_path=fasta,
            chrom_sizes={"chr1": 10000},
            allowed_chroms=["chr1"],
            exclude_intervals={},
            fold_type="chrom_partition",
            fold_args={"k": 2, "test_fold": 0, "val_fold": 1},
        )
        data_config = DataConfig.model_construct(
            inputs={},
            targets={},
            input_len=1000,
            output_len=1000,
            output_bin_size=1,
            max_jitter=0,
            encoding="one_hot",
            log_transform=False,
            reverse_complement=False,
            target_scale=1.0,
            use_sequence=False,
        )
        sampler_config = SamplerConfig.model_construct(
            sampler_type="complexity_matched",
            padded_size=100,
            sampler_args={
                "target_sampler": SamplerConfig.model_construct(
                    sampler_type="random",
                    padded_size=100,
                    sampler_args={"num_intervals": 5},
                ),
                "candidate_sampler": SamplerConfig.model_construct(
                    sampler_type="random",
                    padded_size=100,
                    sampler_args={"num_intervals": 10},
                ),
                "bins": 10,
                "candidate_ratio": 1.0,
                "metrics": ["gc"],
            },
        )

        # Build a fake metrics_cache the mock sampler will expose
        fake_cache = {
            "chr1:0-100(+)": np.array([0.5, 0.3], dtype=np.float32),
            "chr1:100-200(+)": np.array([0.7, 0.1], dtype=np.float32),
        }

        mock_sampler = MagicMock()
        mock_sampler.metrics_cache = fake_cache

        # Patch CerberusDataset at the datamodule import site
        mock_dataset = MagicMock()
        mock_dataset.sampler = mock_sampler
        dataset_patch = patch(
            "cerberus.datamodule.CerberusDataset", return_value=mock_dataset
        )
        mock_dataset_cls = dataset_patch.start()

        yield {
            "genome_config": genome_config,
            "data_config": data_config,
            "sampler_config": sampler_config,
            "tmp_path": tmp_path,
            "fake_cache": fake_cache,
            "mock_dataset_cls": mock_dataset_cls,
            "mock_dataset": mock_dataset,
        }

        dataset_patch.stop()

    def test_prepare_data_creates_cache(self, _mock_datamodule_deps):
        """prepare_data() creates cache dir, .npz, and ready sentinel."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_datamodule_deps
        dm = CerberusDataModule(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            cache_dir=m["tmp_path"] / "cache",
        )

        dm.prepare_data()

        cache_dir = dm._resolve_cache_dir()
        assert cache_dir is not None
        assert (cache_dir / "ready").exists()
        assert (cache_dir / "metrics_cache.npz").exists()

        loaded = load_prepare_cache(cache_dir)
        assert loaded is not None
        assert set(loaded.keys()) == set(m["fake_cache"].keys())

    def test_prepare_data_skips_when_ready(self, _mock_datamodule_deps):
        """prepare_data() is a no-op when ready sentinel already exists."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_datamodule_deps
        dm = CerberusDataModule(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            cache_dir=m["tmp_path"] / "cache",
        )

        # Pre-create the cache
        dm.prepare_data()
        m["mock_dataset_cls"].reset_mock()

        # Second call should not create a dataset
        dm.prepare_data()
        m["mock_dataset_cls"].assert_not_called()

    def test_prepare_data_noop_for_random_sampler(self, _mock_datamodule_deps):
        """prepare_data() is a no-op for sampler types that don't need caching."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_datamodule_deps
        random_config = SamplerConfig.model_construct(
            sampler_type="random",
            padded_size=100,
            sampler_args={"num_intervals": 10},
        )
        dm = CerberusDataModule(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=random_config,
            cache_dir=m["tmp_path"] / "cache",
        )

        m["mock_dataset_cls"].reset_mock()
        dm.prepare_data()
        m["mock_dataset_cls"].assert_not_called()

    def test_setup_loads_cache_from_prepare_data(self, _mock_datamodule_deps):
        """setup() loads the cache written by prepare_data() and passes it to CerberusDataset."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_datamodule_deps
        dm = CerberusDataModule(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            cache_dir=m["tmp_path"] / "cache",
        )

        # Run prepare_data to write cache
        dm.prepare_data()
        m["mock_dataset_cls"].reset_mock()

        # Mock split_folds for setup
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=10)
        mock_val = MagicMock()
        mock_val.__len__ = MagicMock(return_value=5)
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=5)
        m["mock_dataset"].split_folds.return_value = (mock_train, mock_val, mock_test)

        dm.setup()

        # Verify CerberusDataset was called with prepare_cache
        m["mock_dataset_cls"].assert_called_once()
        call_kwargs = m["mock_dataset_cls"].call_args.kwargs
        assert call_kwargs["prepare_cache"] is not None
        assert set(call_kwargs["prepare_cache"].keys()) == set(m["fake_cache"].keys())

    def test_setup_works_without_prepare_data(self, _mock_datamodule_deps):
        """setup() works even if prepare_data() was never called (no cache)."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_datamodule_deps
        dm = CerberusDataModule(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            cache_dir=m["tmp_path"] / "cache",
        )

        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=10)
        mock_val = MagicMock()
        mock_val.__len__ = MagicMock(return_value=5)
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=5)
        m["mock_dataset"].split_folds.return_value = (mock_train, mock_val, mock_test)

        dm.setup()

        m["mock_dataset_cls"].assert_called_once()
        assert m["mock_dataset_cls"].call_args.kwargs["prepare_cache"] is None
