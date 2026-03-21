"""
Comprehensive tests for all changes since v0.9.2.

Covers:
- generate_sub_seeds: edge cases, type enforcement, determinism
- Sampler seed signatures: int-only __init__, default=42
- PeakSampler seed propagation to child samplers
- create_sampler seed propagation for complexity_matched and peak types
- CerberusDataset prepare_cache threading
- cache.py: resolve_cache_dir edge cases, save/load edge cases
- CerberusDataModule: seed storage, _resolve_cache_dir, prepare_data for peak sampler
- train_single / train_multi: seed pass-through, fold rotation
- Ensemble metadata: creation, update, corruption recovery
"""

import logging
import random
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import yaml
from interlap import InterLap

from cerberus.cache import (
    get_default_cache_dir,
    load_prepare_cache,
    resolve_cache_dir,
    save_prepare_cache,
)
from cerberus.config import SamplerConfig
from cerberus.interval import Interval
from cerberus.samplers import (
    ComplexityMatchedSampler,
    ListSampler,
    MultiSampler,
    RandomSampler,
    ScaledSampler,
    create_sampler,
    generate_sub_seeds,
)


# ---------------------------------------------------------------------------
# 1. generate_sub_seeds
# ---------------------------------------------------------------------------


class TestGenerateSubSeeds:
    """Comprehensive tests for generate_sub_seeds."""

    def test_zero_sub_seeds(self):
        """n=0 returns empty list."""
        assert generate_sub_seeds(42, 0) == []

    def test_single_sub_seed(self):
        """n=1 returns list of one int."""
        result = generate_sub_seeds(42, 1)
        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_large_n(self):
        """n=1000 returns 1000 unique ints."""
        result = generate_sub_seeds(42, 1000)
        assert len(result) == 1000
        assert len(set(result)) == 1000

    def test_deterministic_across_calls(self):
        """Same seed + n always produces same list."""
        a = generate_sub_seeds(0, 10)
        b = generate_sub_seeds(0, 10)
        assert a == b

    def test_different_master_seeds_differ(self):
        """Different master seeds produce different sub-seeds."""
        a = generate_sub_seeds(1, 5)
        b = generate_sub_seeds(2, 5)
        assert a != b

    def test_all_ints(self):
        """Every element is int, not None."""
        result = generate_sub_seeds(99, 20)
        assert all(isinstance(s, int) for s in result)

    def test_output_never_none(self):
        """Output list never contains None values."""
        result = generate_sub_seeds(42, 10)
        assert None not in result


# ---------------------------------------------------------------------------
# 2. Sampler seed defaults
# ---------------------------------------------------------------------------


class TestSamplerSeedDefaults:
    """Verify all sampler __init__ signatures default seed to 42."""

    def test_random_sampler_default_seed(self):
        rs = RandomSampler(chrom_sizes={"chr1": 10000}, padded_size=100, num_intervals=5)
        assert rs.seed == 42

    def test_random_sampler_explicit_seed(self):
        rs = RandomSampler(chrom_sizes={"chr1": 10000}, padded_size=100, num_intervals=5, seed=99)
        assert rs.seed == 99

    def test_scaled_sampler_default_seed(self):
        intervals = [Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(10)]
        base = ListSampler(intervals)
        ss = ScaledSampler(base, num_samples=5)
        assert ss.seed == 42

    def test_multi_sampler_default_seed(self):
        rs = RandomSampler(chrom_sizes={"chr1": 10000}, padded_size=100, num_intervals=5, seed=1)
        ms = MultiSampler([rs], chrom_sizes={"chr1": 10000}, exclude_intervals={})
        assert ms.seed == 42


# ---------------------------------------------------------------------------
# 3. PeakSampler seed propagation
# ---------------------------------------------------------------------------


class TestPeakSamplerSeedPropagation:
    """Verify PeakSampler passes its seed to child samplers."""

    @pytest.fixture
    def _mock_peak_children(self):
        """Mock child sampler constructors to capture seed arguments."""
        with patch("cerberus.samplers.IntervalSampler") as mock_interval, \
             patch("cerberus.samplers.RandomSampler") as mock_random, \
             patch("cerberus.samplers.ComplexityMatchedSampler") as mock_cms:

            mock_interval_inst = MagicMock()
            mock_interval_inst.__len__.return_value = 50
            intervals = [Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(50)]
            mock_interval_inst.__iter__.return_value = iter(intervals)
            mock_interval_inst.__getitem__.side_effect = lambda i: intervals[i]
            mock_interval.return_value = mock_interval_inst

            mock_random.return_value = MagicMock()
            mock_cms.return_value = MagicMock()

            yield {
                "interval_sampler": mock_interval,
                "random_sampler": mock_random,
                "complexity_matched_sampler": mock_cms,
            }

    def test_seed_passed_to_candidates(self, _mock_peak_children):
        """RandomSampler (candidates) receives seed from PeakSampler."""
        from cerberus.samplers import PeakSampler

        PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
            seed=77,
        )

        mock_random = _mock_peak_children["random_sampler"]
        assert mock_random.call_args.kwargs["seed"] == 77

    def test_seed_passed_to_negatives(self, _mock_peak_children):
        """ComplexityMatchedSampler (negatives) receives seed from PeakSampler."""
        from cerberus.samplers import PeakSampler

        PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
            seed=77,
        )

        mock_cms = _mock_peak_children["complexity_matched_sampler"]
        assert mock_cms.call_args.kwargs["seed"] == 77

    def test_no_background_skips_children(self, _mock_peak_children):
        """With background_ratio=0, RandomSampler and CMS are never created."""
        from cerberus.samplers import PeakSampler

        PeakSampler(
            intervals_path="peaks.bed",
            fasta_path="genome.fa",
            chrom_sizes={"chr1": 10000},
            padded_size=50,
            background_ratio=0,
            seed=77,
        )

        _mock_peak_children["random_sampler"].assert_not_called()
        _mock_peak_children["complexity_matched_sampler"].assert_not_called()


# ---------------------------------------------------------------------------
# 4. create_sampler seed propagation for complexity_matched
# ---------------------------------------------------------------------------


class TestCreateSamplerSeedPropagation:
    """Verify create_sampler uses generate_sub_seeds for child samplers."""

    @pytest.fixture(autouse=True)
    def _mock_complexity(self):
        with patch("cerberus.samplers.compute_intervals_complexity") as m:
            m.side_effect = lambda intervals, fasta, metrics, center_size=None: np.random.rand(
                len(intervals), len(metrics)
            ).astype(np.float32)
            yield m

    def test_complexity_matched_child_seeds_differ(self):
        """Target and candidate sub-samplers of complexity_matched get different seeds."""
        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "complexity_matched",
            "padded_size": 100,
            "sampler_args": {
                "target_sampler": {"type": "random", "args": {"num_intervals": 5}},
                "candidate_sampler": {"type": "random", "args": {"num_intervals": 10}},
                "bins": 10,
                "candidate_ratio": 1.0,
                "metrics": ["gc"],
            },
        })
        chrom_sizes = {"chr1": 100_000}

        sampler = create_sampler(
            config, chrom_sizes=chrom_sizes, folds=[], exclude_intervals={},
            fasta_path="mock.fa", seed=42,
        )
        assert isinstance(sampler, ComplexityMatchedSampler)

        # The target and candidate samplers should have different derived seeds
        assert isinstance(sampler.target_sampler, RandomSampler)
        assert isinstance(sampler.candidate_sampler, RandomSampler)
        target_seed = sampler.target_sampler.seed
        candidate_seed = sampler.candidate_sampler.seed
        assert target_seed != candidate_seed

    def test_same_seed_reproduces_sampler(self):
        """create_sampler with same seed produces identical intervals."""
        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "random",
            "padded_size": 100,
            "sampler_args": {"num_intervals": 20},
        })

        s1 = create_sampler(config, {"chr1": 100000}, [], {}, seed=42)
        s2 = create_sampler(config, {"chr1": 100000}, [], {}, seed=42)
        assert [str(i) for i in s1] == [str(i) for i in s2]

    def test_different_seed_produces_different_sampler(self):
        """create_sampler with different seed produces different intervals."""
        config: SamplerConfig = cast(SamplerConfig, {
            "sampler_type": "random",
            "padded_size": 100,
            "sampler_args": {"num_intervals": 20},
        })

        s1 = create_sampler(config, {"chr1": 1_000_000}, [], {}, seed=42)
        s2 = create_sampler(config, {"chr1": 1_000_000}, [], {}, seed=99)
        assert [str(i) for i in s1] != [str(i) for i in s2]


# ---------------------------------------------------------------------------
# 5. cache.py edge cases
# ---------------------------------------------------------------------------


class TestCacheEdgeCases:
    """Edge-case tests for cache utilities."""

    def test_resolve_cache_dir_different_chrom_sizes(self, tmp_path):
        """Different chrom_sizes produce different cache dirs."""
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")
        config = cast(SamplerConfig, {"sampler_type": "peak", "sampler_args": {}, "padded_size": 100})

        dir_a = resolve_cache_dir(tmp_path, fasta, config, seed=42, chrom_sizes={"chr1": 1000})
        dir_b = resolve_cache_dir(tmp_path, fasta, config, seed=42, chrom_sizes={"chr1": 2000})
        assert dir_a != dir_b

    def test_resolve_cache_dir_different_padded_size(self, tmp_path):
        """Different padded_size in config produce different cache dirs."""
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")
        chrom_sizes = {"chr1": 10000}

        config_a = cast(SamplerConfig, {"sampler_type": "peak", "sampler_args": {}, "padded_size": 100})
        config_b = cast(SamplerConfig, {"sampler_type": "peak", "sampler_args": {}, "padded_size": 200})

        dir_a = resolve_cache_dir(tmp_path, fasta, config_a, seed=42, chrom_sizes=chrom_sizes)
        dir_b = resolve_cache_dir(tmp_path, fasta, config_b, seed=42, chrom_sizes=chrom_sizes)
        assert dir_a != dir_b

    def test_save_empty_cache(self, tmp_path):
        """Saving and loading an empty cache works."""
        save_prepare_cache(tmp_path / "empty", {})
        loaded = load_prepare_cache(tmp_path / "empty")
        assert loaded is not None
        assert len(loaded) == 0

    def test_save_large_cache(self, tmp_path):
        """Saving and loading a cache with many entries works."""
        cache = {
            f"chr1:{i*100}-{i*100+100}(+)": np.random.rand(3).astype(np.float32)
            for i in range(1000)
        }
        save_prepare_cache(tmp_path / "large", cache)
        loaded = load_prepare_cache(tmp_path / "large")
        assert loaded is not None
        assert len(loaded) == 1000
        for k in cache:
            np.testing.assert_array_almost_equal(loaded[k], cache[k])

    def test_save_overwrites_existing(self, tmp_path):
        """Saving twice overwrites the previous cache."""
        cache1 = {"chr1:0-100(+)": np.array([0.1], dtype=np.float32)}
        cache2 = {"chr1:0-100(+)": np.array([0.9], dtype=np.float32)}

        save_prepare_cache(tmp_path, cache1)
        save_prepare_cache(tmp_path, cache2)

        loaded = load_prepare_cache(tmp_path)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["chr1:0-100(+)"], [0.9])


# ---------------------------------------------------------------------------
# 6. CerberusDataModule seed and cache_dir
# ---------------------------------------------------------------------------


class TestDataModuleSeedStorage:
    """Verify CerberusDataModule stores seed and cache_dir correctly."""

    @pytest.fixture(autouse=True)
    def _mock_validators(self):
        config = {}
        with patch("cerberus.datamodule.validate_genome_config", return_value={
            "fasta_path": "pyproject.toml",
            "chrom_sizes": {"chr1": 1000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
            "exclude_intervals": {},
        }), \
            patch("cerberus.datamodule.validate_data_config", return_value={}), \
            patch("cerberus.datamodule.validate_sampler_config", return_value={
                "sampler_type": "random",
                "padded_size": 100,
                "sampler_args": {"num_intervals": 10},
            }), \
            patch("cerberus.datamodule.validate_data_and_sampler_compatibility"):
            yield

    def test_default_seed(self):
        from cerberus.datamodule import CerberusDataModule
        dm = CerberusDataModule(
            genome_config={}, data_config={}, sampler_config={},  # type: ignore
        )
        assert dm.seed == 42

    def test_explicit_seed(self):
        from cerberus.datamodule import CerberusDataModule
        dm = CerberusDataModule(
            genome_config={}, data_config={}, sampler_config={},  # type: ignore
            seed=999,
        )
        assert dm.seed == 999

    def test_default_cache_dir(self):
        from cerberus.datamodule import CerberusDataModule
        dm = CerberusDataModule(
            genome_config={}, data_config={}, sampler_config={},  # type: ignore
        )
        assert dm.cache_dir == get_default_cache_dir()

    def test_explicit_cache_dir(self, tmp_path):
        from cerberus.datamodule import CerberusDataModule
        dm = CerberusDataModule(
            genome_config={}, data_config={}, sampler_config={},  # type: ignore
            cache_dir=tmp_path / "custom",
        )
        assert dm.cache_dir == tmp_path / "custom"

    def test_resolve_cache_dir_none_for_random(self):
        """_resolve_cache_dir returns None for random sampler."""
        from cerberus.datamodule import CerberusDataModule
        dm = CerberusDataModule(
            genome_config={}, data_config={}, sampler_config={},  # type: ignore
        )
        assert dm._resolve_cache_dir() is None

    def test_resolve_cache_dir_returns_path_for_peak(self, tmp_path):
        """_resolve_cache_dir returns a Path for peak sampler."""
        from cerberus.datamodule import CerberusDataModule

        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        with patch("cerberus.datamodule.validate_genome_config", return_value={
            "fasta_path": str(fasta),
            "chrom_sizes": {"chr1": 1000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
            "exclude_intervals": {},
        }), \
            patch("cerberus.datamodule.validate_data_config", return_value={}), \
            patch("cerberus.datamodule.validate_sampler_config", return_value={
                "sampler_type": "peak",
                "padded_size": 100,
                "sampler_args": {"intervals_path": "peaks.bed", "background_ratio": 1.0},
            }), \
            patch("cerberus.datamodule.validate_data_and_sampler_compatibility"):

            dm = CerberusDataModule(
                genome_config={}, data_config={}, sampler_config={},  # type: ignore
                cache_dir=tmp_path / "cache",
            )
            result = dm._resolve_cache_dir()
            assert result is not None
            assert str(result).startswith(str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# 7. train_single / train_multi seed propagation
# ---------------------------------------------------------------------------


class TestTrainSeedPropagation:
    """Verify train_single and train_multi pass seed to CerberusDataModule."""

    def test_train_single_passes_seed(self):
        """train_single(seed=77) passes seed=77 to CerberusDataModule."""
        with patch("cerberus.train.CerberusDataModule") as mock_dm_cls, \
             patch("cerberus.train.instantiate"), \
             patch("cerberus.train._train") as mock_train:

            from cerberus.train import train_single

            train_single(
                genome_config={"fold_args": {"k": 3}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                test_fold=0,
                root_dir="/tmp/test_seed",
                seed=77,
            )

            mock_dm_cls.assert_called_once()
            assert mock_dm_cls.call_args.kwargs["seed"] == 77

    def test_train_single_default_seed(self):
        """train_single() defaults seed to 42."""
        with patch("cerberus.train.CerberusDataModule") as mock_dm_cls, \
             patch("cerberus.train.instantiate"), \
             patch("cerberus.train._train"):

            from cerberus.train import train_single

            train_single(
                genome_config={"fold_args": {"k": 3}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                root_dir="/tmp/test_seed_default",
            )

            assert mock_dm_cls.call_args.kwargs["seed"] == 42

    def test_train_multi_passes_seed_to_all_folds(self):
        """train_multi(seed=55) passes seed=55 to every train_single call."""
        with patch("cerberus.train.train_single") as mock_ts:
            mock_ts.return_value = MagicMock()

            from cerberus.train import train_multi

            train_multi(
                genome_config={"fold_args": {"k": 3}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                root_dir="/tmp/test_multi_seed",
                seed=55,
            )

            assert mock_ts.call_count == 3
            for c in mock_ts.call_args_list:
                assert c.kwargs["seed"] == 55

    def test_train_multi_rotating_val_fold(self):
        """train_multi uses val_fold=(i+1)%k for each fold."""
        with patch("cerberus.train.train_single") as mock_ts:
            mock_ts.return_value = MagicMock()

            from cerberus.train import train_multi

            train_multi(
                genome_config={"fold_args": {"k": 4}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                root_dir="/tmp/test_multi_val",
            )

            assert mock_ts.call_count == 4
            expected_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for c, (test_f, val_f) in zip(mock_ts.call_args_list, expected_pairs):
                assert c.kwargs["test_fold"] == test_f
                assert c.kwargs["val_fold"] == val_f

    def test_train_multi_all_folds_share_root_dir(self):
        """train_multi passes same root_dir to every train_single call."""
        with patch("cerberus.train.train_single") as mock_ts:
            mock_ts.return_value = MagicMock()

            from cerberus.train import train_multi

            train_multi(
                genome_config={"fold_args": {"k": 3}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                root_dir="/tmp/shared_root",
            )

            for c in mock_ts.call_args_list:
                assert c.kwargs["root_dir"] == "/tmp/shared_root"


# ---------------------------------------------------------------------------
# 8. train_single fold directory and metadata
# ---------------------------------------------------------------------------


class TestTrainSingleFoldStructure:
    """Verify train_single creates fold subdirectory and updates metadata."""

    def test_fold_dir_created(self, tmp_path):
        """train_single creates fold_N subdirectory."""
        with patch("cerberus.train.CerberusDataModule"), \
             patch("cerberus.train.instantiate"), \
             patch("cerberus.train._train") as mock_train:

            from cerberus.train import train_single

            root = tmp_path / "exp"
            train_single(
                genome_config={"fold_args": {"k": 5}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                test_fold=2,
                root_dir=root,
            )

            # _train should receive fold_2 as root_dir
            call_kwargs = mock_train.call_args.kwargs
            assert Path(call_kwargs["root_dir"]) == root / "fold_2"

    def test_metadata_created(self, tmp_path):
        """train_single creates ensemble_metadata.yaml."""
        with patch("cerberus.train.CerberusDataModule"), \
             patch("cerberus.train.instantiate"), \
             patch("cerberus.train._train"):

            from cerberus.train import train_single

            root = tmp_path / "exp_meta"
            train_single(
                genome_config={"fold_args": {"k": 5}},  # type: ignore
                data_config={},  # type: ignore
                sampler_config={},  # type: ignore
                model_config={},  # type: ignore
                train_config={},  # type: ignore
                test_fold=1,
                root_dir=root,
            )

            meta_path = root / "ensemble_metadata.yaml"
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            assert meta["folds"] == [1]

    def test_metadata_accumulates_folds(self, tmp_path):
        """Sequential train_single calls accumulate folds in metadata."""
        with patch("cerberus.train.CerberusDataModule"), \
             patch("cerberus.train.instantiate"), \
             patch("cerberus.train._train"):

            from cerberus.train import train_single

            root = tmp_path / "exp_accum"
            for fold in [0, 2, 1]:
                train_single(
                    genome_config={"fold_args": {"k": 5}},  # type: ignore
                    data_config={},  # type: ignore
                    sampler_config={},  # type: ignore
                    model_config={},  # type: ignore
                    train_config={},  # type: ignore
                    test_fold=fold,
                    root_dir=root,
                )

            with open(root / "ensemble_metadata.yaml") as f:
                meta = yaml.safe_load(f)
            assert set(meta["folds"]) == {0, 1, 2}


# ---------------------------------------------------------------------------
# 9. Ensemble metadata edge cases
# ---------------------------------------------------------------------------


class TestEnsembleMetadata:
    """Tests for update_ensemble_metadata edge cases."""

    def test_corrupt_metadata_warns(self, tmp_path):
        """Corrupt YAML triggers a warning and recovers."""
        import io
        from cerberus.model_ensemble import update_ensemble_metadata

        meta_path = tmp_path / "ensemble_metadata.yaml"
        meta_path.write_text(": :\\n  invalid: [yaml: {{{")

        # Capture log output directly via a temporary handler
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("cerberus.model_ensemble")
        logger.addHandler(handler)
        try:
            update_ensemble_metadata(tmp_path, fold=5)
        finally:
            logger.removeHandler(handler)

        assert "Corrupt ensemble metadata" in log_buffer.getvalue()

        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [5]

    def test_idempotent_fold_addition(self, tmp_path):
        """Adding same fold twice does not duplicate it."""
        from cerberus.model_ensemble import update_ensemble_metadata

        update_ensemble_metadata(tmp_path, fold=0)
        update_ensemble_metadata(tmp_path, fold=0)

        with open(tmp_path / "ensemble_metadata.yaml") as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0]

    def test_sorted_output(self, tmp_path):
        """Folds are stored sorted."""
        from cerberus.model_ensemble import update_ensemble_metadata

        for fold in [3, 1, 4, 0, 2]:
            update_ensemble_metadata(tmp_path, fold=fold)

        with open(tmp_path / "ensemble_metadata.yaml") as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 10. prepare_data with peak sampler
# ---------------------------------------------------------------------------


class TestPrepareDataPeakSampler:
    """Verify prepare_data() works for peak sampler type."""

    @pytest.fixture
    def _mock_peak_datamodule(self, tmp_path):
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        genome_config = {
            "fasta_path": str(fasta),
            "chrom_sizes": {"chr1": 10000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
            "exclude_intervals": [],
        }
        data_config = {
            "use_sequence": False,
            "inputs": {},
            "targets": {},
        }
        sampler_config = {
            "sampler_type": "peak",
            "padded_size": 100,
            "sampler_args": {
                "intervals_path": "peaks.bed",
                "background_ratio": 1.0,
            },
        }

        fake_cache = {
            "chr1:0-100(+)": np.array([0.5], dtype=np.float32),
        }

        # Mock negatives with metrics_cache
        mock_negatives = MagicMock()
        mock_negatives.metrics_cache = fake_cache

        mock_sampler = MagicMock()
        mock_sampler.negatives = mock_negatives

        mock_dataset = MagicMock()
        mock_dataset.sampler = mock_sampler

        patches = [
            patch("cerberus.datamodule.validate_genome_config", return_value=genome_config),
            patch("cerberus.datamodule.validate_data_config", return_value=data_config),
            patch("cerberus.datamodule.validate_sampler_config", return_value=sampler_config),
            patch("cerberus.datamodule.validate_data_and_sampler_compatibility"),
            patch("cerberus.datamodule.CerberusDataset", return_value=mock_dataset),
        ]

        mocks = [p.start() for p in patches]

        yield {
            "genome_config": genome_config,
            "data_config": data_config,
            "sampler_config": sampler_config,
            "tmp_path": tmp_path,
            "fake_cache": fake_cache,
            "mock_dataset_cls": mocks[4],
        }

        for p in patches:
            p.stop()

    def test_peak_prepare_data_creates_cache(self, _mock_peak_datamodule):
        """prepare_data() with peak sampler writes cache from negatives.metrics_cache."""
        from cerberus.datamodule import CerberusDataModule

        m = _mock_peak_datamodule
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

        loaded = load_prepare_cache(cache_dir)
        assert loaded is not None
        assert set(loaded.keys()) == set(m["fake_cache"].keys())


# ---------------------------------------------------------------------------
# 11. Datamodule seed formula (epoch, rank, world_size)
# ---------------------------------------------------------------------------


class TestDataModuleSeedFormula:
    """Verify the resample seed formula: base + epoch * world_size + rank."""

    @pytest.fixture
    def _dm_with_mock_trainer(self):
        """Create a DataModule with mocked trainer and datasets."""
        with patch("cerberus.datamodule.validate_genome_config", return_value={
            "fasta_path": "pyproject.toml",
            "chrom_sizes": {"chr1": 1000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
            "exclude_intervals": {},
        }), \
            patch("cerberus.datamodule.validate_data_config", return_value={}), \
            patch("cerberus.datamodule.validate_sampler_config", return_value={
                "sampler_type": "random",
                "padded_size": 100,
                "sampler_args": {"num_intervals": 10},
            }), \
            patch("cerberus.datamodule.validate_data_and_sampler_compatibility"):

            from cerberus.datamodule import CerberusDataModule

            dm = CerberusDataModule(
                genome_config={}, data_config={}, sampler_config={},  # type: ignore
                seed=100,
            )

            # Mock trainer
            trainer = MagicMock()
            dm.trainer = trainer  # type: ignore

            # Mock datasets
            train_ds = MagicMock()
            train_ds.__len__ = MagicMock(return_value=50)
            dm.train_dataset = train_ds  # type: ignore
            dm._is_initialized = True

            yield dm, trainer, train_ds

    def test_epoch_0_rank_0(self, _dm_with_mock_trainer):
        dm, trainer, train_ds = _dm_with_mock_trainer
        trainer.current_epoch = 0
        trainer.global_rank = 0
        trainer.world_size = 1

        dm.train_dataloader()
        train_ds.resample.assert_called_with(seed=100)

    def test_epoch_2_rank_0_world_4(self, _dm_with_mock_trainer):
        """seed = 100 + 2*4 + 0 = 108"""
        dm, trainer, train_ds = _dm_with_mock_trainer
        trainer.current_epoch = 2
        trainer.global_rank = 0
        trainer.world_size = 4

        dm.train_dataloader()
        train_ds.resample.assert_called_with(seed=108)

    def test_epoch_1_rank_3_world_4(self, _dm_with_mock_trainer):
        """seed = 100 + 1*4 + 3 = 107"""
        dm, trainer, train_ds = _dm_with_mock_trainer
        trainer.current_epoch = 1
        trainer.global_rank = 3
        trainer.world_size = 4

        dm.train_dataloader()
        train_ds.resample.assert_called_with(seed=107)

    def test_ranks_get_different_seeds(self, _dm_with_mock_trainer):
        """Different ranks at same epoch get different seeds."""
        dm, trainer, train_ds = _dm_with_mock_trainer
        trainer.current_epoch = 0
        trainer.world_size = 4

        seeds = []
        for rank in range(4):
            trainer.global_rank = rank
            train_ds.resample.reset_mock()
            dm.train_dataloader()
            seeds.append(train_ds.resample.call_args.kwargs["seed"])

        assert len(set(seeds)) == 4


# ---------------------------------------------------------------------------
# 12. CerberusDataset prepare_cache → create_sampler full chain
# ---------------------------------------------------------------------------


class TestDatasetSamplerCacheChain:
    """Verify CerberusDataset passes prepare_cache through _initialize_sampler to create_sampler."""

    @pytest.fixture
    def _patched_dataset(self):
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

        started = [p.start() for p in patches]
        cs_mock_ref = started[6]
        cs_mock_ref.return_value = MagicMock()

        yield {
            "genome_config": genome_config,
            "data_config": data_config,
            "sampler_config": sampler_config,
            "create_sampler": cs_mock_ref,
        }

        for p in patches:
            p.stop()

    def test_seed_and_cache_passed_together(self, _patched_dataset):
        """CerberusDataset passes both seed and prepare_cache to create_sampler."""
        from cerberus.dataset import CerberusDataset

        m = _patched_dataset
        cache = {"k": np.array([1.0], dtype=np.float32)}

        CerberusDataset(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            seed=77,
            prepare_cache=cache,
        )

        kwargs = m["create_sampler"].call_args.kwargs
        assert kwargs["seed"] == 77
        assert kwargs["prepare_cache"] is cache

    def test_no_cache_passes_none(self, _patched_dataset):
        """CerberusDataset without cache passes prepare_cache=None."""
        from cerberus.dataset import CerberusDataset

        m = _patched_dataset
        CerberusDataset(
            genome_config=m["genome_config"],
            data_config=m["data_config"],
            sampler_config=m["sampler_config"],
            seed=42,
        )

        kwargs = m["create_sampler"].call_args.kwargs
        assert kwargs["prepare_cache"] is None


# ---------------------------------------------------------------------------
# 13. MultiSampler de-correlation via generate_sub_seeds
# ---------------------------------------------------------------------------


class TestMultiSamplerDeCorrelation:
    """Verify MultiSampler assigns distinct derived seeds to sub-samplers."""

    def test_two_identical_sub_samplers_get_different_seeds(self):
        """Two identical RandomSamplers in a MultiSampler get different derived seeds."""
        chrom_sizes = {"chr1": 100_000}
        rs1 = RandomSampler(chrom_sizes, 100, 20, seed=42)
        rs2 = RandomSampler(chrom_sizes, 100, 20, seed=42)

        # Before MultiSampler, they're identical
        assert [str(i) for i in rs1] == [str(i) for i in rs2]

        MultiSampler([rs1, rs2], chrom_sizes, exclude_intervals={}, seed=99)

        # After MultiSampler resamples them with derived seeds, they differ
        assert [str(i) for i in rs1] != [str(i) for i in rs2]

    def test_three_sub_samplers_all_differ(self):
        """Three sub-samplers all get distinct seeds."""
        chrom_sizes = {"chr1": 100_000}
        samplers: list[RandomSampler] = [
            RandomSampler(chrom_sizes, 100, 20, seed=42)
            for _ in range(3)
        ]
        MultiSampler(list(samplers), chrom_sizes, exclude_intervals={}, seed=7)  # type: ignore[arg-type]

        interval_sets = [frozenset(str(i) for i in s) for s in samplers]
        # All three should be different
        assert len(set(interval_sets)) == 3


# ---------------------------------------------------------------------------
# 14. Full seed chain: DataModule → setup → Dataset → create_sampler
# ---------------------------------------------------------------------------


class TestFullSeedChain:
    """End-to-end test verifying seed flows from DataModule all the way to create_sampler."""

    def test_seed_reaches_create_sampler(self, tmp_path):
        """DataModule(seed=77) → setup() → CerberusDataset(seed=77) → create_sampler(seed=77)."""
        fasta = tmp_path / "genome.fa"
        fasta.write_text(">chr1\nACGT\n")

        genome_config = {
            "fasta_path": str(fasta),
            "chrom_sizes": {"chr1": 10000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
            "exclude_intervals": [],
        }
        data_config = {
            "use_sequence": False,
            "encoding": "one_hot",
            "inputs": {},
            "targets": {},
            "target_scale": 1.0,
            "reverse_complement": False,
        }
        sampler_config = {
            "sampler_type": "random",
            "padded_size": 100,
            "sampler_args": {"num_intervals": 10},
        }

        with patch("cerberus.datamodule.validate_genome_config", return_value=genome_config), \
             patch("cerberus.datamodule.validate_data_config", return_value=data_config), \
             patch("cerberus.datamodule.validate_sampler_config", return_value=sampler_config), \
             patch("cerberus.datamodule.validate_data_and_sampler_compatibility"), \
             patch("cerberus.datamodule.CerberusDataset") as mock_ds:

            mock_instance = MagicMock()
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=5)
            mock_val = MagicMock()
            mock_val.__len__ = MagicMock(return_value=3)
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=2)
            mock_instance.split_folds.return_value = (mock_train, mock_val, mock_test)
            mock_ds.return_value = mock_instance

            from cerberus.datamodule import CerberusDataModule

            dm = CerberusDataModule(
                genome_config=genome_config,  # type: ignore
                data_config=data_config,  # type: ignore
                sampler_config=sampler_config,  # type: ignore
                seed=77,
            )
            dm.setup()

            # CerberusDataset was called with seed=77
            assert mock_ds.call_args.kwargs["seed"] == 77


# ---------------------------------------------------------------------------
# 15. _train calls prepare_data before setup
# ---------------------------------------------------------------------------


class TestTrainCallsPrepareThenSetup:
    """Verify _train() calls prepare_data() before setup()."""

    def test_prepare_data_called_before_setup(self):
        """_train calls prepare_data() then setup() in correct order."""
        from cerberus.train import _train

        mock_dm = MagicMock()
        call_order = []
        mock_dm.prepare_data.side_effect = lambda: call_order.append("prepare_data")
        mock_dm.setup.side_effect = lambda **kw: call_order.append("setup")
        mock_dm.compute_median_counts.return_value = 100.0

        with patch("cerberus.train.configure_callbacks", return_value=[]), \
             patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm: mc), \
             patch("cerberus.train.instantiate") as mock_inst, \
             patch("pytorch_lightning.Trainer") as mock_trainer_cls:

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer_cls.return_value = mock_trainer

            _train(
                model_config={"loss_args": {}, "model_args": {}, "pretrained": []},  # type: ignore
                data_config={},  # type: ignore
                datamodule=mock_dm,
                train_config={
                    "batch_size": 32,
                    "max_epochs": 1,
                    "reload_dataloaders_every_n_epochs": 0,
                    "gradient_clip_val": None,
                },  # type: ignore
            )

        assert call_order == ["prepare_data", "setup"]
