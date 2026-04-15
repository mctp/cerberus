"""Tests for cerberus.predict_misc — high-level inference utilities."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ProfileCountOutput
from cerberus.predict_misc import (
    create_eval_dataset,
    load_bed_intervals,
    observed_log_counts,
    predict_log_counts,
)

# ---------------------------------------------------------------------------
# Dummy model — returns ProfileCountOutput with known log_counts
# ---------------------------------------------------------------------------


class DummyBPNetModel(nn.Module):
    """Returns ProfileCountOutput with log_counts = log(100 + pseudocount)."""

    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        B = x.shape[0]
        logits = torch.zeros((B, 1, self.output_dim), device=x.device)
        log_counts = torch.full(
            (B, 1),
            math.log(100.0 + 1.0),
            device=x.device,
        )
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config_and_ensemble(tmp_path, loss_cls, loss_args):
    """Create genome, dataset, ensemble, and return (config, ensemble)."""
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 2000 + "\n")
    (tmp_path / "genome.fa.fai").write_text("chr1\t2000\t6\t2000\t2001\n")

    genome_config = create_genome_config(
        name="test",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={},
    )

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=50,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    model_config = ModelConfig.model_construct(
        name="dummy",
        model_cls="tests.test_predict_misc.DummyBPNetModel",
        loss_cls=loss_cls,
        loss_args=loss_args,
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={"metrics": {}},
        model_args={},
        pretrained=[],
        count_pseudocount=loss_args.get("count_pseudocount", 0.0),
    )

    # Minimal train/sampler configs for CerberusConfig construction
    train_config = TrainConfig.model_construct(
        batch_size=1,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=1,
        optimizer="adam",
        scheduler_type="constant",
        scheduler_args={},
        filter_bias_and_bn=False,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=100,
        sampler_args={},
    )

    config = CerberusConfig.model_construct(
        genome_config=genome_config,
        data_config=data_config,
        model_config_=model_config,
        sampler_config=sampler_config,
        train_config=train_config,
    )

    model = DummyBPNetModel(input_len=100, output_len=50, output_bin_size=1)
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    torch.save(model.state_dict(), fold_dir / "model.pt")

    with open(tmp_path / "ensemble_metadata.yaml", "w") as f:
        yaml.dump({"folds": [0]}, f)

    # Provide a minimal CerberusConfig as the mock return value
    mock_config = CerberusConfig.model_construct(
        genome_config=genome_config,
        data_config=data_config,
        model_config_=model_config,
        sampler_config=sampler_config,
        train_config=train_config,
    )

    with (
        patch(
            "cerberus.model_ensemble.find_latest_hparams",
            return_value=Path("hparams.yaml"),
        ),
        patch("cerberus.model_ensemble.parse_hparams_config", return_value=mock_config),
    ):
        ensemble = ModelEnsemble(
            tmp_path,
            model_config,
            data_config,
            genome_config,
            torch.device("cpu"),
        )

    return config, ensemble


@pytest.fixture
def mse_setup(tmp_path):
    """Config + ensemble with MSE loss (count_pseudocount=1.0)."""
    return _make_config_and_ensemble(
        tmp_path,
        loss_cls="cerberus.loss.MSEMultinomialLoss",
        loss_args={"count_pseudocount": 1.0},
    )


@pytest.fixture
def poisson_setup(tmp_path):
    """Config + ensemble with Poisson loss (no pseudocount)."""
    return _make_config_and_ensemble(
        tmp_path,
        loss_cls="cerberus.loss.PoissonMultinomialLoss",
        loss_args={"count_pseudocount": 0.0},
    )


# ---------------------------------------------------------------------------
# create_eval_dataset
# ---------------------------------------------------------------------------


class TestCreateEvalDataset:
    def test_returns_dataset(self, mse_setup):
        config, _ = mse_setup
        ds = create_eval_dataset(config)
        assert isinstance(ds, CerberusDataset)

    def test_no_sampler(self, mse_setup):
        config, _ = mse_setup
        ds = create_eval_dataset(config)
        assert ds.sampler is None

    def test_is_not_train(self, mse_setup):
        config, _ = mse_setup
        ds = create_eval_dataset(config)
        assert ds.is_train is False


# ---------------------------------------------------------------------------
# load_bed_intervals
# ---------------------------------------------------------------------------


class TestLoadBedIntervals:
    def test_loads_bed(self, mse_setup, tmp_path):
        config, _ = mse_setup
        bed = tmp_path / "peaks.bed"
        bed.write_text("chr1\t500\t600\nchr1\t800\t900\n")

        intervals = load_bed_intervals(config, bed)
        assert len(intervals) == 2
        for iv in intervals:
            assert isinstance(iv, Interval)
            assert len(iv) == config.data_config.input_len

    def test_str_path(self, mse_setup, tmp_path):
        """Accepts string paths, not just Path objects."""
        config, _ = mse_setup
        bed = tmp_path / "peaks.bed"
        bed.write_text("chr1\t500\t600\n")

        intervals = load_bed_intervals(config, str(bed))
        assert len(intervals) == 1

    def test_empty_bed_returns_empty(self, mse_setup, tmp_path):
        config, _ = mse_setup
        bed = tmp_path / "empty.bed"
        bed.write_text("")

        intervals = load_bed_intervals(config, bed)
        assert intervals == []


# ---------------------------------------------------------------------------
# predict_log_counts
# ---------------------------------------------------------------------------


class TestPredictLogCounts:
    def test_returns_correct_count(self, mse_setup):
        """DummyBPNetModel: log_counts = log(101). MSE loss with pseudocount=1."""
        config, ensemble = mse_setup
        ds = create_eval_dataset(config)
        intervals = [Interval("chr1", 500, 600)]

        results = predict_log_counts(ensemble, ds, intervals)
        assert len(results) == 1
        assert abs(results[0] - math.log(101.0)) < 0.01

    def test_multiple_intervals(self, mse_setup):
        config, ensemble = mse_setup
        ds = create_eval_dataset(config)
        intervals = [Interval("chr1", 500, 600), Interval("chr1", 700, 800)]

        results = predict_log_counts(ensemble, ds, intervals)
        assert len(results) == 2

    def test_empty_intervals(self, mse_setup):
        config, ensemble = mse_setup
        ds = create_eval_dataset(config)
        results = predict_log_counts(ensemble, ds, [])
        assert results == []

    def test_poisson_no_pseudocount(self, poisson_setup):
        """With Poisson loss, pseudocount=0 so log_counts pass through unchanged."""
        config, ensemble = poisson_setup
        ds = create_eval_dataset(config)
        intervals = [Interval("chr1", 500, 600)]

        results = predict_log_counts(ensemble, ds, intervals)
        assert len(results) == 1
        # Single channel, so log_counts is just flattened (no multi-channel aggregation)
        assert abs(results[0] - math.log(101.0)) < 0.01


# ---------------------------------------------------------------------------
# observed_log_counts
# ---------------------------------------------------------------------------


class _OnesTargetExtractor:
    """Returns a (1, L) tensor of ones for any interval — sum equals L."""

    def extract(self, interval: Interval) -> torch.Tensor:
        length = interval.end - interval.start
        return torch.ones((1, length), dtype=torch.float32)


def _make_dataset_with_target(config: CerberusConfig) -> CerberusDataset:
    """Build a dataset with an injected ones-returning target extractor.

    ``create_eval_dataset`` cannot inject a custom extractor, so we construct
    the dataset directly to keep the tests free of real BigWig fixtures.
    """
    return CerberusDataset(
        genome_config=config.genome_config,
        data_config=config.data_config,
        sampler_config=None,
        sequence_extractor=None,
        target_signal_extractor=_OnesTargetExtractor(),
        sampler=None,
        exclude_intervals={},
        is_train=False,
    )


class TestObservedLogCounts:
    def test_mse_pseudocount_applied(self, mse_setup):
        """MSE loss: obs = log(sum + pseudocount) over output_len-cropped window.

        ones extractor → sum = output_len = 50; pseudocount = 1.0;
        target_scale = 1.0 → expect log(51).
        """
        config, _ = mse_setup
        ds = _make_dataset_with_target(config)
        intervals = [Interval("chr1", 500, 600)]  # input_len-sized

        results = observed_log_counts(ds, intervals, config)
        assert len(results) == 1
        assert abs(results[0] - math.log(51.0)) < 1e-5

    def test_poisson_no_pseudocount(self, poisson_setup):
        """Poisson loss: obs = log(sum) over output_len-cropped window.

        ones extractor → sum = 50; no pseudocount → expect log(50).
        """
        config, _ = poisson_setup
        ds = _make_dataset_with_target(config)
        intervals = [Interval("chr1", 500, 600)]

        results = observed_log_counts(ds, intervals, config)
        assert len(results) == 1
        assert abs(results[0] - math.log(50.0)) < 1e-5

    def test_crops_to_output_len(self, mse_setup):
        """Verify the function crops to output_len, not input_len.

        If it summed over input_len (100) instead of output_len (50), the
        result would be log(101), not log(51).
        """
        config, _ = mse_setup
        ds = _make_dataset_with_target(config)
        intervals = [Interval("chr1", 500, 600)]

        result = observed_log_counts(ds, intervals, config)[0]
        assert abs(result - math.log(51.0)) < 1e-5
        # sanity: definitely not summing over input_len
        assert abs(result - math.log(101.0)) > 0.5

    def test_target_scale_applied(self, mse_setup):
        """target_scale multiplies raw counts before log."""
        config, _ = mse_setup
        scaled_data = config.data_config.model_copy(update={"target_scale": 2.0})
        scaled_config = config.model_copy(update={"data_config": scaled_data})
        ds = _make_dataset_with_target(scaled_config)
        intervals = [Interval("chr1", 500, 600)]

        result = observed_log_counts(ds, intervals, scaled_config)[0]
        # sum=50, scale=2 → 100, plus pseudocount 1 → log(101)
        assert abs(result - math.log(101.0)) < 1e-5

    def test_multiple_intervals_and_batching(self, mse_setup):
        """Returns one float per interval, regardless of batch_size."""
        config, _ = mse_setup
        ds = _make_dataset_with_target(config)
        intervals = [Interval("chr1", 500, 600) for _ in range(7)]

        results = observed_log_counts(ds, intervals, config, batch_size=3)
        assert len(results) == 7
        for r in results:
            assert abs(r - math.log(51.0)) < 1e-5

    def test_empty_intervals(self, mse_setup):
        config, _ = mse_setup
        ds = _make_dataset_with_target(config)
        assert observed_log_counts(ds, [], config) == []

    def test_raises_without_target_extractor(self, mse_setup):
        """Dataset without target_signal_extractor → RuntimeError."""
        config, _ = mse_setup
        ds = create_eval_dataset(config)  # config.targets is empty
        assert ds.target_signal_extractor is None

        with pytest.raises(RuntimeError, match="target_signal_extractor"):
            observed_log_counts(ds, [Interval("chr1", 500, 600)], config)

    def test_mirrors_predict_log_counts_when_obs_matches_pred(self, mse_setup):
        """Sanity: when ground-truth happens to equal model prediction, both
        helpers return the same value (in the same log-space).

        DummyBPNetModel emits log_counts = log(101). Set target_scale and the
        ones extractor so observed total + pseudocount = 101 → log(101).
        """
        config, ensemble = mse_setup
        # output_len=50, pseudocount=1: need scale*50 = 100 → scale=2.0
        scaled_data = config.data_config.model_copy(update={"target_scale": 2.0})
        scaled_config = config.model_copy(update={"data_config": scaled_data})
        ds = _make_dataset_with_target(scaled_config)
        intervals = [Interval("chr1", 500, 600), Interval("chr1", 700, 800)]

        pred = predict_log_counts(ensemble, ds, intervals)
        obs = observed_log_counts(ds, intervals, scaled_config)

        for p, o in zip(pred, obs, strict=True):
            assert abs(p - o) < 1e-4
