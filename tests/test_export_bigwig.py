"""
Tests for tools/export_bigwig.py CLI and the predict_to_bigwig integration.

Covers:
  1. Basic integration (file creation).
  2. Linear count reconstruction for ProfileCountOutput and ProfileLogRates.
  3. target_scale and output_bin_size normalization.
"""
import math
import pytest
import torch
import torch.nn as nn
import numpy as np
import yaml
import pybigtools
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch

from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config
from cerberus.config import (
    CerberusConfig, GenomeConfig, DataConfig, ModelConfig,
    TrainConfig, SamplerConfig,
)
from cerberus.output import ModelOutput, ProfileCountOutput, ProfileLogRates
from cerberus.predict_bigwig import predict_to_bigwig, _process_island


# ---------------------------------------------------------------------------
# Dummy models returning different ModelOutput types
# ---------------------------------------------------------------------------


@dataclass
class DummyLogitsOnly(ModelOutput):
    """Plain logits — no log_counts, no log_rates."""
    logits: torch.Tensor

    def detach(self):
        return DummyLogitsOnly(logits=self.logits.detach())


class DummyModelLogitsOnly(nn.Module):
    """Returns DummyLogitsOnly (fallback path in _process_island)."""
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        return DummyLogitsOnly(
            logits=torch.ones((x.shape[0], 1, self.output_dim), device=x.device)
        )


class BPNetLikeModel(nn.Module):
    """Returns ProfileCountOutput with known logits and log_counts."""
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        B = x.shape[0]
        # Uniform logits → softmax = 1/output_dim per bin
        logits = torch.zeros((B, 1, self.output_dim), device=x.device)
        # log_counts = log(100) → total predicted counts = 100
        log_counts = torch.full((B, 1), math.log(100.0), device=x.device)
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


class LogRatesModel(nn.Module):
    """Returns ProfileLogRates with known log_rates."""
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        B = x.shape[0]
        # log_rates = log(2.0) → exp gives 2.0 counts per bin
        log_rates = torch.full(
            (B, 1, self.output_dim), math.log(2.0), device=x.device
        )
        return ProfileLogRates(log_rates=log_rates)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_setup(
    tmp_path,
    model_cls,
    model_cls_path,
    target_scale=1.0,
    output_bin_size=1,
    loss_cls="cerberus.loss.PoissonMultinomialLoss",
    loss_args=None,
):
    """Create genome + dataset + ensemble with a given model class."""
    genome = tmp_path / "genome.fa"
    with open(genome, "w") as f:
        f.write(">chr1\n" + "A" * 2000 + "\n")
    with open(tmp_path / "genome.fa.fai", "w") as f:
        f.write("chr1\t2000\t6\t2000\t2001\n")

    genome_config = create_genome_config(
        name="test",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={},
    )

    if loss_args is None:
        loss_args = {"count_pseudocount": 0.0}

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=50,
        output_bin_size=output_bin_size,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=target_scale,
        use_sequence=True,
    )

    dataset = CerberusDataset(genome_config, data_config, sampler_config=None)

    model = model_cls(input_len=100, output_len=50, output_bin_size=output_bin_size)

    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    torch.save(model.state_dict(), fold_dir / "model.pt")

    with open(tmp_path / "ensemble_metadata.yaml", "w") as f:
        yaml.dump({"folds": [0]}, f)

    model_config = ModelConfig.model_construct(
        name="dummy",
        model_cls=model_cls_path,
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
        batch_size=1, max_epochs=1, learning_rate=1e-3,
        weight_decay=0.0, patience=1, optimizer="adam",
        scheduler_type="constant", scheduler_args={},
        filter_bias_and_bn=False,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8, gradient_clip_val=None,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval", padded_size=100, sampler_args={},
    )

    mock_config = CerberusConfig.model_construct(
        genome_config=genome_config,
        data_config=data_config,
        model_config_=model_config,
        sampler_config=sampler_config,
        train_config=train_config,
    )

    with patch(
        "cerberus.model_ensemble.ModelEnsemble._find_hparams",
        return_value=Path("hparams.yaml"),
    ), patch("cerberus.model_ensemble.parse_hparams_config", return_value=mock_config):
        ensemble = ModelEnsemble(
            tmp_path, model_config, data_config, genome_config, torch.device("cpu")
        )

    return dataset, ensemble, tmp_path


@pytest.fixture
def bigwig_setup(tmp_path):
    """Minimal setup with DummyModelLogitsOnly (fallback path)."""
    return _make_setup(
        tmp_path,
        DummyModelLogitsOnly,
        "tests.test_export_bigwig.DummyModelLogitsOnly",
    )


@pytest.fixture
def bpnet_setup(tmp_path):
    """Setup with BPNetLikeModel returning ProfileCountOutput."""
    return _make_setup(
        tmp_path, BPNetLikeModel, "tests.test_export_bigwig.BPNetLikeModel"
    )


@pytest.fixture
def logrates_setup(tmp_path):
    """Setup with LogRatesModel returning ProfileLogRates."""
    return _make_setup(
        tmp_path, LogRatesModel, "tests.test_export_bigwig.LogRatesModel"
    )


def _read_bigwig_values(path):
    """Read all values from a bigwig file as a list of (chrom, start, end, value)."""
    bw = pybigtools.open(str(path))  # type: ignore[attr-defined]
    results = []
    for chrom, length in bw.chroms().items():
        for start, end, value in bw.records(chrom, 0, length):
            results.append((chrom, int(start), int(end), float(value)))
    return results


# ---------------------------------------------------------------------------
# 1. Basic integration tests
# ---------------------------------------------------------------------------


def test_predict_to_bigwig_creates_file(bigwig_setup):
    """predict_to_bigwig produces a non-empty BigWig file."""
    dataset, ensemble, tmp_path = bigwig_setup
    output_path = tmp_path / "test_output.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_predict_to_bigwig_custom_stride(bigwig_setup):
    """predict_to_bigwig accepts a custom stride without errors."""
    dataset, ensemble, tmp_path = bigwig_setup
    output_path = tmp_path / "test_stride.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        stride=10,
        use_folds=["test", "val"],
        batch_size=4,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_predict_to_bigwig_all_folds(bigwig_setup):
    """predict_to_bigwig with all folds produces output."""
    dataset, ensemble, tmp_path = bigwig_setup
    output_path = tmp_path / "test_all_folds.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["train", "test", "val"],
        batch_size=4,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. ProfileCountOutput reconstruction (BPNet/Dalmatian path)
# ---------------------------------------------------------------------------


def test_bpnet_bigwig_values_are_positive(bpnet_setup):
    """ProfileCountOutput bigwig: all exported values are positive finite floats.

    Exact values depend on sliding window overlap/averaging, but all must be
    positive (softmax * exp(log_counts) is always > 0).
    """
    dataset, ensemble, tmp_path = bpnet_setup
    output_path = tmp_path / "bpnet.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
    )

    entries = _read_bigwig_values(output_path)
    assert len(entries) > 0

    values = [e[3] for e in entries]
    for v in values:
        assert np.isfinite(v), f"Non-finite value: {v}"
        assert v > 0, f"Expected positive, got {v}"


def test_bpnet_target_scale_halves_values(tmp_path):
    """target_scale=2.0 halves exported values compared to target_scale=1.0."""
    # Scale=1
    dir1 = tmp_path / "s1"
    dir1.mkdir()
    ds1, ens1, _ = _make_setup(
        dir1, BPNetLikeModel, "tests.test_export_bigwig.BPNetLikeModel",
        target_scale=1.0,
    )
    out1 = dir1 / "out.bw"
    predict_to_bigwig(out1, ds1, ens1, use_folds=["test", "val"], batch_size=4)

    # Scale=2
    dir2 = tmp_path / "s2"
    dir2.mkdir()
    ds2, ens2, _ = _make_setup(
        dir2, BPNetLikeModel, "tests.test_export_bigwig.BPNetLikeModel",
        target_scale=2.0,
    )
    out2 = dir2 / "out.bw"
    predict_to_bigwig(out2, ds2, ens2, use_folds=["test", "val"], batch_size=4)

    vals1 = [e[3] for e in _read_bigwig_values(out1)]
    vals2 = [e[3] for e in _read_bigwig_values(out2)]

    # Same genome → same number of entries
    assert len(vals1) == len(vals2)
    # Each value in scale=2 should be half of scale=1
    for v1, v2 in zip(vals1, vals2):
        assert abs(v2 - v1 / 2.0) < 1e-5, f"scale=2 value {v2} != {v1}/2"


# ---------------------------------------------------------------------------
# 3. ProfileLogRates reconstruction (ASAP/Gopher path)
# ---------------------------------------------------------------------------


def test_logrates_reconstruction_values(logrates_setup):
    """ProfileLogRates: exported values = exp(log_rates).

    LogRatesModel outputs log_rates = log(2.0), so each bin = 2.0.
    """
    dataset, ensemble, tmp_path = logrates_setup
    output_path = tmp_path / "logrates.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
    )

    entries = _read_bigwig_values(output_path)
    assert len(entries) > 0

    values = [e[3] for e in entries]
    for v in values:
        assert abs(v - 2.0) < 0.01, f"Expected ~2.0, got {v}"


def test_logrates_with_target_scale(tmp_path):
    """target_scale=4.0 divides log_rates values by 4."""
    dataset, ensemble, tmp_path = _make_setup(
        tmp_path,
        LogRatesModel,
        "tests.test_export_bigwig.LogRatesModel",
        target_scale=4.0,
    )
    output_path = tmp_path / "logrates_scaled.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
    )

    entries = _read_bigwig_values(output_path)
    values = [e[3] for e in entries]
    # exp(log(2)) / 4 = 0.5
    for v in values:
        assert abs(v - 0.5) < 0.01, f"Expected ~0.5, got {v}"


# ---------------------------------------------------------------------------
# 4. _process_island unit tests (numpy/torch type safety)
# ---------------------------------------------------------------------------


def test_process_island_profile_count_output_types(bpnet_setup):
    """_process_island correctly handles ProfileCountOutput with torch tensors
    internally (the np.asarray conversion must not fail)."""
    dataset, ensemble, _ = bpnet_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    assert len(results) > 0
    for chrom, start, end, val in results:
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert np.isfinite(val), f"Non-finite value: {val}"


def test_process_island_logrates_output_types(logrates_setup):
    """_process_island correctly handles ProfileLogRates with torch tensors."""
    dataset, ensemble, _ = logrates_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    assert len(results) > 0
    for chrom, start, end, val in results:
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert np.isfinite(val), f"Non-finite value: {val}"


def test_process_island_values_are_python_floats_not_numpy(bpnet_setup):
    """Regression: yield values must be Python float, not numpy scalar.

    pybigtools.write expects (str, int, int, float) tuples. A numpy 0-d
    array would pass isinstance(x, float) but could cause subtle issues.
    This test ensures float() conversion in _process_island works correctly.
    """
    dataset, ensemble, _ = bpnet_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    for _, _, _, val in results:
        assert type(val) is float, f"Expected exact float type, got {type(val)}"



def test_bpnet_log_counts_shape_after_overlapping_windows(bpnet_setup):
    """Regression: per-window reconstruction must happen before spatial merging.

    Bug: _process_island merged logits across all overlapping windows into one
    long array, then applied softmax over the entire merged length. Since softmax
    normalizes to sum=1, the per-bin values were divided by the merged length
    instead of the per-window output_len, producing predictions ~1000x too small.

    Fix: reconstruct linear signal (softmax(logits) * exp(log_counts)) per window,
    then spatially merge the linear signals. This preserves the correct per-bin
    scale because softmax normalizes over the correct output_len (50 bins).

    This test uses two overlapping intervals and verifies per-bin values are ~2.0
    (= exp(log(100)) / 50 bins), not ~0.001.
    """
    dataset, ensemble, _ = bpnet_setup
    # Two overlapping windows — forces spatial aggregation.
    # input_len=100, output_len=50, so output covers center 50bp of each window.
    # Window 1: [475, 575) → output [500, 550)
    # Window 2: [500, 600) → output [525, 575)
    # Overlap region: [525, 550) — 25 bins of overlap
    intervals = [Interval("chr1", 475, 575), Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    # Output should cover [500, 575) = 75 bins
    assert len(results) == 75, f"Expected 75 bins, got {len(results)}"

    # BPNetLikeModel: uniform logits (all 0), log_counts=log(100), output_dim=50.
    # Per-window reconstruction: softmax over 50 bins → 100/50 = 2.0 per bin.
    # Two windows with 25-bin overlap → 75 output bins, each ≈ 2.0.
    # In the overlap region, identical values are averaged → still 2.0.
    # The bug produced per-bin ≈ 0.001 (softmax over merged length).
    values = [val for _, _, _, val in results]
    for val in values:
        assert abs(val - 2.0) < 0.1, (
            f"Per-bin value should be ~2.0, got {val:.4f} "
            f"(scaling bug if ~0.001)"
        )


def test_process_island_bpnet_correct_values(bpnet_setup):
    """Verify _process_island computes softmax(logits)*exp(log_counts) correctly.

    BPNetLikeModel: uniform logits (all 0), log_counts=log(100), output_dim=50.
    Expected per-bin = 100/50 = 2.0.
    """
    dataset, ensemble, _ = bpnet_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    assert len(results) == 50  # output_len = 50, bin_size = 1
    for _, _, _, val in results:
        assert abs(val - 2.0) < 0.01, f"Expected ~2.0, got {val}"


def test_process_island_logrates_correct_values(logrates_setup):
    """Verify _process_island computes exp(log_rates) correctly.

    LogRatesModel: log_rates=log(2.0). Expected per-bin = 2.0.
    """
    dataset, ensemble, _ = logrates_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals,
            dataset,
            ensemble,
            use_folds=["test", "val"],
            batch_size=4,
            count_pseudocount=0.0,
        )
    )

    assert len(results) == 50
    for _, _, _, val in results:
        assert abs(val - 2.0) < 0.01, f"Expected ~2.0, got {val}"


# ---------------------------------------------------------------------------
# 4b. Pseudocount inversion regression tests
# ---------------------------------------------------------------------------


class MSEBPNetModel(nn.Module):
    """BPNet-like model where log_counts = log(count + pseudocount).

    Simulates MSE loss training with count_pseudocount=150.
    With true_count=100 and pseudocount=150, log_counts = log(250).
    output_dim bins, uniform logits → each bin = true_count / output_dim.
    """
    PSEUDOCOUNT = 150.0
    TRUE_COUNT = 100.0

    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        B = x.shape[0]
        logits = torch.zeros((B, 1, self.output_dim), device=x.device)
        # MSE loss trains model to predict log(count + pseudocount)
        log_counts = torch.full(
            (B, 1),
            math.log(self.TRUE_COUNT + self.PSEUDOCOUNT),
            device=x.device,
        )
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


def test_pseudocount_inversion_correct_values(tmp_path):
    """Regression: with pseudocount=150, reconstruction must subtract it.

    MSEBPNetModel predicts log_counts = log(100 + 150) = log(250).
    With count_pseudocount=150: total = exp(log(250)) - 150 = 100.
    Per-bin with output_dim=50: 100/50 = 2.0.
    Without pseudocount inversion: 250/50 = 5.0 (WRONG).
    """
    ds, ens, _ = _make_setup(
        tmp_path, MSEBPNetModel, "tests.test_export_bigwig.MSEBPNetModel",
    )
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals, ds, ens,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=150.0,
        )
    )

    assert len(results) == 50
    for _, _, _, val in results:
        assert abs(val - 2.0) < 0.01, (
            f"Expected ~2.0 with pseudocount inversion, got {val:.4f}"
        )


def test_pseudocount_zero_gives_inflated_values(tmp_path):
    """Without pseudocount inversion, MSE-trained model gives inflated values.

    MSEBPNetModel: log_counts = log(250), output_dim=50.
    With count_pseudocount=0: total = 250, per-bin = 250/50 = 5.0.
    """
    ds, ens, _ = _make_setup(
        tmp_path, MSEBPNetModel, "tests.test_export_bigwig.MSEBPNetModel",
    )
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals, ds, ens,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )

    assert len(results) == 50
    for _, _, _, val in results:
        assert abs(val - 5.0) < 0.01, (
            f"Expected ~5.0 without pseudocount inversion, got {val:.4f}"
        )


def test_pseudocount_inversion_bigwig_end_to_end(tmp_path):
    """End-to-end: BigWig values reflect pseudocount-corrected signal."""
    ds, ens, _ = _make_setup(
        tmp_path, MSEBPNetModel, "tests.test_export_bigwig.MSEBPNetModel",
        loss_cls="cerberus.loss.MSEMultinomialLoss",
        loss_args={"count_pseudocount": 150.0},
    )
    output_path = tmp_path / "pseudo.bw"

    predict_to_bigwig(
        output_path=output_path,
        dataset=ds,
        model_ensemble=ens,
        use_folds=["test", "val"],
        batch_size=4,
    )

    entries = _read_bigwig_values(output_path)
    values = [e[3] for e in entries]
    assert len(values) > 0
    for val in values:
        assert abs(val - 2.0) < 0.1, (
            f"Expected ~2.0 in BigWig with pseudocount=150, got {val:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. _reconstruct_linear_signal unit tests
# ---------------------------------------------------------------------------


from cerberus.predict_bigwig import _reconstruct_linear_signal


def test_reconstruct_profile_count_sum_equals_total_counts():
    """softmax(logits) * exp(log_counts) must sum to exp(log_counts) per channel."""
    log_total = 7.0  # exp(7) ≈ 1096.6
    output = ProfileCountOutput(
        logits=torch.randn(1, 100),   # non-uniform
        log_counts=torch.tensor([log_total]),
    )
    signal = _reconstruct_linear_signal(output)
    assert signal.shape == (1, 100)
    assert abs(signal.sum().item() - math.exp(log_total)) < 0.1


def test_reconstruct_profile_count_nonuniform_logits():
    """Non-uniform logits produce a peaked profile, not flat."""
    logits = torch.zeros(1, 50)
    logits[0, 25] = 10.0  # strong peak at center
    output = ProfileCountOutput(
        logits=logits,
        log_counts=torch.tensor([math.log(100.0)]),
    )
    signal = _reconstruct_linear_signal(output)
    # Peak should be much larger than flanks
    assert signal[0, 25].item() > signal[0, 0].item() * 10


def test_reconstruct_profile_count_multichannel():
    """Multi-channel ProfileCountOutput: each channel gets its own total."""
    output = ProfileCountOutput(
        logits=torch.zeros(2, 50),
        log_counts=torch.tensor([math.log(100.0), math.log(200.0)]),
    )
    signal = _reconstruct_linear_signal(output)
    assert signal.shape == (2, 50)
    assert abs(signal[0].sum().item() - 100.0) < 0.1
    assert abs(signal[1].sum().item() - 200.0) < 0.1


def test_reconstruct_logrates():
    """ProfileLogRates: signal = exp(log_rates)."""
    output = ProfileLogRates(
        log_rates=torch.full((1, 50), math.log(3.0)),
    )
    signal = _reconstruct_linear_signal(output)
    assert signal.shape == (1, 50)
    assert torch.allclose(signal, torch.full((1, 50), 3.0), atol=1e-5)


def test_reconstruct_logits_only_fallback():
    """ProfileLogits fallback: returns raw logits unchanged."""
    from cerberus.output import ProfileLogits

    logits = torch.randn(1, 50)
    output = ProfileLogits(logits=logits)
    signal = _reconstruct_linear_signal(output)
    assert torch.equal(signal, logits)


def test_reconstruct_unknown_output_raises():
    """Unknown output type raises ValueError."""
    @dataclass
    class UnknownOutput(ModelOutput):
        data: torch.Tensor
        def detach(self): return self

    output = UnknownOutput(data=torch.randn(1, 50))
    with pytest.raises(ValueError, match="Cannot extract profile track"):
        _reconstruct_linear_signal(output)


def test_reconstruct_numerically_stable_with_large_logits():
    """Softmax must be numerically stable even with very large logit values."""
    logits = torch.full((1, 50), 500.0)  # large, would overflow naive exp
    logits[0, 10] = 510.0  # peak
    output = ProfileCountOutput(
        logits=logits,
        log_counts=torch.tensor([math.log(100.0)]),
    )
    signal = _reconstruct_linear_signal(output)
    assert torch.all(torch.isfinite(signal))
    assert abs(signal.sum().item() - 100.0) < 0.1


# ---------------------------------------------------------------------------
# 6. _process_island: coordinate correctness
# ---------------------------------------------------------------------------


def test_process_island_coordinates_contiguous(bpnet_setup):
    """Output tuples must have contiguous, non-overlapping (chrom, start, end)."""
    dataset, ensemble, _ = bpnet_setup
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals, dataset, ensemble,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )

    for i in range(len(results) - 1):
        chrom_a, _, end_a, _ = results[i]
        chrom_b, start_b, _, _ = results[i + 1]
        assert chrom_a == chrom_b == "chr1"
        assert end_a == start_b, f"Gap at index {i}: end={end_a}, next_start={start_b}"

    # First start and last end should match output interval
    # input_len=100, output_len=50 → offset=25, output covers [525, 575)
    assert results[0][1] == 525
    assert results[-1][2] == 575


def test_process_island_many_overlapping_windows_correct_scale(bpnet_setup):
    """Many overlapping windows should still produce correct per-bin values.

    With uniform logits and stride=output_len//2, all windows predict 2.0 per bin.
    After averaging overlapping predictions, the result should still be 2.0.
    """
    dataset, ensemble, _ = bpnet_setup
    output_len = 50
    stride = output_len // 2  # 25

    # 6 overlapping windows covering [525, 700)
    intervals = [
        Interval("chr1", 500 + i * stride, 600 + i * stride)
        for i in range(6)
    ]

    results = list(
        _process_island(
            intervals, dataset, ensemble,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )

    values = [val for _, _, _, val in results]
    for val in values:
        assert abs(val - 2.0) < 0.1, (
            f"Per-bin should be ~2.0 regardless of overlap count, got {val:.4f}"
        )


def test_process_island_target_scale_applied(tmp_path):
    """target_scale divides the reconstructed signal."""
    ds, ens, _ = _make_setup(
        tmp_path, BPNetLikeModel, "tests.test_export_bigwig.BPNetLikeModel",
        target_scale=2.0,
    )
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals, ds, ens,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )

    # Without scale: 100/50 = 2.0. With target_scale=2.0: 2.0/2.0 = 1.0
    for _, _, _, val in results:
        assert abs(val - 1.0) < 0.01, f"Expected ~1.0 with scale=2, got {val}"


def test_process_island_empty_island(bpnet_setup):
    """Empty interval list yields no results."""
    dataset, ensemble, _ = bpnet_setup
    results = list(
        _process_island(
            [], dataset, ensemble,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )
    assert results == []


# ---------------------------------------------------------------------------
# 7. Region-restricted prediction
# ---------------------------------------------------------------------------


def test_predict_to_bigwig_single_region(bpnet_setup):
    """predict_to_bigwig with a single region produces output only in that region."""
    dataset, ensemble, tmp_path = bpnet_setup
    output_path = tmp_path / "region.bw"

    region = Interval("chr1", 500, 700, "+")
    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
        regions=[region],
    )

    assert output_path.exists()
    entries = _read_bigwig_values(output_path)
    assert len(entries) > 0

    # All entries should be within or near the requested region
    for chrom, start, end, val in entries:
        assert chrom == "chr1"
        assert start >= 400, f"Entry start {start} is too far before region"
        assert end <= 800, f"Entry end {end} is too far after region"
        assert val > 0


def test_predict_to_bigwig_region_smaller_than_genome_wide(bpnet_setup):
    """Region-restricted prediction produces fewer entries than genome-wide."""
    dataset, ensemble, tmp_path = bpnet_setup

    # Genome-wide
    gw_path = tmp_path / "gw.bw"
    predict_to_bigwig(
        gw_path, dataset, ensemble,
        use_folds=["test", "val"], batch_size=4,
    )

    # Single small region
    region_path = tmp_path / "region.bw"
    predict_to_bigwig(
        region_path, dataset, ensemble,
        use_folds=["test", "val"], batch_size=4,
        regions=[Interval("chr1", 800, 900, "+")],
    )

    gw_entries = _read_bigwig_values(gw_path)
    region_entries = _read_bigwig_values(region_path)
    assert len(region_entries) < len(gw_entries)


def test_predict_to_bigwig_multiple_regions(bpnet_setup):
    """Multiple regions produce output in each region."""
    dataset, ensemble, tmp_path = bpnet_setup
    output_path = tmp_path / "multi_region.bw"

    regions = [
        Interval("chr1", 500, 600, "+"),
        Interval("chr1", 1000, 1100, "+"),
    ]
    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        use_folds=["test", "val"],
        batch_size=4,
        regions=regions,
    )

    entries = _read_bigwig_values(output_path)
    assert len(entries) > 0

    # Should have entries near both regions
    starts = {e[1] for e in entries}
    has_first_region = any(400 <= s <= 700 for s in starts)
    has_second_region = any(900 <= s <= 1200 for s in starts)
    assert has_first_region, "Missing entries near first region"
    assert has_second_region, "Missing entries near second region"


def test_predict_to_bigwig_region_values_match_genome_wide(bpnet_setup):
    """Values in a region should match the genome-wide prediction at the same coordinates."""
    dataset, ensemble, tmp_path = bpnet_setup

    # BPNetLikeModel with uniform logits → all values should be ~2.0
    # regardless of whether we predict genome-wide or region-restricted
    region_path = tmp_path / "region_vals.bw"
    predict_to_bigwig(
        region_path, dataset, ensemble,
        use_folds=["test", "val"], batch_size=4,
        regions=[Interval("chr1", 600, 800, "+")],
    )

    entries = _read_bigwig_values(region_path)
    values = [e[3] for e in entries]
    for val in values:
        assert abs(val - 2.0) < 0.1, f"Region value should be ~2.0, got {val}"


class MultiChannelBPNetModel(nn.Module):
    """Returns ProfileCountOutput with 2 channels."""
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size

    def forward(self, x):
        B = x.shape[0]
        logits = torch.zeros((B, 2, self.output_dim), device=x.device)
        log_counts = torch.full((B, 2), math.log(100.0), device=x.device)
        # Make channel 1 have double the counts
        log_counts[:, 1] = math.log(200.0)
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


def test_process_island_multichannel_exports_channel_zero(tmp_path):
    """Multi-channel model: only channel 0 is exported to BigWig."""
    ds, ens, _ = _make_setup(
        tmp_path, MultiChannelBPNetModel,
        "tests.test_export_bigwig.MultiChannelBPNetModel",
    )
    intervals = [Interval("chr1", 500, 600)]

    results = list(
        _process_island(
            intervals, ds, ens,
            use_folds=["test", "val"], batch_size=4, count_pseudocount=0.0,
        )
    )

    # Channel 0: 100/50 = 2.0 (not channel 1's 200/50 = 4.0)
    for _, _, _, val in results:
        assert abs(val - 2.0) < 0.01, f"Expected ch0 value ~2.0, got {val}"
