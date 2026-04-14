from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ModelOutput, aggregate_models


@dataclass
class MockOutput(ModelOutput):
    logits: torch.Tensor

    def detach(self):
        return self


class MockModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor):
        return MockOutput(logits=torch.ones_like(x) * self.value)


# Helper to create a mock fold map
def create_mock_folds(intervals_per_fold):
    folds = []
    for intervals in intervals_per_fold:
        fold_map = {}
        by_chrom = {}
        for chrom, start, end in intervals:
            if chrom not in by_chrom:
                by_chrom[chrom] = []
            by_chrom[chrom].append((start, end))

        for chrom, ints in by_chrom.items():
            tree_mock = MagicMock()

            def make_find(ints=ints):
                def find(query_range):
                    q_start, q_end = query_range
                    matches = []
                    for s, e in ints:
                        if s < q_end and e > q_start:
                            matches.append((s, e))
                    return matches

                return find

            tree_mock.find.side_effect = make_find(ints)
            fold_map[chrom] = tree_mock

        folds.append(fold_map)
    return folds


def _make_cerberus_config(output_len=100, output_bin_size=1):
    """Create a minimal CerberusConfig for ensemble tests."""
    return CerberusConfig.model_construct(
        data_config=DataConfig.model_construct(
            inputs={},
            targets={},
            input_len=100,
            output_len=output_len,
            output_bin_size=output_bin_size,
            max_jitter=0,
            encoding="ACGT",
            log_transform=False,
            reverse_complement=False,
            target_scale=1.0,
            use_sequence=True,
        ),
        genome_config=GenomeConfig.model_construct(
            name="mock",
            fasta_path="mock.fa",
            chrom_sizes={"chr1": 1000000},
            allowed_chroms=["chr1"],
            exclude_intervals={},
            fold_type="chrom_partition",
            fold_args={"k": 5, "test_fold": None, "val_fold": None},
        ),
        sampler_config=SamplerConfig.model_construct(
            sampler_type="random",
            padded_size=100,
            sampler_args={"num_intervals": 10},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1,
            max_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            patience=5,
            optimizer="adam",
            scheduler_type="default",
            scheduler_args={},
            filter_bias_and_bn=True,
            reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8,
            gradient_clip_val=None,
        ),
        model_config_=ModelConfig.model_construct(
            name="mock",
            model_cls="torch.nn.Linear",
            loss_cls="torch.nn.MSELoss",
            loss_args={},
            metrics_cls="torchmetrics.MeanSquaredError",
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
        ),
    )


def create_ensemble(models, folds, output_len=100, output_bin_size=1):
    cerberus_config = _make_cerberus_config(
        output_len=output_len, output_bin_size=output_bin_size
    )
    with (
        patch("cerberus.model_ensemble._ModelManager") as mock_cls,
        patch(
            "cerberus.model_ensemble.find_latest_hparams",
            return_value=Path("hparams.yaml"),
        ),
        patch(
            "cerberus.model_ensemble.parse_hparams_config", return_value=cerberus_config
        ),
    ):
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, folds)

        return ModelEnsemble(checkpoint_path=".", device=torch.device("cpu"))


def test_initialization():
    models = {"0": MockModel(1.0), "1": MockModel(2.0)}
    folds = []
    ensemble = create_ensemble(models, folds)

    assert len(ensemble) == 2
    assert "0" in ensemble
    assert "1" in ensemble
    assert ensemble.cerberus_config is not None
    assert ensemble.cerberus_config.data_config.output_len == 100


def test_forward_no_folds():
    models = {"0": MockModel(1.0), "1": MockModel(2.0)}
    ensemble = create_ensemble(models, folds=[], output_len=100, output_bin_size=1)

    x = torch.zeros(1, 1, 2)
    output = ensemble(x, intervals=None)

    assert cast(MockOutput, output).logits[0, 0, 0].item() == 1.5


def test_forward_selection():
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])

    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds, output_len=100, output_bin_size=1)

    x = torch.zeros(1, 1, 2)

    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    assert cast(MockOutput, out).logits[0, 0, 0].item() == 0.0

    interval = Interval("chr1", 100, 150)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    assert cast(MockOutput, out).logits[0, 0, 0].item() == 1.0

    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["val"])
    assert cast(MockOutput, out).logits[0, 0, 0].item() == 1.0


def test_get_partitions_for_interval():
    # Fold 0 covers chr1:0-100, fold 1 covers chr1:100-200
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    # Interval in fold 0's region
    assert ensemble._get_partitions_for_interval(Interval("chr1", 0, 50)) == {0}
    # Interval in fold 1's region
    assert ensemble._get_partitions_for_interval(Interval("chr1", 100, 150)) == {1}
    # Unknown chromosome → empty
    assert ensemble._get_partitions_for_interval(Interval("chrX", 0, 50)) == set()


def test_get_partitions_for_interval_no_folds():
    ensemble = create_ensemble({"0": MockModel(1.0)}, folds=[])
    assert ensemble._get_partitions_for_interval(Interval("chr1", 0, 50)) == set()


def test_partitions_to_model_indices_test():
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    # Partition 0, use_folds=["test"] → model 0
    assert ensemble._partitions_to_model_indices({0}, ["test"]) == {0}
    # Partition 1, use_folds=["test"] → model 1
    assert ensemble._partitions_to_model_indices({1}, ["test"]) == {1}


def test_partitions_to_model_indices_val():
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    # Partition 0, use_folds=["val"] → model (0-1)%2 = 1
    assert ensemble._partitions_to_model_indices({0}, ["val"]) == {1}
    # Partition 1, use_folds=["val"] → model (1-1)%2 = 0
    assert ensemble._partitions_to_model_indices({1}, ["val"]) == {0}


def test_partitions_to_model_indices_empty():
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    ensemble = create_ensemble({"0": MockModel(0.0), "1": MockModel(1.0)}, folds)

    # Empty partitions → empty model indices
    assert ensemble._partitions_to_model_indices(set(), ["test"]) == set()
    # No folds → empty regardless
    ensemble_no_folds = create_ensemble({"0": MockModel(1.0)}, folds=[])
    assert ensemble_no_folds._partitions_to_model_indices({0}, ["test"]) == set()


def test_forward_heterogeneous_batch():
    """Batch with samples from different fold partitions routes each correctly."""
    # Fold 0: chr1:0-100, Fold 1: chr1:100-200
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    # Model 0 returns 0.0, Model 1 returns 1.0
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)  # 2 samples
    intervals = [
        Interval("chr1", 0, 50),    # → partition 0 → model 0 (test) → value 0.0
        Interval("chr1", 100, 150),  # → partition 1 → model 1 (test) → value 1.0
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    # Sample 0 should get model 0's output (0.0)
    assert logits[0, 0, 0].item() == pytest.approx(0.0)
    # Sample 1 should get model 1's output (1.0)
    assert logits[1, 0, 0].item() == pytest.approx(1.0)


def test_forward_heterogeneous_batch_val_folds():
    """Heterogeneous batch with use_folds=['val'] routes correctly."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0, val → model (0-1)%2=1 → value 1.0
        Interval("chr1", 100, 150),  # partition 1, val → model (1-1)%2=0 → value 0.0
    ]

    out = ensemble(x, intervals=intervals, use_folds=["val"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(1.0)
    assert logits[1, 0, 0].item() == pytest.approx(0.0)


def test_forward_heterogeneous_test_and_val():
    """With use_folds=['test', 'val'], each sample is seen by 2 models and averaged."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0: test=model0(0.0), val=model1(1.0) → mean=0.5
        Interval("chr1", 100, 150),  # partition 1: test=model1(1.0), val=model0(0.0) → mean=0.5
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test", "val"])
    logits = cast(MockOutput, out).logits

    # Both samples see both models → average
    assert logits[0, 0, 0].item() == pytest.approx(0.5)
    assert logits[1, 0, 0].item() == pytest.approx(0.5)


def test_forward_homogeneous_batch_unchanged():
    """Homogeneous batch (all same partition) behaves same as before."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)
    # Both samples in partition 0
    intervals = [Interval("chr1", 0, 50), Interval("chr1", 10, 60)]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    # Both → model 0 → 0.0
    assert logits[0, 0, 0].item() == pytest.approx(0.0)
    assert logits[1, 0, 0].item() == pytest.approx(0.0)


def test_forward_unknown_chrom_falls_back():
    """Sample with unknown chrom falls back to all models."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(1, 1, 2)
    intervals = [Interval("chrX", 0, 50)]  # unknown chrom

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    # Falls back to all models averaged: (0.0 + 1.0) / 2 = 0.5
    assert logits[0, 0, 0].item() == pytest.approx(0.5)


def test_forward_heterogeneous_3_folds():
    """Three-fold model with samples spanning all three partitions."""
    folds = create_mock_folds([
        [("chr1", 0, 100)],
        [("chr1", 100, 200)],
        [("chr1", 200, 300)],
    ])
    models = {"0": MockModel(10.0), "1": MockModel(20.0), "2": MockModel(30.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(3, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0 → test=model0(10)
        Interval("chr1", 100, 150),  # partition 1 → test=model1(20)
        Interval("chr1", 200, 250),  # partition 2 → test=model2(30)
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(10.0)
    assert logits[1, 0, 0].item() == pytest.approx(20.0)
    assert logits[2, 0, 0].item() == pytest.approx(30.0)


def test_forward_heterogeneous_3_folds_val():
    """Three-fold model, val folds: rotation (p-1)%3."""
    folds = create_mock_folds([
        [("chr1", 0, 100)],
        [("chr1", 100, 200)],
        [("chr1", 200, 300)],
    ])
    models = {"0": MockModel(10.0), "1": MockModel(20.0), "2": MockModel(30.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(3, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0, val → model (0-1)%3=2 → 30
        Interval("chr1", 100, 150),  # partition 1, val → model (1-1)%3=0 → 10
        Interval("chr1", 200, 250),  # partition 2, val → model (2-1)%3=1 → 20
    ]

    out = ensemble(x, intervals=intervals, use_folds=["val"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(30.0)
    assert logits[1, 0, 0].item() == pytest.approx(10.0)
    assert logits[2, 0, 0].item() == pytest.approx(20.0)


def test_forward_heterogeneous_mixed_known_unknown_chroms():
    """Batch mixing known and unknown chromosomes."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(3, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0 → test=model0(0.0)
        Interval("chrX", 0, 50),     # unknown → all models → mean(0, 1)=0.5
        Interval("chr1", 100, 150),  # partition 1 → test=model1(1.0)
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(0.0)
    assert logits[1, 0, 0].item() == pytest.approx(0.5)
    assert logits[2, 0, 0].item() == pytest.approx(1.0)


def test_forward_heterogeneous_partial_overlap():
    """Two samples in same partition, one in different — tests sub-batching."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(4, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0 → model 0 (0.0)
        Interval("chr1", 10, 60),   # partition 0 → model 0 (0.0)
        Interval("chr1", 100, 150),  # partition 1 → model 1 (1.0)
        Interval("chr1", 0, 50),    # partition 0 → model 0 (0.0)
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(0.0)
    assert logits[1, 0, 0].item() == pytest.approx(0.0)
    assert logits[2, 0, 0].item() == pytest.approx(1.0)
    assert logits[3, 0, 0].item() == pytest.approx(0.0)


def test_aggregate_models_masked():
    """aggregate_models with masks averages only contributing models per sample."""
    # Model 0 output: all 10.0, Model 1 output: all 20.0
    out0 = MockOutput(logits=torch.tensor([[10.0, 10.0], [10.0, 10.0]]))
    out1 = MockOutput(logits=torch.tensor([[20.0, 20.0], [20.0, 20.0]]))

    # Model 0 sees sample 0 only, Model 1 sees sample 1 only
    # Zero-fill for non-contributing samples
    out0_masked = MockOutput(logits=torch.tensor([[10.0, 10.0], [0.0, 0.0]]))
    out1_masked = MockOutput(logits=torch.tensor([[0.0, 0.0], [20.0, 20.0]]))
    masks = [
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    ]

    agg = aggregate_models([out0_masked, out1_masked], method="mean", masks=masks)
    result = cast(MockOutput, agg).logits

    # Sample 0: only model 0 → 10.0; Sample 1: only model 1 → 20.0
    assert result[0, 0].item() == pytest.approx(10.0)
    assert result[1, 0].item() == pytest.approx(20.0)


def test_aggregate_models_masked_both_contribute():
    """When both models contribute to a sample, average is normal mean."""
    out0 = MockOutput(logits=torch.tensor([[10.0], [10.0]]))
    out1 = MockOutput(logits=torch.tensor([[20.0], [20.0]]))
    masks = [
        torch.tensor([True, True]),
        torch.tensor([True, True]),
    ]

    agg = aggregate_models([out0, out1], method="mean", masks=masks)
    result = cast(MockOutput, agg).logits

    assert result[0, 0].item() == pytest.approx(15.0)
    assert result[1, 0].item() == pytest.approx(15.0)


def test_aggregate_models_no_mask_unchanged():
    """masks=None gives identical result to original behavior."""
    out0 = MockOutput(logits=torch.tensor([[10.0], [10.0]]))
    out1 = MockOutput(logits=torch.tensor([[20.0], [20.0]]))

    agg_none = aggregate_models([out0, out1], method="mean", masks=None)
    agg_default = aggregate_models([out0, out1], method="mean")

    torch.testing.assert_close(
        cast(MockOutput, agg_none).logits,
        cast(MockOutput, agg_default).logits,
    )


def test_aggregate_models_masked_rejects_median():
    """Masked aggregation only supports mean, not median."""
    out0 = MockOutput(logits=torch.tensor([[10.0]]))
    masks = [torch.tensor([True])]

    with pytest.raises(ValueError, match="Masked aggregation only supports 'mean'"):
        aggregate_models([out0], method="median", masks=masks)


def test_forward_single_sample_batch():
    """Single-sample batch still works correctly with routing."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(1, 1, 2)
    intervals = [Interval("chr1", 100, 150)]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    assert logits[0, 0, 0].item() == pytest.approx(1.0)


def test_forward_no_intervals_all_models():
    """intervals=None still runs all models and averages."""
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)
    out = ensemble(x, intervals=None)
    logits = cast(MockOutput, out).logits

    # All models run: mean(0, 1) = 0.5
    assert logits[0, 0, 0].item() == pytest.approx(0.5)
    assert logits[1, 0, 0].item() == pytest.approx(0.5)


def test_zero_fill_does_not_leak_through_masked_aggregation():
    """Non-contributing model's zero-filled output must not affect the result.

    The scatter step fills non-participating sample positions with zeros.
    This test verifies the coupling: aggregate_models() with masks zeroes
    out non-contributing entries before summing, so the fill value (zero)
    never reaches the final output — even when zero would be a meaningful
    prediction (e.g. classification logits where zero = uniform prior).
    """
    folds = create_mock_folds([[("chr1", 0, 100)], [("chr1", 100, 200)]])
    # Model 0 returns 42.0, Model 1 returns 99.0
    models = {"0": MockModel(42.0), "1": MockModel(99.0)}
    ensemble = create_ensemble(models, folds)

    x = torch.zeros(2, 1, 2)
    intervals = [
        Interval("chr1", 0, 50),    # partition 0 → test=model0 only
        Interval("chr1", 100, 150),  # partition 1 → test=model1 only
    ]

    out = ensemble(x, intervals=intervals, use_folds=["test"])
    logits = cast(MockOutput, out).logits

    # Sample 0 must be exactly model 0's value — no contamination from
    # model 1's zero-filled placeholder at this position.
    assert logits[0, 0, 0].item() == pytest.approx(42.0)
    # Sample 1 must be exactly model 1's value — no contamination from
    # model 0's zero-filled placeholder at this position.
    assert logits[1, 0, 0].item() == pytest.approx(99.0)

    # Verify directly that the zero fill doesn't bleed through by
    # checking the internal _forward_models output before aggregation.
    batch_outputs, masks = ensemble._forward_models(
        x, intervals=intervals, use_folds=["test"]
    )
    assert masks is not None
    assert len(batch_outputs) == 2

    # Model 0's output: sample 0 has real data (42), sample 1 is zero-filled
    m0_logits = cast(MockOutput, batch_outputs[0]).logits
    assert m0_logits[0, 0, 0].item() == pytest.approx(42.0)
    assert m0_logits[1, 0, 0].item() == pytest.approx(0.0)  # zero fill
    assert masks[0].tolist() == [True, False]

    # Model 1's output: sample 0 is zero-filled, sample 1 has real data (99)
    m1_logits = cast(MockOutput, batch_outputs[1]).logits
    assert m1_logits[0, 0, 0].item() == pytest.approx(0.0)  # zero fill
    assert m1_logits[1, 0, 0].item() == pytest.approx(99.0)
    assert masks[1].tolist() == [False, True]


def test_aggregate():
    out1 = MockOutput(logits=torch.ones(2, 2))
    out2 = MockOutput(logits=torch.ones(2, 2) * 3)

    agg = aggregate_models([out1, out2], method="mean")
    assert torch.allclose(cast(MockOutput, agg).logits, torch.ones(2, 2) * 2)


def test_forward_interval_aggregation():
    input_len = 100
    output_len = 20
    output_bin_size = 1

    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len

        def forward(self, x):
            batch_size = x.shape[0]
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(
        models, folds, output_len=output_len, output_bin_size=output_bin_size
    )

    intervals = [Interval("chr1", 0, 100), Interval("chr1", 10, 110)]

    x = torch.zeros(2, 4, input_len)

    output = ensemble(x, intervals=intervals, aggregation="interval+model")

    logits = cast(MockOutput, output).logits
    merged_interval = output.out_interval

    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 40
    assert merged_interval.end == 70
    assert logits.shape[-1] == 30


def test_predict_intervals_method():
    input_len = 100
    output_len = 20
    output_bin_size = 1

    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len

        def forward(self, x):
            batch_size = x.shape[0]
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(
        models, folds, output_len=output_len, output_bin_size=output_bin_size
    )

    dataset = MagicMock()
    dataset.data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        max_jitter=0,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    intervals = [Interval("chr1", 0, 100), Interval("chr1", 10, 110)]

    output = ensemble.predict_intervals(
        intervals, dataset, use_folds=["test"], aggregation="model", batch_size=2
    )

    merged_interval = output.out_interval
    assert merged_interval is not None
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 40
    assert merged_interval.end == 70
    assert cast(MockOutput, output).logits.shape[-1] == 30


def test_predict_output_intervals_method():
    input_len = 100
    output_len = 20
    output_bin_size = 1

    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len

        def forward(self, x):
            batch_size = x.shape[0]
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(
        models, folds, output_len=output_len, output_bin_size=output_bin_size
    )

    dataset = MagicMock()
    dataset.data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        max_jitter=0,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    intervals = [Interval("chr1", 200, 300)]

    outputs = ensemble.predict_output_intervals(
        intervals,
        dataset,
        stride=50,
        use_folds=["test"],
        aggregation="model",
        batch_size=2,
    )

    assert len(outputs) == 1
    out = outputs[0]
    merged_interval = out.out_interval
    assert merged_interval is not None

    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 200
    assert merged_interval.end == 270
    assert cast(MockOutput, out).logits.shape[-1] == 70
