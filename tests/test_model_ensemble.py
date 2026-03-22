import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from typing import cast
from pathlib import Path

from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
    GenomeConfig,
    SamplerConfig,
    CerberusConfig,
)
from cerberus.output import ModelOutput, aggregate_models
from cerberus.interval import Interval

@dataclass
class MockOutput(ModelOutput):
    logits: torch.Tensor
    def detach(self): return self

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
        for (chrom, start, end) in intervals:
            if chrom not in by_chrom: by_chrom[chrom] = []
            by_chrom[chrom].append((start, end))

        for chrom, ints in by_chrom.items():
            tree_mock = MagicMock()

            def make_find(ints=ints):
                def find(query_range):
                    q_start, q_end = query_range
                    matches = []
                    for (s, e) in ints:
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
            inputs={}, targets={},
            input_len=100, output_len=output_len, output_bin_size=output_bin_size,
            max_jitter=0, encoding="ACGT", log_transform=False,
            reverse_complement=False, target_scale=1.0, use_sequence=True,
        ),
        genome_config=GenomeConfig.model_construct(
            name="mock", fasta_path="mock.fa",
            chrom_sizes={"chr1": 1000000}, allowed_chroms=["chr1"],
            exclude_intervals={}, fold_type="chrom_partition",
            fold_args={"k": 5, "test_fold": None, "val_fold": None},
        ),
        sampler_config=SamplerConfig.model_construct(
            sampler_type="random", padded_size=100,
            sampler_args={"num_intervals": 10},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1, max_epochs=1, learning_rate=1e-3,
            weight_decay=0.0, patience=5, optimizer="adam",
            scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8, gradient_clip_val=None,
        ),
        model_config_=ModelConfig.model_construct(
            name="mock", model_cls="torch.nn.Linear",
            loss_cls="torch.nn.MSELoss", loss_args={},
            metrics_cls="torchmetrics.MeanSquaredError", metrics_args={},
            model_args={}, pretrained=[], count_pseudocount=0.0,
        ),
    )

def create_ensemble(models, folds, output_len=100, output_bin_size=1):
    cerberus_config = _make_cerberus_config(output_len=output_len, output_bin_size=output_bin_size)
    with patch("cerberus.model_ensemble._ModelManager") as mock_cls, \
         patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value=cerberus_config):
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, folds)

        return ModelEnsemble(
            checkpoint_path=".",
            device=torch.device("cpu")
        )

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

    assert cast(MockOutput, output).logits[0,0,0].item() == 1.5

def test_forward_selection():
    folds = create_mock_folds([
        [("chr1", 0, 100)],
        [("chr1", 100, 200)]
    ])

    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds, output_len=100, output_bin_size=1)

    x = torch.zeros(1, 1, 2)

    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    assert cast(MockOutput, out).logits[0,0,0].item() == 0.0

    interval = Interval("chr1", 100, 150)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    assert cast(MockOutput, out).logits[0,0,0].item() == 1.0

    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["val"])
    assert cast(MockOutput, out).logits[0,0,0].item() == 1.0

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
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)

    intervals = [
        Interval("chr1", 0, 100),
        Interval("chr1", 10, 110)
    ]

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
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)

    dataset = MagicMock()
    dataset.data_config = DataConfig.model_construct(
        inputs={}, targets={},
        input_len=input_len, output_len=output_len, output_bin_size=output_bin_size,
        max_jitter=0, encoding="ACGT", log_transform=False,
        reverse_complement=False, target_scale=1.0, use_sequence=True,
    )
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    intervals = [
        Interval("chr1", 0, 100),
        Interval("chr1", 10, 110)
    ]

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
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)

    dataset = MagicMock()
    dataset.data_config = DataConfig.model_construct(
        inputs={}, targets={},
        input_len=input_len, output_len=output_len, output_bin_size=output_bin_size,
        max_jitter=0, encoding="ACGT", log_transform=False,
        reverse_complement=False, target_scale=1.0, use_sequence=True,
    )
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    intervals = [Interval("chr1", 200, 300)]

    outputs = ensemble.predict_output_intervals(
        intervals, dataset, stride=50, use_folds=["test"], aggregation="model", batch_size=2
    )

    assert len(outputs) == 1
    out = outputs[0]
    merged_interval = out.out_interval
    assert merged_interval is not None

    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 200
    assert merged_interval.end == 270
    assert cast(MockOutput, out).logits.shape[-1] == 70
