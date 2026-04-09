from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

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
from cerberus.output import ModelOutput


@dataclass
class MockOutputWithCounts(ModelOutput):
    logits: torch.Tensor
    log_counts: torch.Tensor

    def detach(self):
        return self


class MockModelWithCounts(nn.Module):
    def __init__(self, output_len):
        super().__init__()
        self.output_len = output_len

    def forward(self, x):
        batch_size = x.shape[0]
        return MockOutputWithCounts(
            logits=torch.ones(batch_size, 1, self.output_len),
            log_counts=torch.ones(batch_size, 1),
        )


def _make_cerberus_config(output_len=100, output_bin_size=1):
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


def test_predict_large_region_with_counts():
    input_len = 100
    output_len = 20
    output_bin_size = 1

    models = {"0": MockModelWithCounts(output_len)}
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

    intervals = [Interval("chr1", 0, 100)]

    outputs = ensemble.predict_output_intervals(
        intervals,
        dataset,
        stride=10,
        use_folds=["test"],
        aggregation="model",
        batch_size=2,
    )

    assert len(outputs) == 1
    out = outputs[0]
    out = cast(MockOutputWithCounts, out)

    assert out.out_interval is not None
    assert out.logits.ndim >= 2

    assert out.log_counts.ndim == 1, (
        f"log_counts should be (C,) after aggregation, got shape {out.log_counts.shape}"
    )
