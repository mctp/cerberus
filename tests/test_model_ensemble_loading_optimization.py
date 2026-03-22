from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.model_ensemble import ModelEnsemble
from cerberus.module import CerberusModule


# Define a simple dummy model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, hidden_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_len * 4, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        # x: (B, 4, L)
        b, c, l = x.shape
        x_flat = x.view(b, -1)
        h = self.linear(x_flat)
        return self.output(h)

@pytest.fixture
def mock_ensemble_dir(tmp_path):
    ensemble_dir = tmp_path / "test_ensemble"
    ensemble_dir.mkdir()

    metadata = {"folds": [0]}
    with open(ensemble_dir / "ensemble_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    fold_dir = ensemble_dir / "fold_0"
    fold_dir.mkdir()

    model = SimpleModel(input_len=10, output_len=1, output_bin_size=1)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[f"model.{k}"] = v

    checkpoint = {"state_dict": state_dict}
    torch.save(checkpoint, fold_dir / "val_loss=0.01.ckpt")

    return ensemble_dir

def test_model_ensemble_loads_stripped_weights(mock_ensemble_dir):
    model_config = ModelConfig.model_construct(
        name="SimpleModel",
        model_cls="tests.test_model_loading_optimization.SimpleModel",
        loss_cls="torch.nn.MSELoss",
        metrics_cls="torchmetrics.MeanSquaredError",
        loss_args={},
        metrics_args={},
        model_args={"hidden_dim": 10},
        pretrained=[],
        count_pseudocount=0.0,
    )

    data_config = DataConfig.model_construct(
        inputs={}, targets={},
        input_len=10, output_len=1,
        output_bin_size=1, max_jitter=0,
        encoding="ACGT", log_transform=False,
        reverse_complement=False, target_scale=1.0,
        use_sequence=True,
    )

    genome_config = GenomeConfig.model_construct(
        name="hg38", fasta_path=Path("genome.fa"),
        exclude_intervals={}, allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1000},
        fold_type="chrom_partition",
        fold_args={"k": 1, "test_fold": None, "val_fold": None},
    )

    cerberus_config = CerberusConfig.model_construct(
        data_config=data_config,
        genome_config=genome_config,
        sampler_config=SamplerConfig.model_construct(
            sampler_type="random", padded_size=10,
            sampler_args={"num_intervals": 10},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1, max_epochs=1, learning_rate=1e-3, weight_decay=0.0,
            patience=5, optimizer="adam", scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0, adam_eps=1e-8,
            gradient_clip_val=None,
        ),
        model_config_=model_config,
    )

    from cerberus import module
    original_import = module.import_class

    def mock_import(name):
        if name == "tests.test_model_loading_optimization.SimpleModel":
            return SimpleModel
        return original_import(name)

    module.import_class = mock_import

    try:
        with patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
             patch("cerberus.model_ensemble.parse_hparams_config", return_value=cerberus_config):
            ensemble = ModelEnsemble(
                mock_ensemble_dir,
                model_config=model_config,
                data_config=data_config,
                genome_config=genome_config,
                device="cpu"
            )

        assert "0" in ensemble
        loaded_model = ensemble["0"]

        simple_model = cast(SimpleModel, loaded_model)
        assert torch.all(simple_model.linear.weight == 1.0)

        assert isinstance(loaded_model, SimpleModel)
        assert not isinstance(loaded_model, CerberusModule)

    finally:
        module.import_class = original_import
