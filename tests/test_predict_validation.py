
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from dataclasses import dataclass
from cerberus.output import ModelOutput
from cerberus.config import (
    DataConfig, GenomeConfig, SamplerConfig, TrainConfig, ModelConfig,
    CerberusConfig, FoldArgs, RandomSamplerArgs,
)

@dataclass
class MockOutput(ModelOutput):
    out_track: torch.Tensor

MOCK_DATA_CONFIG = DataConfig.model_construct(
    input_len=1000,
    output_len=200,
    output_bin_size=1,
    inputs={"seq": "mock.bw"},
    targets={"signal": "mock.bw"},
    use_sequence=True,
    encoding="onehot",
    max_jitter=0,
    reverse_complement=False,
    target_scale=1.0,
    log_transform=False,
)

MOCK_GENOME_CONFIG = GenomeConfig.model_construct(
    name="mock_genome",
    chrom_sizes={"chr1": 1000000},
    allowed_chroms=["chr1"],
    fold_type="random",
    fold_args=FoldArgs.model_construct(k=2, test_fold=None, val_fold=None),
    exclude_intervals={},
    fasta_path="mock.fa",
)

class MockEnsemble(ModelEnsemble):
    def __init__(self):
        nn.ModuleDict.__init__(self)
        self.device = torch.device("cpu")
        self.cerberus_config = CerberusConfig.model_construct(
            data_config=MOCK_DATA_CONFIG,
            genome_config=MOCK_GENOME_CONFIG,
            sampler_config=SamplerConfig.model_construct(
                sampler_type="random",
                padded_size=1000,
                sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
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
        self.folds = []
        # Mocking the internal dictionary of models
        self.update({"0": nn.Linear(1,1)})

    def forward(self, x, intervals=None, use_folds=None, aggregation="model"):
        batch_size = x.shape[0]
        return MockOutput(
            out_track=torch.zeros(batch_size, 1, MOCK_DATA_CONFIG.output_len),
            out_interval=None
        )

def test_predict_intervals_invalid_length():
    """Verify that predict_intervals raises ValueError for incorrect interval lengths."""
    # Setup Mock Dataset
    dataset = MagicMock(spec=CerberusDataset)
    dataset.data_config = MOCK_DATA_CONFIG

    # Mock get_interval to simulate Jitter behavior (resize to input_len)
    def side_effect_get_interval(interval):
        length = MOCK_DATA_CONFIG.input_len
        return {
            "inputs": torch.randn(4, length),
            "targets": torch.randn(1, length),
            "intervals": str(interval)
        }
    dataset.get_interval = MagicMock(side_effect=side_effect_get_interval)

    ensemble = MockEnsemble()

    # Create interval of incorrect length (2000 vs 1000)
    iv = Interval("chr1", 1000, 3000)

    with pytest.raises(ValueError, match="match model input length"):
        ensemble.predict_intervals([iv], dataset)

def test_predict_intervals_valid_length():
    """Verify that predict_intervals works for correct interval lengths."""
    # Setup Mock Dataset
    dataset = MagicMock(spec=CerberusDataset)
    dataset.data_config = MOCK_DATA_CONFIG

    def side_effect_get_interval(interval):
        length = MOCK_DATA_CONFIG.input_len
        return {
            "inputs": torch.randn(4, length),
            "targets": torch.randn(1, length),
            "intervals": str(interval)
        }
    dataset.get_interval = MagicMock(side_effect=side_effect_get_interval)

    ensemble = MockEnsemble()

    # Create interval of correct length (1000)
    iv = Interval("chr1", 1000, 2000)

    # Should not raise
    ensemble.predict_intervals([iv], dataset)
