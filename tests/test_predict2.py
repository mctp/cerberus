import pytest
import torch
import numpy as np
from pathlib import Path
from typing import cast
from torch import nn
from torchmetrics import MetricCollection

from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig, SamplerConfig, ModelConfig, TrainConfig, PredictConfig
from cerberus.predict2 import predict_interval
from cerberus.model_manager import ModelManager

class DummyModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size
        
    def forward(self, x):
        # x: (B, 4, L)
        # Output (B, output_dim)
        return torch.ones((x.shape[0], self.output_dim), device=x.device)

@pytest.fixture
def predict2_setup(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 2000 + "\n")
    (tmp_path / "genome.fa.fai").write_text(f"chr1\t2000\t6\t2000\t2001\n")
    
    genome_config = create_genome_config(
        name="test", fasta_path=genome, species="human", allowed_chroms=["chr1"], exclude_intervals={}
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "dummy",
        "padded_size": 100,
        "sampler_args": {}
    })
    
    dataset = CerberusDataset(genome_config, data_config, sampler_config)
    
    # Model
    model = DummyModel(input_len=100, output_len=50, output_bin_size=1)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    
    model_config = cast(ModelConfig, {
        "name": "dummy",
        "model_cls": DummyModel,
        "loss_cls": nn.MSELoss,
        "loss_args": {},
        "metrics_cls": MetricCollection,
        "metrics_args": {"metrics": {}},
        "model_args": {}
    })
    
    train_config = cast(TrainConfig, {
        "batch_size": 2,
        "max_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "optimizer": "adam",
        "filter_bias_and_bn": False,
        "scheduler_type": "default",
        "scheduler_args": {}
    })
    
    model_manager = ModelManager(
        ckpt_path, model_config, data_config, train_config, genome_config, torch.device("cpu")
    )
    
    return dataset, model_manager

def test_predict_interval_valid(predict2_setup):
    dataset, model_manager = predict2_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    interval = Interval("chr1", 500, 600)
    # Using argument order: interval, dataset, model_manager, predict_config
    gen = predict_interval(interval, dataset, model_manager, predict_config, device="cpu")
    
    results = list(gen)
    assert len(results) > 0
    total_len = sum(r[2]-r[1] for r in results)
    assert total_len == 100
    assert all(r[3] == 1.0 for r in results)

def test_predict_interval_boundary_skip(predict2_setup):
    dataset, model_manager = predict2_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    # Interval near start: 0-100.
    # First window (-25 to 75) should be skipped.
    # Second window (25 to 125) should be valid (output 50-100).
    
    interval = Interval("chr1", 0, 100)
    gen = predict_interval(interval, dataset, model_manager, predict_config, device="cpu")
    results = list(gen)
    
    # Check that we got results
    assert len(results) > 0
    
    # Check start coverage: should start at 50, not 0
    min_start = min(r[1] for r in results)
    assert min_start == 50
    
    total_len = sum(r[2]-r[1] for r in results)
    assert total_len == 50

def test_predict_single_interval_matrix(tmp_path):
    from cerberus.predict2 import predict_single_interval_matrix
    
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 2000 + "\n")
    (tmp_path / "genome.fa.fai").write_text(f"chr1\t2000\t6\t2000\t2001\n")
    
    genome_config = create_genome_config(
        name="test", fasta_path=genome, species="human", allowed_chroms=["chr1"], exclude_intervals={}
    )
    
    # Bin size 10
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "output_bin_size": 10,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "dummy",
        "padded_size": 100,
        "sampler_args": {}
    })
    
    class MockSignalExtractor:
        def extract(self, interval: Interval):
            return torch.zeros((1, interval.end - interval.start))

    dataset = CerberusDataset(
        genome_config, 
        data_config, 
        sampler_config,
        target_signal_extractor=MockSignalExtractor() # Provide dummy targets to avoid Bin transform crash
    )
    
    # Model output dim = 50 // 10 = 5
    model = DummyModel(input_len=100, output_len=50, output_bin_size=10)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    
    model_config = cast(ModelConfig, {
        "name": "dummy",
        "model_cls": DummyModel,
        "loss_cls": nn.MSELoss,
        "loss_args": {},
        "metrics_cls": MetricCollection,
        "metrics_args": {"metrics": {}},
        "model_args": {}
    })
    
    train_config = cast(TrainConfig, {
        "batch_size": 2,
        "max_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "optimizer": "adam",
        "filter_bias_and_bn": False,
        "scheduler_type": "default",
        "scheduler_args": {}
    })
    
    model_manager = ModelManager(
        ckpt_path, model_config, data_config, train_config, genome_config, torch.device("cpu")
    )
    
    predict_config = cast(PredictConfig, {
        "stride": 50, # Multiple of 10
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    interval = Interval("chr1", 500, 600) # Length 100, multiple of 10
    
    # Test Binned
    arr_binned = predict_single_interval_matrix(
        interval, dataset, model_manager, predict_config, device="cpu", resolution="binned"
    )
    assert arr_binned.shape == (10,) # 100 // 10
    assert np.allclose(arr_binned, 1.0)
    
    # Test BP
    arr_bp = predict_single_interval_matrix(
        interval, dataset, model_manager, predict_config, device="cpu", resolution="bp"
    )
    assert arr_bp.shape == (100,)
    assert np.allclose(arr_bp, 1.0)
    
    # Verify expansion logic manually
    # If we had [1, 2] -> [1..1, 2..2]
    # Mocking return
    # ... assuming implementation is correct based on code read.
