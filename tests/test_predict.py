import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import cast, List, Tuple, Union, Any, Dict, Iterable
from unittest.mock import MagicMock, Mock
from torchmetrics import MetricCollection

from cerberus.interval import parse_intervals, merge_intervals, Interval
from cerberus.predict import (
    predict_intervals, 
    _aggregate_ensemble_outputs, 
    _aggregate_overlapping_output_intervals
)
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config
from cerberus.model_manager import ModelManager
from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    PredictConfig,
    SamplerConfig,
)

# --- Fixtures and Helpers for Integration Tests ---

class DummyModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size
        
    def forward(self, x):
        # x: (B, 4, L)
        # Output (B, 1, output_dim)
        return (torch.ones((x.shape[0], 1, self.output_dim), device=x.device),)

@pytest.fixture
def integration_setup(tmp_path):
    genome = tmp_path / "genome.fa"
    with open(genome, "w") as f:
        f.write(">chr1\n" + "A" * 2000 + "\n")
    with open(tmp_path / "genome.fa.fai", "w") as f:
        f.write(f"chr1\t2000\t6\t2000\t2001\n")
    
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

# --- Fixtures and Helpers for Unit Tests (Mocking) ---

class MockModel(nn.Module):
    def __init__(self, value=1.0, output_len=None):
        super().__init__()
        self.value = value
        self.output_len = output_len
    
    def forward(self, x):
        out = x * self.value
        if self.output_len is not None:
            # Center crop
            curr = out.shape[-1]
            if curr > self.output_len:
                diff = curr - self.output_len
                start = diff // 2
                out = out[..., start : start + self.output_len]
        return (out,)

class MockTupleModel(nn.Module):
    def __init__(self, value1=1.0, value2=2.0, output_len=None):
        super().__init__()
        self.value1 = value1
        self.value2 = value2
        self.output_len = output_len
        
    def forward(self, x):
        out1 = x * self.value1
        out2 = x * self.value2
        if self.output_len is not None:
            curr = out1.shape[-1]
            if curr > self.output_len:
                diff = curr - self.output_len
                start = diff // 2
                out1 = out1[..., start : start + self.output_len]
                out2 = out2[..., start : start + self.output_len]
        return (out1, out2)

class MockScalarModel(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        
    def forward(self, x):
        # Return (Batch, 1) scalar
        return (torch.ones(x.shape[0], 1) * self.value,)

@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    # input_len=100, output_len=60 => offset=20
    dataset.data_config = {"input_len": 100, "output_len": 60, "output_bin_size": 1}
    # dataset.get_interval returns dict with "inputs"
    dataset.get_interval.return_value = {"inputs": torch.ones(4, 100)}
    return dataset

@pytest.fixture
def mock_model_manager():
    manager = MagicMock()
    return manager

# --- Tests for Interval Utilities (from original test_predict.py) ---

@pytest.fixture
def genome_setup_basic(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 1000 + "\n")
        f.write(">chr2\n" + "G" * 1000 + "\n")
        
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    
    genome_config: GenomeConfig = {
        "name": "test_genome",
        "fasta_path": fasta_path,
        "exclude_intervals": {},
        "allowed_chroms": ["chr1", "chr2"],
        "chrom_sizes": chrom_sizes,
        "fold_type": "chrom_partition",
        "fold_args": {"k": 2}
    }
    return genome_config

def test_parse_intervals(genome_setup_basic, tmp_path):
    genome_config = genome_setup_basic
    
    # Test string intervals
    intervals = ["chr1:0-100", "chr2"]
    parsed = parse_intervals(intervals, [], genome_config)
    assert len(parsed) == 2
    assert parsed[0] == Interval("chr1", 0, 100)
    assert parsed[1] == Interval("chr2", 0, 1000)
    
    # Test file intervals
    bed_path = tmp_path / "test.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t200\t300\n")
    
    parsed_file = parse_intervals([], [bed_path], genome_config)
    assert len(parsed_file) == 1
    assert parsed_file[0] == Interval("chr1", 200, 300)
    
    # Test default whole genome
    parsed_default = parse_intervals([], [], genome_config)
    assert len(parsed_default) == 2 # chr1 and chr2
    assert Interval("chr1", 0, 1000) in parsed_default
    assert Interval("chr2", 0, 1000) in parsed_default

    # Test invalid format
    with pytest.raises(ValueError, match="Invalid interval format"):
        parse_intervals(["chr1:invalid"], [], genome_config)
        
    # Test unknown chrom
    with pytest.raises(ValueError, match="Chromosome chr3 not found"):
        parse_intervals(["chr3"], [], genome_config)

def test_parse_intervals_edge_cases(genome_setup_basic, tmp_path):
    genome_config = genome_setup_basic
    
    # 1. Mixed Inputs
    bed_path = tmp_path / "mixed.bed"
    with open(bed_path, "w") as f:
        f.write("chr2\t500\t600\n")
        
    parsed = parse_intervals(["chr1:100-200"], [bed_path], genome_config)
    assert len(parsed) == 2
    assert Interval("chr1", 100, 200) in parsed
    assert Interval("chr2", 500, 600) in parsed

    # 2. BED File Robustness (extra columns, empty lines)
    robust_bed = tmp_path / "robust.bed"
    with open(robust_bed, "w") as f:
        f.write("chr1\t100\t200\tname\t0\t+\n") # Extra columns
        f.write("chr2\n") # Too few columns
        f.write("\n") # Empty line
        
    parsed_robust = parse_intervals([], [robust_bed], genome_config)
    assert len(parsed_robust) == 1
    assert parsed_robust[0] == Interval("chr1", 100, 200)

    # 3. Malformed Coordinates in BED
    bad_bed = tmp_path / "bad.bed"
    with open(bad_bed, "w") as f:
        f.write("chr1\tstart\tend\n")
        
    with pytest.raises(ValueError):
        parse_intervals([], [bad_bed], genome_config)

    # 4. Whitespace in string format
    # "chr1: 100 - 200" -> int() handles spaces around numbers
    parsed_spaces = parse_intervals(["chr1: 100 - 200"], [], genome_config)
    assert parsed_spaces[0] == Interval("chr1", 100, 200)

    # 5. Non-existent file
    with pytest.raises(FileNotFoundError):
        parse_intervals([], [tmp_path / "non_existent.bed"], genome_config)

    # 6. Chromosome not in config (Coordinate based)
    # The current implementation DOES NOT check if chrom is in config if coordinates are provided
    parsed_unknown = parse_intervals(["chrUnknown:0-100"], [], genome_config)
    assert parsed_unknown[0] == Interval("chrUnknown", 0, 100)

def test_merge_intervals():
    # Test merging overlapping
    intervals = [Interval("chr1", 0, 100), Interval("chr1", 50, 150)]
    merged = merge_intervals(intervals)
    assert len(merged) == 1
    assert merged[0] == Interval("chr1", 0, 150)
    
    # Test merging adjacent
    intervals = [Interval("chr1", 0, 100), Interval("chr1", 100, 200)]
    merged = merge_intervals(intervals)
    assert len(merged) == 1
    assert merged[0] == Interval("chr1", 0, 200)
    
    # Test non-overlapping
    intervals = [Interval("chr1", 0, 100), Interval("chr1", 200, 300)]
    merged = merge_intervals(intervals)
    assert len(merged) == 2
    assert merged[0] == Interval("chr1", 0, 100)
    assert merged[1] == Interval("chr1", 200, 300)
    
    # Test different chroms
    intervals = [Interval("chr1", 0, 100), Interval("chr2", 0, 100)]
    merged = merge_intervals(intervals)
    assert len(merged) == 2

# --- Integration Tests for Predict Logic (from original test_predict2.py) ---

def test_predict_intervals_valid(integration_setup):
    dataset, model_manager = integration_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    interval = Interval("chr1", 500, 600)
    # Using argument order: intervals, dataset, model_manager, predict_config
    # Returns (values, merged_interval)
    values, merged_interval = predict_intervals([interval], dataset, model_manager, predict_config, device="cpu")
    
    # Output dim is 50. Model returns (B, 1, 50).
    # Aggregation yields (C, n_bins) = (1, 50).
    assert isinstance(values, tuple)
    assert values[0].shape == (1, 50)
    assert np.allclose(values[0], 1.0)
    
    # Input 500-600 (center 550). Output len 50. 
    # Output interval: 525-575.
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 525
    assert merged_interval.end == 575

def test_predict_intervals_boundary_skip(integration_setup):
    dataset, model_manager = integration_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    # Input 0-100. Length 100 (input_len).
    # Center 50. Output len 50.
    # Output 25-75.
    
    interval = Interval("chr1", 0, 100)
    values, merged_interval = predict_intervals([interval], dataset, model_manager, predict_config, device="cpu")
    
    assert isinstance(values, tuple)
    assert values[0].shape == (1, 50)
    assert np.allclose(values[0], 1.0)
    
    assert merged_interval.start == 25
    assert merged_interval.end == 75

def test_predict_intervals_batching(integration_setup):
    # Test multiple intervals to ensure they are batched together
    dataset, model_manager = integration_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    # Intervals must be input_len (100)
    intervals = [
        Interval("chr1", 1000, 1100),
        Interval("chr1", 1100, 1200),
        Interval("chr1", 1200, 1300),
        Interval("chr1", 1300, 1400)
    ]
    
    values, merged_interval = predict_intervals(intervals, dataset, model_manager, predict_config, device="cpu")
    
    assert merged_interval.start == 1025
    assert merged_interval.end == 1375
    span = 1375 - 1025 # 350
    assert isinstance(values, tuple)
    assert values[0].shape == (1, 350)
    
    # Check filled regions (should be 1.0)
    # Relative starts: 0, 100, 200, 300
    # Length 50.
    # 0-50, 100-150, 200-250, 300-350 should be 1.0
    # Gaps: 50-100, 150-200, 250-300 should be 0.0
    
    assert np.allclose(values[0][0, 0:50], 1.0)
    assert np.allclose(values[0][0, 50:100], 0.0)
    assert np.allclose(values[0][0, 100:150], 1.0)
    assert np.allclose(values[0][0, 150:200], 0.0)
    assert np.allclose(values[0][0, 200:250], 1.0)
    assert np.allclose(values[0][0, 250:300], 0.0)
    assert np.allclose(values[0][0, 300:350], 1.0)

# --- Unit Tests for Logic (from original test_predict3.py) ---

def test_predict_interval_validation(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 50) # Wrong length
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    with pytest.raises(ValueError, match="length 50, expected 100"):
        predict_intervals([interval], mock_dataset, mock_model_manager, config, device="cpu")

def test_predict_interval_single_model(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    # output_len=60 to match dataset config
    model = MockModel(value=2.0, output_len=60)
    mock_model_manager.get_models.return_value = [model]
    
    output, out_interval = predict_intervals([interval], mock_dataset, mock_model_manager, config, device="cpu")
    
    # Output should be length 60
    assert isinstance(output, tuple)
    assert output[0].shape[-1] == 60
    assert np.allclose(output[0], np.ones((4, 60)) * 2.0)
    assert out_interval.start == 20
    assert out_interval.end == 80

def test_predict_interval_mean_aggregation(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    model1 = MockModel(value=2.0, output_len=60)
    model2 = MockModel(value=4.0, output_len=60)
    mock_model_manager.get_models.return_value = [model1, model2]
    
    output, out_interval = predict_intervals([interval], mock_dataset, mock_model_manager, config, device="cpu")
    
    assert isinstance(output, tuple)
    assert output[0].shape[-1] == 60
    assert np.allclose(output[0], np.ones((4, 60)) * 3.0)
    assert out_interval.start == 20

def test_predict_interval_tuple_output(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    model1 = MockTupleModel(value1=2.0, value2=10.0, output_len=60)
    model2 = MockTupleModel(value1=4.0, value2=20.0, output_len=60)
    mock_model_manager.get_models.return_value = [model1, model2]
    
    output, out_interval = predict_intervals([interval], mock_dataset, mock_model_manager, config, device="cpu")
    
    assert isinstance(output, tuple)
    assert output[0].shape[-1] == 60
    assert np.allclose(output[0], np.ones((4, 60)) * 3.0)
    assert np.allclose(output[1], np.ones((4, 60)) * 15.0)

def test_predict_intervals_overlap(mock_dataset, mock_model_manager):
    # Output is length 60, offset 20.
    # Int1: 0-100 -> Output 20-80
    # Int2: 10-110 -> Output 30-90
    interval_1 = Interval("chr1", 0, 100)
    interval_2 = Interval("chr1", 10, 110)
    
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    # Mock model manager to return models that output constant 1 for Int1 and 2 for Int2
    
    def get_interval_side_effect(interval):
        if interval.start == 0:
            return {"inputs": torch.ones(1, 100)} # (C=1, L=100)
        else:
            return {"inputs": torch.ones(1, 100) * 2.0}
            
    mock_dataset.get_interval.side_effect = get_interval_side_effect
    
    # Model must return output_len=60
    class MockCroppedModel(nn.Module):
        def __init__(self, value=1.0):
            super().__init__()
            self.value = value
        def forward(self, x):
            return (x[:, :, 20:80] * self.value,)

    model = MockCroppedModel(value=1.0)
    mock_model_manager.get_models.return_value = [model]
    
    # Run
    intervals = [interval_1, interval_2]
    results = predict_intervals(intervals, mock_dataset, mock_model_manager, config, device="cpu")
    
    # results is (array, interval)
    arr, merged_interval = results
    
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 20
    assert merged_interval.end == 90
    
    # Range: 20 to 90. Length 70.
    # arr shape: (1, 70)
    assert isinstance(arr, tuple)
    assert arr[0].shape == (1, 70)
    
    # [20, 30): From Int1 (val 1) -> Indices 0-10
    # [30, 80): Overlap (val 1 and 2) -> Indices 10-60 -> (1+2)/2 = 1.5
    # [80, 90): From Int2 (val 2) -> Indices 60-70
    
    assert np.allclose(arr[0][0, 0:10], 1.0)
    assert np.allclose(arr[0][0, 10:60], 1.5)
    assert np.allclose(arr[0][0, 60:70], 2.0)

def test_predict_intervals_scalar_broadcast(mock_dataset, mock_model_manager):
    # Scalar output
    interval_1 = Interval("chr1", 0, 100)
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    model = MockScalarModel(value=5.0)
    mock_model_manager.get_models.return_value = [model]
    
    results = predict_intervals([interval_1], mock_dataset, mock_model_manager, config, device="cpu")
    
    arr, merged_interval = results
    
    # Output length 60. Scalar should be broadcast to 60.
    assert isinstance(arr, tuple)
    assert arr[0].shape == (1, 60)
    assert np.allclose(arr[0], 5.0)
    assert merged_interval.start == 20
    assert merged_interval.end == 80

def test_predict_intervals_tuple_recursive(mock_dataset, mock_model_manager):
    # Tuple output (Profile, Scalar)
    # Output length 60 (for profile). Scalar broadcast.
    mock_dataset.data_config["output_bin_size"] = 1
    
    interval_1 = Interval("chr1", 0, 100)
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    class MockCorrectProfileModel(nn.Module):
        def forward(self, x):
            # Return (Profile 60, Scalar)
            prof = torch.ones(x.shape[0], 1, 60) * 3.0
            scalar = torch.ones(x.shape[0], 1) * 7.0
            return (prof, scalar)

    model = MockCorrectProfileModel()
    mock_model_manager.get_models.return_value = [model]
    
    results = predict_intervals([interval_1], mock_dataset, mock_model_manager, config, device="cpu")
    
    values, merged_interval = results
    assert isinstance(values, tuple)
    assert len(values) == 2
    
    track_prof = values[0]
    track_scalar = values[1]
    
    # Check Profile
    assert track_prof.shape == (1, 60)
    assert np.allclose(track_prof, 3.0)
    
    # Check Scalar
    assert track_scalar.shape == (1, 60)
    assert np.allclose(track_scalar, 7.0)
    
    assert merged_interval.start == 20
    assert merged_interval.end == 80

def test_predict_intervals_empty_input(mock_dataset, mock_model_manager):
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    with pytest.raises(ValueError, match="No intervals provided"):
        predict_intervals([], mock_dataset, mock_model_manager, config, device="cpu")

def test_aggregate_ensemble_outputs_empty():
    with pytest.raises(ValueError, match="No outputs to aggregate"):
        _aggregate_ensemble_outputs([], method="mean")

def test_aggregate_overlapping_output_intervals_empty():
    with pytest.raises(ValueError, match="No results to aggregate"):
        _aggregate_overlapping_output_intervals([], output_bin_size=1, output_len=100)

def test_merged_interval_is_multiple_of_bin_size(mock_dataset, mock_model_manager):
    # Setup intervals such that merged interval should be multiple of bin_size=10
    mock_dataset.data_config["output_bin_size"] = 10
    mock_dataset.data_config["output_len"] = 100
    
    # Input len 100. Output len 100.
    # Interval 1: 0-100. Output 0-100.
    # Interval 2: 50-150. Output 50-150.
    # Merged: 0-150.
    # 150 % 10 == 0.
    
    interval_1 = Interval("chr1", 0, 100)
    interval_2 = Interval("chr1", 50, 150)
    intervals = [interval_1, interval_2]
    
    config = cast(PredictConfig, {"use_folds": ["test"], "aggregation": "mean"})
    
    # Mock models
    class MockModelBin10(nn.Module):
        def forward(self, x):
            # Output length 100 // 10 = 10 bins
            return (torch.ones(x.shape[0], 1, 10),)

    model = MockModelBin10()
    mock_model_manager.get_models.return_value = [model]
    
    # Dataset needs to return inputs for these intervals
    mock_dataset.get_interval.return_value = {"inputs": torch.ones(4, 100)}
    
    values, merged_interval = predict_intervals(intervals, mock_dataset, mock_model_manager, config, device="cpu")
    
    merged_len = merged_interval.end - merged_interval.start
    assert merged_len % 10 == 0
    assert merged_len == 150

def test_predict_intervals_batching_param(integration_setup):
    # Test batch_size parameter
    dataset, model_manager = integration_setup
    
    predict_config = cast(PredictConfig, {
        "stride": 50,
        "intervals": [],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    })
    
    intervals = [
        Interval("chr1", 1000, 1100),
        Interval("chr1", 1100, 1200),
        Interval("chr1", 1200, 1300),
        Interval("chr1", 1300, 1400)
    ]
    
    # Run with batch_size=2
    values, merged_interval = predict_intervals(
        intervals, dataset, model_manager, predict_config, device="cpu", batch_size=2
    )
    
    assert merged_interval.start == 1025
    assert merged_interval.end == 1375
    assert values[0].shape == (1, 350)
    
    # Check values similar to test_predict_intervals_batching
    assert np.allclose(values[0][0, 0:50], 1.0)
    assert np.allclose(values[0][0, 50:100], 0.0)
    assert np.allclose(values[0][0, 100:150], 1.0)
    assert np.allclose(values[0][0, 150:200], 0.0)
    assert np.allclose(values[0][0, 200:250], 1.0)
    assert np.allclose(values[0][0, 250:300], 0.0)
    assert np.allclose(values[0][0, 300:350], 1.0)
