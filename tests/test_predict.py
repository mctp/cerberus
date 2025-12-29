import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pybigtools
import os
from torchmetrics import MetricCollection

from cerberus.predict import predict_to_bigwig, parse_intervals, merge_intervals, _predict_to
from cerberus.interval import Interval
from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    PredictConfig,
)

class DummyModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.output_dim = output_len // output_bin_size
        
    def forward(self, x):
        # x: (B, 4, L)
        # Output 1.0 everywhere
        return torch.ones((x.shape[0], self.output_dim), device=x.device)

@pytest.fixture
def genome_setup(tmp_path):
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

def test_parse_intervals(genome_setup, tmp_path):
    genome_config = genome_setup
    
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

def test_parse_intervals_edge_cases(genome_setup, tmp_path):
    genome_config = genome_setup
    
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

def test_predict_generator_logic(tmp_path, genome_setup):
    # Tests the generator function _predict_to directly without writing file
    genome_config = genome_setup
    
    # Setup model and checkpoint
    model = DummyModel(input_len=100, output_len=50, output_bin_size=1)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    
    data_config: DataConfig = {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "max_jitter": 0,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    }
    
    model_config: ModelConfig = {
        "name": "dummy",
        "model_cls": DummyModel,
        "loss_cls": nn.MSELoss,
        "loss_args": {},
        "metrics_cls": MetricCollection,
        "metrics_args": {"metrics": {}},
        "model_args": {}
    }
    
    train_config: TrainConfig = {
        "batch_size": 2,
        "max_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "optimizer": "adam",
        "filter_bias_and_bn": False,
        "scheduler_type": "default",
        "scheduler_args": {}
    }
    
    predict_config: PredictConfig = {
        "stride": 50,
        "intervals": ["chr1:0-200"],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    }
    
    # Run generator
    gen = _predict_to(
        checkpoint_path=ckpt_path,
        genome_config=genome_config,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        predict_config=predict_config,
        batch_size=2,
        device="cpu"
    )
    
    # Collect results
    results = list(gen)
    
    # Check basic properties
    assert len(results) > 0
    
    # Check content: (chrom, start, end, val)
    for res in results:
        assert res[0] == "chr1"
        assert res[1] >= 0 and res[2] <= 200
        # Value should be 1.0 from DummyModel
        assert np.isclose(res[3], 1.0)
    
    # Check coverage
    # We expect coverage from 0 to 200
    # The generator yields chunks of model_bin_size (1)
    # So we should have 200 entries if everything is covered
    total_len = sum(r[2] - r[1] for r in results)
    assert total_len == 200


def test_predict_to_bigwig_integration(tmp_path, genome_setup):
    # Setup (reuse logic from previous test, but use fixture)
    genome_config = genome_setup
    
    # Checkpoint
    model = DummyModel(input_len=100, output_len=50, output_bin_size=1)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    
    data_config: DataConfig = {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "max_jitter": 0,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    }
    
    model_config: ModelConfig = {
        "name": "dummy",
        "model_cls": DummyModel,
        "loss_cls": nn.MSELoss,
        "loss_args": {},
        "metrics_cls": MetricCollection,
        "metrics_args": {"metrics": {}},
        "model_args": {}
    }
    
    train_config: TrainConfig = {
        "batch_size": 2,
        "max_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "optimizer": "adam",
        "filter_bias_and_bn": False,
        "scheduler_type": "default",
        "scheduler_args": {}
    }
    
    predict_config: PredictConfig = {
        "stride": 50,
        "intervals": ["chr1:0-200"],
        "intervals_paths": [],
        "use_folds": ["test"],
        "aggregation": "mean"
    }
    
    output_bw = tmp_path / "output.bw"
    
    # Run
    predict_to_bigwig(
        output_path=str(output_bw),
        checkpoint_path=ckpt_path,
        genome_config=genome_config,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        predict_config=predict_config,
        batch_size=2,
        device="cpu"
    )
    
    assert output_bw.exists()
    
    # Verify content
    with pybigtools.open(str(output_bw)) as bw: # type: ignore
        vals = list(bw.records("chr1", 0, 200))
        assert len(vals) > 0
        assert np.isclose(vals[0][2], 1.0)
