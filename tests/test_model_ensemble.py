import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from typing import cast

from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import ModelConfig, DataConfig, TrainConfig, GenomeConfig, PredictConfig
from cerberus.output import ModelOutput
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
    # intervals_per_fold: list of list of (chrom, start, end)
    # returns list of dicts. dict[chrom] -> mock with find
    folds = []
    for intervals in intervals_per_fold:
        fold_map = {}
        # Simple mock that checks if queried interval overlaps any in the list
        # This is enough for the ModelEnsemble logic which uses .find()
        
        # We need to support fold_map[chrom].find((start, end))
        # We can use a real IntervalTree or a mock.
        # Let's use a mock for simplicity and avoiding dependency on intervaltree package in test if not installed (though cerberus depends on it).
        
        # Group by chrom
        by_chrom = {}
        for (chrom, start, end) in intervals:
            if chrom not in by_chrom: by_chrom[chrom] = []
            by_chrom[chrom].append((start, end))
            
        for chrom, ints in by_chrom.items():
            tree_mock = MagicMock()
            
            def make_find(ints=ints):
                def find(query_range):
                    q_start, q_end = query_range
                    # Check overlap
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

def create_ensemble(models, folds, output_len=100, output_bin_size=1):
    with patch("cerberus.model_ensemble._ModelManager") as mock_cls:
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, folds)
        
        dummy_conf = {}
        data_conf = {"output_len": output_len, "output_bin_size": output_bin_size}
        
        return ModelEnsemble(
            checkpoint_path="dummy",
            model_config=cast(ModelConfig, dummy_conf),
            data_config=cast(DataConfig, data_conf),
            train_config=cast(TrainConfig, dummy_conf),
            genome_config=cast(GenomeConfig, dummy_conf),
            device=torch.device("cpu")
        )

def test_initialization():
    models = {"0": MockModel(1.0), "1": MockModel(2.0)}
    folds = []
    ensemble = create_ensemble(models, folds)
    
    assert len(ensemble) == 2
    assert "0" in ensemble
    assert "1" in ensemble

def test_forward_no_folds():
    # k=0 case (single model behavior)
    models = {"0": MockModel(1.0), "1": MockModel(2.0)}
    ensemble = create_ensemble(models, folds=[], output_len=100, output_bin_size=1)
    
    x = torch.zeros(1, 1, 2)
    output = ensemble(x, intervals=None) # No intervals provided, defaults to model aggregation
    
    # Should return mean of 1.0 and 2.0 -> 1.5
    assert output.logits[0,0,0].item() == 1.5

def test_forward_selection():
    # Setup 2 folds
    # Fold 0: chr1:0-100
    # Fold 1: chr1:100-200
    folds = create_mock_folds([
        [("chr1", 0, 100)],   # Fold 0
        [("chr1", 100, 200)]  # Fold 1
    ])
    
    models = {"0": MockModel(0.0), "1": MockModel(1.0)}
    ensemble = create_ensemble(models, folds, output_len=100, output_bin_size=1)
    
    x = torch.zeros(1, 1, 2)
    
    # Interval in Fold 0 (0-50)
    # Target partition: 0
    # use_folds=['test'] -> Select model 0
    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    # Returns aggregated output (single object)
    assert out.logits[0,0,0].item() == 0.0
    
    # Interval in Fold 1 (100-150)
    # Target partition: 1
    # use_folds=['test'] -> Select model 1
    interval = Interval("chr1", 100, 150)
    out = ensemble(x, intervals=[interval], use_folds=["test"])
    assert out.logits[0,0,0].item() == 1.0
    
    # Interval in Fold 0
    # use_folds=['val'] -> (p_idx - 1) % 2.
    # p_idx = 0. val_idx = -1 % 2 = 1.
    # Select model 1.
    interval = Interval("chr1", 0, 50)
    out = ensemble(x, intervals=[interval], use_folds=["val"])
    assert out.logits[0,0,0].item() == 1.0

def test_aggregate():
    ensemble = create_ensemble({}, [], output_len=100, output_bin_size=1)
    
    out1 = MockOutput(logits=torch.ones(2, 2))
    out2 = MockOutput(logits=torch.ones(2, 2) * 3)
    
    agg = ensemble._aggregate_models([out1, out2], method="mean")
    assert torch.allclose(agg.logits, torch.ones(2, 2) * 2)

def test_forward_interval_aggregation():
    # Test that aggregation="interval" correctly centers input intervals
    # Input len 100, Output len 20.
    
    # Setup
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len
        def forward(self, x):
            # Return fixed size output
            batch_size = x.shape[0]
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)
    
    # Input Intervals: Two intervals offset by 10bp
    # Interval 1: 0-100. Center (output) should be 40-60.
    # Interval 2: 10-110. Center (output) should be 50-70.
    intervals = [
        Interval("chr1", 0, 100),
        Interval("chr1", 10, 110)
    ]
    
    x = torch.zeros(2, 4, input_len) # Batch 2
    
    # Run forward with aggregation="interval+model" (since "interval" is removed)
    # With 1 model, "interval+model" behaves like "interval" but returns single object
    output = ensemble(x, intervals=intervals, aggregation="interval+model")
    
    logits = output.logits
    merged_interval = output.out_interval
    
    # Check Merged Interval
    # Should be union of 40-60 and 50-70 -> 40-70.
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 40
    assert merged_interval.end == 70
    
    # Check output shape (30 bins)
    assert logits.shape[-1] == 30

def test_predict_intervals_method():
    # Setup
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len
        def forward(self, x):
            # Return fixed size output
            batch_size = x.shape[0]
            # (Batch, 1, OutputLen)
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)
    
    # Mock Dataset
    dataset = MagicMock()
    dataset.data_config = {"input_len": input_len, "output_len": output_len, "output_bin_size": output_bin_size}
    # Return inputs: (Channels=4, Length=100)
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    # Intervals
    intervals = [
        Interval("chr1", 0, 100),
        Interval("chr1", 10, 110)
    ]
    
    predict_config = cast(PredictConfig, {"aggregation": "model", "use_folds": ["test"], "stride": 10})
    
    # Run predict_intervals
    output = ensemble.predict_intervals(
        intervals, dataset, predict_config, batch_size=2
    )
    
    merged_interval = output.out_interval
    assert merged_interval is not None
    
    # Interval 1: 0-100 -> Output 40-60
    # Interval 2: 10-110 -> Output 50-70
    # Union: 40-70. Length 30.
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 40
    assert merged_interval.end == 70
    assert output.logits.shape[-1] == 30 # type: ignore

def test_predict_output_intervals_method():
    # Setup
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    class FixedSizeMockModel(nn.Module):
        def __init__(self, output_len):
            super().__init__()
            self.output_len = output_len
        def forward(self, x):
            # Return fixed size output
            batch_size = x.shape[0]
            # (Batch, 1, OutputLen)
            return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

    models = {"0": FixedSizeMockModel(output_len)}
    folds = []
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)
    
    # Mock Dataset
    dataset = MagicMock()
    dataset.data_config = {"input_len": input_len, "output_len": output_len, "output_bin_size": output_bin_size}
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    # Target Interval: chr1:200-300 (100bp)
    # Stride: 50
    # Input len: 100, Output len: 20. Offset: (100-20)//2 = 40.
    
    # Tile 1: Target start 200. Input start = 200 - 40 = 160. Input end 260. Output: 200-220.
    # Tile 2: Target start 250. Input start = 250 - 40 = 210. Input end 310. Output: 250-270.
    
    intervals = [Interval("chr1", 200, 300)]
    
    predict_config = cast(PredictConfig, {"aggregation": "model", "use_folds": ["test"], "stride": 50})
    
    # Run predict_output_intervals
    outputs = ensemble.predict_output_intervals(
        intervals, dataset, predict_config, batch_size=2
    )
    
    assert len(outputs) == 1
    out = outputs[0]
    merged_interval = out.out_interval
    assert merged_interval is not None
    
    # Check bounds
    # The method tiles the TARGET interval.
    # It generates prediction for windows starting at 200, 250.
    # Window 1 output: 200-220
    # Window 2 output: 250-270
    # Merged interval: 200-270 (union of outputs)
    
    assert merged_interval.chrom == "chr1"
    assert merged_interval.start == 200
    assert merged_interval.end == 270
    assert out.logits.shape[-1] == 70 # type: ignore
