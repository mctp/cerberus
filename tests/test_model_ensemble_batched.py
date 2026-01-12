import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from typing import cast
import itertools

from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import ModelConfig, DataConfig, GenomeConfig
from cerberus.output import ModelOutput
from cerberus.interval import Interval

@dataclass
class MockOutput(ModelOutput):
    logits: torch.Tensor
    def detach(self): return self

class FixedSizeMockModel(nn.Module):
    def __init__(self, output_len):
        super().__init__()
        self.output_len = output_len
    def forward(self, x):
        # Return fixed size output
        batch_size = x.shape[0]
        # (Batch, 1, OutputLen)
        return MockOutput(logits=torch.ones(batch_size, 1, self.output_len))

def create_ensemble(models, output_len=100, output_bin_size=1):
    with patch("cerberus.model_ensemble._ModelManager") as mock_cls, \
         patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=MagicMock()), \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value={
             "model_config": {}, "data_config": {}, "genome_config": {}
         }):
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, [])
        
        dummy_conf = {}
        data_conf = {"output_len": output_len, "output_bin_size": output_bin_size, "input_len": 100}
        
        return ModelEnsemble(
            checkpoint_path=".",
            model_config=cast(ModelConfig, dummy_conf),
            data_config=cast(DataConfig, data_conf),
            genome_config=cast(GenomeConfig, dummy_conf),
            device=torch.device("cpu")
        )

def test_predict_intervals_batched():
    # Setup
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    models = {"0": FixedSizeMockModel(output_len)}
    ensemble = create_ensemble(models, output_len=output_len, output_bin_size=output_bin_size)
    
    # Mock Dataset
    dataset = MagicMock()
    dataset.data_config = {"input_len": input_len, "output_len": output_len, "output_bin_size": output_bin_size}
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    # Intervals: 5 intervals
    intervals = [Interval("chr1", i*100, i*100+100) for i in range(5)]

    # Batch size 2. Expected 3 batches (2, 2, 1)
    generator = ensemble.predict_intervals_batched(
        intervals, dataset, use_folds=["test"], aggregation="model", batch_size=2
    )
    
    batches = list(generator)
    assert len(batches) == 3
    
    # Check batch 1
    out1, ints1 = batches[0]
    assert len(ints1) == 2
    assert cast(MockOutput, out1).logits.shape[0] == 2
    
    # Check batch 3
    out3, ints3 = batches[2]
    assert len(ints3) == 1
    assert cast(MockOutput, out3).logits.shape[0] == 1

def test_predict_intervals_consistency():
    """
    Ensure refactored predict_intervals still works correctly by aggregating results from batched generator.
    """
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    models = {"0": FixedSizeMockModel(output_len)}
    ensemble = create_ensemble(models, output_len=output_len, output_bin_size=output_bin_size)
    
    # Mock Dataset
    dataset = MagicMock()
    dataset.data_config = {"input_len": input_len, "output_len": output_len, "output_bin_size": output_bin_size}
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    # Intervals: 2 disjoint intervals that map to disjoint output intervals
    # Input 0-100 -> Output 40-60
    # Input 200-300 -> Output 240-260
    intervals = [
        Interval("chr1", 0, 100),
        Interval("chr1", 200, 300)
    ]

    # Run predict_intervals
    output = ensemble.predict_intervals(
        intervals, dataset, use_folds=["test"], aggregation="model", batch_size=1
    )
    
    merged_interval = output.out_interval
    # Union of 40-60 and 240-260 -> 40-260
    assert merged_interval.start == 40
    assert merged_interval.end == 260
    
    # We expect 220 bins (260-40)
    assert cast(MockOutput, output).logits.shape[-1] == 220
