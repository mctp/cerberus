
import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from typing import cast

from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import ModelConfig, DataConfig, GenomeConfig
from cerberus.output import ModelOutput
from cerberus.interval import Interval

@dataclass
class MockOutputWithCounts(ModelOutput):
    logits: torch.Tensor
    log_counts: torch.Tensor
    def detach(self): return self

class MockModelWithCounts(nn.Module):
    def __init__(self, output_len):
        super().__init__()
        self.output_len = output_len
    def forward(self, x):
        # Return logits (B, 1, L) and log_counts (B, 1)
        batch_size = x.shape[0]
        return MockOutputWithCounts(
            logits=torch.ones(batch_size, 1, self.output_len),
            log_counts=torch.ones(batch_size, 1)
        )

# Helper to create ensemble (simplified version of one in test_model_ensemble.py)
def create_ensemble(models, folds, output_len=100, output_bin_size=1):
    with patch("cerberus.model_ensemble._ModelManager") as mock_cls:
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, folds)
        
        dummy_conf = {}
        data_conf = {"output_len": output_len, "output_bin_size": output_bin_size}
        
        return ModelEnsemble(
            checkpoint_path=".",
            model_config=cast(ModelConfig, dummy_conf),
            data_config=cast(DataConfig, data_conf),
            genome_config=cast(GenomeConfig, dummy_conf),
            device=torch.device("cpu")
        )

def test_predict_large_region_with_counts():
    """
    Regression test for bug where scalar outputs (log_counts) caused broadcasting errors
    during hierarchical aggregation of partial tracks.
    """
    # Setup
    input_len = 100
    output_len = 20
    output_bin_size = 1
    
    models = {"0": MockModelWithCounts(output_len)}
    folds = []
    ensemble = create_ensemble(models, folds, output_len=output_len, output_bin_size=output_bin_size)
    
    # Mock Dataset
    dataset = MagicMock()
    dataset.data_config = {"input_len": input_len, "output_len": output_len, "output_bin_size": output_bin_size}
    dataset.get_interval.return_value = {"inputs": torch.zeros(4, input_len)}

    # Target Interval: Large enough to require multiple tiles
    # Target: 0-100.
    # Stride: 10.
    # Tiles will cover the region.
    # This triggers batching and merging of overlapping predictions.
    intervals = [Interval("chr1", 0, 100)]
    
    # Run predict_output_intervals
    # This should not raise ValueError
    outputs = ensemble.predict_output_intervals(
        intervals, dataset, stride=10, use_folds=["test"], aggregation="model", batch_size=2
    )
    
    assert len(outputs) == 1
    out = outputs[0]
    # Cast for type checker
    out = cast(MockOutputWithCounts, out)
    
    assert out.out_interval is not None
    assert out.logits.ndim >= 2
    
    # Check that log_counts is now a track (because it was aggregated spatially)
    # The output is over the union of tiles.
    # Since we aggregated overlapping scalar counts, the result should be a spatial track
    # representing the count density/average over the region.
    # Its length should match logits length (which matches the union interval).
    assert out.log_counts.shape[-1] == out.logits.shape[-1]
