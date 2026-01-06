import pytest
import torch
import numpy as np
import dataclasses
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    ModelOutput, 
    unbatch_modeloutput, 
    aggregate_tensor_track_values, 
    aggregate_intervals, 
    aggregate_models
)
from cerberus.interval import Interval

# --- Fixtures and Helpers ---

@dataclasses.dataclass
class SimpleOutput(ModelOutput):
    """Simple dataclass for testing ModelOutput behavior."""
    logits: torch.Tensor
    profile: torch.Tensor
    # out_interval is inherited

@pytest.fixture
def mock_intervals():
    return [
        Interval("chr1", 100, 200, "+"),
        Interval("chr1", 200, 300, "+")
    ]

# --- Tests for _unbatch_modeloutput ---

def test_unbatch_modeloutput_basic(mock_intervals):
    """Test splitting a batched output into individual dictionaries."""
    batch_size = 2
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # (B, C)
    profile = torch.randn(2, 1, 100) # (B, C, L)
    
    batched = SimpleOutput(
        logits=logits,
        profile=profile,
        out_interval=None # Usually None for batched
    )
    
    result = unbatch_modeloutput(batched, batch_size)
    
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert torch.equal(result[0]["logits"], logits[0])
    assert torch.equal(result[1]["logits"], logits[1])
    assert torch.equal(result[0]["profile"], profile[0])
    
    # Check that scalar/metadata is duplicated
    # (Though SimpleOutput doesn't have extra fields, let's verify behavior if we added one)
    # The current implementation of _unbatch_modeloutput iterates over keys.
    
def test_unbatch_modeloutput_mismatch_raises():
    """Test that it handles (or fails?) if batch size doesn't match tensor dim 0."""
    # The code relies on torch.unbind(val, dim=0). If dim 0 != batch_size, unbind works but the list length differs.
    # The code then loops range(batch_size). 
    # If unbind produces N items and batch_size != N, index error will occur or data loss.
    
    batch_size = 3
    logits = torch.tensor([[1.0], [2.0]]) # Size 2
    batched = SimpleOutput(logits=logits, profile=torch.tensor([1,2]), out_interval=None)
    
    with pytest.raises(IndexError):
        unbatch_modeloutput(batched, batch_size)

# --- Tests for _aggregate_models ---

def test_aggregate_models_mean():
    """Test averaging across models."""
    out1 = SimpleOutput(
        logits=torch.tensor([1.0, 2.0]),
        profile=torch.tensor([[1.0, 1.0]]),
        out_interval=None
    )
    out2 = SimpleOutput(
        logits=torch.tensor([3.0, 4.0]),
        profile=torch.tensor([[3.0, 3.0]]),
        out_interval=None
    )
    
    # Stack inputs as if from different models
    # The method expects list[ModelOutput]
    outputs = [out1, out2]
    
    agg = aggregate_models(outputs, method="mean")
    
    assert isinstance(agg, SimpleOutput)
    # Mean of [1,2] and [3,4] is [2,3]
    assert torch.allclose(agg.logits, torch.tensor([2.0, 3.0]))
    # Mean of [[1,1]] and [[3,3]] is [[2,2]]
    assert torch.allclose(agg.profile, torch.tensor([[2.0, 2.0]]))

def test_aggregate_models_median():
    """Test median aggregation."""
    out1 = SimpleOutput(logits=torch.tensor([1.0]), profile=torch.tensor([1.0]), out_interval=None)
    out2 = SimpleOutput(logits=torch.tensor([10.0]), profile=torch.tensor([10.0]), out_interval=None)
    out3 = SimpleOutput(logits=torch.tensor([3.0]), profile=torch.tensor([3.0]), out_interval=None)
    
    outputs = [out1, out2, out3]
    agg = aggregate_models(outputs, method="median")
    
    assert isinstance(agg, SimpleOutput)
    # Median of 1, 10, 3 is 3
    assert torch.allclose(agg.logits, torch.tensor([3.0]))

def test_aggregate_models_invalid_method():
    out1 = SimpleOutput(logits=torch.tensor([1.0]), profile=torch.tensor([1.0]), out_interval=None)
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_models([out1], method="invalid")

# --- Tests for _aggregate_tensor_track_values ---

def test_aggregate_tensor_track_values_disjoint():
    """Test aggregating two disjoint intervals."""
    # Interval 1: 0-10, Value 1
    # Interval 2: 20-30, Value 2
    # Merged: 0-30
    # Bin size: 1
    
    int1 = Interval("chr1", 0, 10, "+")
    val1 = torch.full((1, 10), 1.0) # (C, L)
    
    int2 = Interval("chr1", 20, 30, "+")
    val2 = torch.full((1, 10), 2.0)
    
    merged = Interval("chr1", 0, 30, "+")
    
    outputs = [val1, val2]
    intervals = [int1, int2]
    
    res = aggregate_tensor_track_values(
        outputs, intervals, merged, output_len=10, output_bin_size=1
    )
    
    # Result should be (C, 30)
    # 0-10: 1.0
    # 10-20: 0.0 (no coverage)
    # 20-30: 2.0
    
    assert res.shape == (1, 30)
    assert np.allclose(res[0, 0:10], 1.0)
    assert np.allclose(res[0, 10:20], 0.0)
    assert np.allclose(res[0, 20:30], 2.0)

def test_aggregate_tensor_track_values_overlap_average():
    """Test overlapping intervals are averaged."""
    # Int1: 0-20, val=2
    # Int2: 10-30, val=4
    # Overlap: 10-20 -> (2+4)/2 = 3
    
    int1 = Interval("chr1", 0, 20, "+")
    val1 = torch.full((1, 20), 2.0)
    
    int2 = Interval("chr1", 10, 30, "+")
    val2 = torch.full((1, 20), 4.0)
    
    merged = Interval("chr1", 0, 30, "+")
    
    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=20, output_bin_size=1
    )
    
    assert res.shape == (1, 30)
    assert np.allclose(res[0, 0:10], 2.0)   # Only val1
    assert np.allclose(res[0, 10:20], 3.0)  # Average
    assert np.allclose(res[0, 20:30], 4.0)  # Only val2

def test_aggregate_tensor_track_values_scalar_broadcast():
    """Test that scalar outputs (e.g. counts) are broadcast over the interval."""
    # Int1: 0-10, val=5 (scalar)
    # Int2: 5-15, val=15 (scalar)
    # Merged: 0-15
    # Overlap 5-10 -> (5+15)/2 = 10
    
    int1 = Interval("chr1", 0, 10, "+")
    val1 = torch.tensor([5.0]) # (C,)
    
    int2 = Interval("chr1", 5, 15, "+")
    val2 = torch.tensor([15.0]) # (C,)
    
    merged = Interval("chr1", 0, 15, "+")
    
    # We pretend output_len corresponds to the interval length for scalars to define 'bin count'?
    # Actually logic uses: int_bins = output_len // output_bin_size
    # So if we say output_len=10, bin_size=1, each scalar covers 10 bins starting at rel_start.
    
    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=10, output_bin_size=1
    )
    
    # res shape: 0-15 = 15 bins
    # val1 covers 0-10
    # val2 covers 5-15
    
    assert res.shape == (1, 15)
    assert np.allclose(res[0, 0:5], 5.0)
    assert np.allclose(res[0, 5:10], 10.0) # Average
    assert np.allclose(res[0, 10:15], 15.0)

# --- Tests for _aggregate_intervals ---

def test_aggregate_intervals_integration():
    """Test full aggregation of unbatched dicts."""
    # Simulate two overlapping predictions
    int1 = Interval("chr1", 0, 10, "+")
    out1 = {"logits": torch.tensor([10.0]), "profile": torch.full((1, 10), 1.0)}
    
    int2 = Interval("chr1", 5, 15, "+")
    out2 = {"logits": torch.tensor([20.0]), "profile": torch.full((1, 10), 2.0)}
    
    outputs = [out1, out2]
    intervals = [int1, int2]
    
    merged = aggregate_intervals(
        outputs, intervals, output_len=10, output_bin_size=1, output_cls=SimpleOutput
    )
    
    assert isinstance(merged, SimpleOutput)
    assert merged.out_interval is not None
    assert merged.out_interval.start == 0
    assert merged.out_interval.end == 15
    
    # Check profile (tensor track)
    # 0-5: 1.0
    # 5-10: 1.5
    # 10-15: 2.0
    prof = merged.profile.numpy()
    assert np.allclose(prof[0, 0:5], 1.0)
    assert np.allclose(prof[0, 5:10], 1.5)
    assert np.allclose(prof[0, 10:15], 2.0)
    
    # Check logits (scalar track broadcast)
    # 0-5: 10.0
    # 5-10: (10+20)/2 = 15.0
    # 10-15: 20.0
    logs = merged.logits.numpy()
    assert np.allclose(logs[0, 0:5], 10.0)
    assert np.allclose(logs[0, 5:10], 15.0)
    assert np.allclose(logs[0, 10:15], 20.0)

def test_aggregate_intervals_no_cls_raises():
    int1 = Interval("chr1", 0, 10, "+")
    out1 = {"logits": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="output_cls must be provided"):
        aggregate_intervals([out1], [int1], 10, 1)

# --- Tests for predict_intervals and predict_output_intervals ---

from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.data_config = {"input_len": 100, "output_len": 10}
    # Setup get_interval to return dummy tensors
    def get_interval_side_effect(interval):
        # Return (Channels, Length)
        return {"inputs": torch.zeros(4, 100)} 
    ds.get_interval.side_effect = get_interval_side_effect
    return ds

@pytest.fixture
def mock_ensemble():
    # We create an instance but patch the init to avoid loading
    with patch("cerberus.model_ensemble._ModelManager") as mock_manager:
        # We need to set attributes manually that init would set
        # We must initialize nn.ModuleDict properly so torch internals work
        def mock_init(self, *args, **kwargs):
            torch.nn.ModuleDict.__init__(self)
            
        with patch.object(ModelEnsemble, "__init__", mock_init):
            # Pass 5 args to match signature (checkpoint_path, model_config, data_config, genome_config, device)
            # Types are ignored since __init__ is patched
            ens = ModelEnsemble(None, None, None, None, None) # type: ignore
            ens.device = torch.device("cpu")
            ens.cerberus_config = { # type: ignore
                "data_config": {
                    "inputs": {},
                    "targets": {},
                    "input_len": 1000,
                    "output_len": 10,
                    "output_bin_size": 1,
                    "max_jitter": 0,
                    "encoding": "onehot",
                    "log_transform": False,
                    "reverse_complement": False,
                    "use_sequence": True
                }
            }
            ens.folds = []
            return ens

def test_predict_intervals_basic(mock_ensemble, mock_dataset, mock_intervals):
    """Test basic prediction flow with batching."""
    
    # Mock forward to return a batched SimpleOutput
    # We increase output_len to 100 to match input length so intervals are contiguous
    mock_ensemble.cerberus_config["data_config"]["output_len"] = 100
    # Also update dataset config so predict_intervals uses the correct length
    mock_dataset.data_config["output_len"] = 100
    
    def forward_side_effect(x, intervals, use_folds, aggregation):
        batch_size = x.shape[0]
        # Return dummy output
        return SimpleOutput(
            logits=torch.full((batch_size, 1), 1.0),
            profile=torch.full((batch_size, 1, 100), 1.0),
            out_interval=None
        )
    
    with patch.object(mock_ensemble, "forward", side_effect=forward_side_effect) as mock_forward:
        result = mock_ensemble.predict_intervals(
            mock_intervals, # 2 intervals
            mock_dataset,
            use_folds=["test"],
            aggregation="model",
            batch_size=1
        )
        
        # Check calls
        assert mock_forward.call_count == 2 # batch_size=1, 2 items
        
        # Result should be aggregated (merged intervals)
        assert isinstance(result, SimpleOutput)
        assert result.out_interval is not None
        # Intervals are centered. 
        # Int1: 100-200 (len 100) -> center 150 -> output (len 100) 100-200
        # Int2: 200-300 (len 100) -> center 250 -> output (len 100) 200-300
        # Merged: 100-300
        assert result.out_interval.start == 100
        assert result.out_interval.end == 300
        
        # Check values
        # All models returned 1.0 everywhere.
        # Intervals are now contiguous, so result should be 1.0 everywhere.
        assert torch.allclose(result.logits, torch.tensor([1.0]))
        assert torch.allclose(result.profile, torch.tensor([1.0]))

def test_predict_intervals_empty_raises(mock_ensemble, mock_dataset):
    with pytest.raises(RuntimeError, match="No results generated"):
        mock_ensemble.predict_intervals([], mock_dataset, use_folds=["test"])

def test_predict_intervals_aggregation_interval_model(mock_ensemble, mock_dataset, mock_intervals):
    """Test with aggregation='interval+model' passed to forward."""
    
    # In this mode, forward returns a MERGED output for the batch
    def forward_side_effect(x, intervals, use_folds, aggregation):
        # Mimic what forward returns for interval+model: a single merged output
        # Union of intervals in batch
        min_start = min(i.start for i in intervals)
        max_end = max(i.end for i in intervals)
        out_int = Interval("chr1", min_start, max_end, "+")
        
        # Just return dummy data
        # Profile must match interval length (200 bins) to be treated as track
        return SimpleOutput(
            logits=torch.tensor([1.0]),
            profile=torch.full((1, 200), 1.0),
            out_interval=out_int
        )

    with patch.object(mock_ensemble, "forward", side_effect=forward_side_effect):
        # Run with batch_size=2 (all in one batch)
        result = mock_ensemble.predict_intervals(
            mock_intervals,
            mock_dataset,
            use_folds=["test"],
            aggregation="interval+model",
            batch_size=2
        )
        
        assert isinstance(result, SimpleOutput)
        # Final aggregation logic is the same

def test_predict_output_intervals_tiling(mock_ensemble, mock_dataset):
    """Test tiling logic in predict_output_intervals."""
    # Target: 0-200
    # Input len: 100
    # Output len: 10
    # Stride: 50
    # Offset: (100 - 10)/2 = 45
    
    target = Interval("chr1", 0, 200, "+")
    
    # Mock predict_intervals so we don't need full chain
    # predict_intervals returns a ModelOutput
    mock_ensemble.predict_intervals = MagicMock()
    mock_ensemble.predict_intervals.return_value = SimpleOutput(
        logits=torch.tensor([1.0]), profile=torch.tensor([1.0]), out_interval=target
    )
    
    results = mock_ensemble.predict_output_intervals(
        [target],
        mock_dataset,
        stride=50,
        use_folds=["test"],
        batch_size=64
    )
    
    assert len(results) == 1
    
    # Check what predict_intervals was called with
    # First call args
    call_args = mock_ensemble.predict_intervals.call_args
    intervals_arg = call_args[0][0] # First arg is intervals list
    
    # Expected tiles:
    # Start = 0.
    # Tile 1: input_start = 0 - 45 = -45, end = 55. (Wait, negative indices? implementation allows it)
    # Stride 50.
    # Tile 2: current=50. input_start = 50 - 45 = 5. end = 105.
    # Tile 3: current=100. input_start = 100 - 45 = 55. end = 155.
    # Tile 4: current=150. input_start = 150 - 45 = 105. end = 205.
    # Tile 5: current=200. Loop condition while current < end (200 < 200 is False).
    
    assert len(intervals_arg) == 4
    assert intervals_arg[0].start == -45
    assert intervals_arg[1].start == 5
    assert intervals_arg[2].start == 55
    assert intervals_arg[3].start == 105
