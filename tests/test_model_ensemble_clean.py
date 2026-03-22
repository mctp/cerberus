import dataclasses

import numpy as np
import pytest
import torch

from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    ModelOutput,
    aggregate_intervals,
    aggregate_models,
    aggregate_tensor_track_values,
    unbatch_modeloutput,
)

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
    
    # Scalar inputs are collapsed back to (C,) after overlap-weighted averaging.
    # Bins 0-4: 5.0, bins 5-9: mean(5,15)=10.0, bins 10-14: 15.0
    # Overall mean: (5*5 + 10*5 + 15*5) / 15 = 10.0
    assert res.shape == (1,)
    assert np.allclose(res[0], 10.0)

def test_aggregate_tensor_track_values_scalar_no_dilution_from_gaps():
    """Regression: scalar averaging must not be diluted by empty bins in gaps.

    Bug: when intervals have gaps (e.g., blacklist regions), the merged_interval
    spans the full range including the gap. The gap bins have accumulator=0 and
    counts=0 (clamped to 1). A naive .mean(axis=-1) averages in these zeros,
    diluting the scalar value proportional to the gap size.

    Fix: only average over bins that actually received contributions.
    """
    # Two disjoint intervals with a gap in between
    int1 = Interval("chr1", 0, 10, "+")
    val1 = torch.tensor([10.0])  # (C,)

    int2 = Interval("chr1", 90, 100, "+")
    val2 = torch.tensor([10.0])  # (C,)

    # Merged interval spans 0-100, but only 20 of 100 bins have data
    merged = Interval("chr1", 0, 100, "+")

    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=10, output_bin_size=1
    )

    # Should be 10.0 (average of valid bins only), not 2.0 (diluted by 80 empty bins)
    assert res.shape == (1,)
    assert np.allclose(res[0], 10.0), (
        f"Scalar should be 10.0 (no dilution), got {res[0]:.2f}"
    )

def test_aggregate_tensor_track_values_single_interval_passthrough():
    """Single interval: profile passes through unchanged (no averaging)."""
    interval = Interval("chr1", 10, 30, "+")
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]])
    merged = Interval("chr1", 10, 30, "+")

    res = aggregate_tensor_track_values(
        [values], [interval], merged, output_len=20, output_bin_size=1
    )

    assert res.shape == (1, 20)
    assert np.allclose(res, values.numpy())

def test_aggregate_tensor_track_values_multichannel():
    """Multi-channel profiles are aggregated independently per channel."""
    int1 = Interval("chr1", 0, 10, "+")
    val1 = torch.tensor([[1.0] * 10, [10.0] * 10])  # (2, 10)

    int2 = Interval("chr1", 5, 15, "+")
    val2 = torch.tensor([[3.0] * 10, [30.0] * 10])  # (2, 10)

    merged = Interval("chr1", 0, 15, "+")

    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=10, output_bin_size=1
    )

    assert res.shape == (2, 15)
    # Channel 0: overlap region [5,10) averages to 2.0
    assert np.allclose(res[0, 0:5], 1.0)
    assert np.allclose(res[0, 5:10], 2.0)
    assert np.allclose(res[0, 10:15], 3.0)
    # Channel 1: overlap region averages to 20.0
    assert np.allclose(res[1, 0:5], 10.0)
    assert np.allclose(res[1, 5:10], 20.0)
    assert np.allclose(res[1, 10:15], 30.0)

def test_aggregate_tensor_track_values_bin_size_gt1():
    """output_bin_size > 1: intervals snap to bin grid."""
    # bin_size=4, interval 0-8 → 2 bins, interval 4-12 → 2 bins
    int1 = Interval("chr1", 0, 8, "+")
    val1 = torch.tensor([[1.0, 2.0]])  # (1, 2)

    int2 = Interval("chr1", 4, 12, "+")
    val2 = torch.tensor([[3.0, 4.0]])  # (1, 2)

    merged = Interval("chr1", 0, 12, "+")

    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=2, output_bin_size=4
    )

    # 3 bins: bin0=[0,4), bin1=[4,8), bin2=[8,12)
    assert res.shape == (1, 3)
    assert np.allclose(res[0, 0], 1.0)         # only val1
    assert np.allclose(res[0, 1], (2.0 + 3.0) / 2)  # overlap
    assert np.allclose(res[0, 2], 4.0)         # only val2

def test_aggregate_tensor_track_values_scalar_all_empty():
    """Scalar path with no valid bins returns zeros."""
    # Merged interval has 10 bins but intervals are outside it (edge case)
    # Use a merged interval that doesn't overlap with any interval
    int1 = Interval("chr1", 100, 110, "+")
    val1 = torch.tensor([5.0])  # (C,)

    # But merged interval is 0-10 — interval is outside
    merged = Interval("chr1", 0, 10, "+")

    res = aggregate_tensor_track_values(
        [val1], [int1], merged, output_len=10, output_bin_size=1
    )

    # The scalar goes into bins 100-109 relative to merged start 0,
    # which is outside the 10-bin accumulator → no valid bins
    assert res.shape == (1,)
    assert np.allclose(res[0], 0.0)

def test_aggregate_tensor_track_values_scalar_multichannel():
    """Multi-channel scalars are averaged independently per channel."""
    int1 = Interval("chr1", 0, 10, "+")
    val1 = torch.tensor([5.0, 50.0])  # (2,)

    int2 = Interval("chr1", 5, 15, "+")
    val2 = torch.tensor([15.0, 150.0])  # (2,)

    merged = Interval("chr1", 0, 15, "+")

    res = aggregate_tensor_track_values(
        [val1, val2], [int1, int2], merged, output_len=10, output_bin_size=1
    )

    assert res.shape == (2,)
    # Same math as single-channel: (5*5 + 10*5 + 15*5)/15 = 10.0
    assert np.allclose(res[0], 10.0)
    assert np.allclose(res[1], 100.0)

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
    
    # Check logits (scalar track — collapsed back to (C,) after overlap averaging)
    # Window 1 covers bins 0-10 with value 10.0, window 2 covers 5-15 with 20.0.
    # Overlap-weighted average: (10*10 + 20*10) / (10+10) = 15.0
    logs = merged.logits.numpy()
    assert logs.shape == (1,), f"Scalar should collapse to (C,), got {logs.shape}"
    assert np.allclose(logs[0], 15.0)

def test_aggregate_intervals_no_cls_raises():
    int1 = Interval("chr1", 0, 10, "+")
    out1 = {"logits": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="output_cls must be provided"):
        aggregate_intervals([out1], [int1], 10, 1)

# --- Tests for predict_intervals and predict_output_intervals ---

from unittest.mock import MagicMock, patch

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)


def _make_data_config(input_len=100, output_len=10, output_bin_size=1):
    return DataConfig.model_construct(
        inputs={}, targets={}, input_len=input_len, output_len=output_len,
        output_bin_size=output_bin_size, max_jitter=0, encoding="onehot",
        log_transform=False, reverse_complement=False, target_scale=1.0, use_sequence=True,
    )

def _make_cerberus_config(output_len=10, output_bin_size=1):
    return CerberusConfig.model_construct(
        data_config=_make_data_config(input_len=1000, output_len=output_len, output_bin_size=output_bin_size),
        genome_config=GenomeConfig.model_construct(
            name="mock", fasta_path="mock.fa", chrom_sizes={"chr1": 1000000},
            allowed_chroms=["chr1"], exclude_intervals={}, fold_type="chrom_partition",
            fold_args={"k": 5, "test_fold": None, "val_fold": None},
        ),
        sampler_config=SamplerConfig.model_construct(
            sampler_type="random", padded_size=100,
            sampler_args={"num_intervals": 10},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1, max_epochs=1, learning_rate=1e-3, weight_decay=0.0,
            patience=5, optimizer="adam", scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0, adam_eps=1e-8,
            gradient_clip_val=None,
        ),
        model_config_=ModelConfig.model_construct(
            name="mock", model_cls="torch.nn.Linear", loss_cls="torch.nn.MSELoss",
            loss_args={}, metrics_cls="torchmetrics.MeanSquaredError", metrics_args={},
            model_args={}, pretrained=[], count_pseudocount=0.0,
        ),
    )

@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.data_config = _make_data_config(input_len=100, output_len=10)
    def get_interval_side_effect(interval):
        return {"inputs": torch.zeros(4, 100)}
    ds.get_interval.side_effect = get_interval_side_effect
    return ds

@pytest.fixture
def mock_ensemble():
    with patch("cerberus.model_ensemble._ModelManager"):
        def mock_init(self, *args, **kwargs):
            torch.nn.ModuleDict.__init__(self)

        with patch.object(ModelEnsemble, "__init__", mock_init):
            ens = ModelEnsemble(None, None, None, None, None) # type: ignore
            ens.device = torch.device("cpu")
            ens.cerberus_config = _make_cerberus_config(output_len=10, output_bin_size=1)
            ens.folds = []
            return ens

def test_predict_intervals_basic(mock_ensemble, mock_dataset, mock_intervals):
    """Test basic prediction flow with batching."""

    # We increase output_len to 100 to match input length so intervals are contiguous
    mock_ensemble.cerberus_config = _make_cerberus_config(output_len=100, output_bin_size=1)
    mock_dataset.data_config = _make_data_config(input_len=100, output_len=100)
    
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
