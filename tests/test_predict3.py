import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, Mock
import numpy as np

from cerberus.predict3 import predict_interval, predict_intervals
from cerberus.interval import Interval

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
        return out

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
        return torch.ones(x.shape[0], 1) * self.value

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

# --- Existing Tests for predict_interval ---

def test_predict_interval_validation(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 50) # Wrong length
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    with pytest.raises(ValueError, match="length 50, expected 100"):
        predict_interval(interval, mock_dataset, mock_model_manager, config, device="cpu")

def test_predict_interval_single_model(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    # output_len=60 to match dataset config
    model = MockModel(value=2.0, output_len=60)
    mock_model_manager.get_models.return_value = [model]
    
    output, out_interval = predict_interval(interval, mock_dataset, mock_model_manager, config, device="cpu")
    
    # Output should be length 60
    assert output.shape[-1] == 60
    assert np.allclose(output, np.ones((4, 60)) * 2.0)
    assert out_interval.start == 20
    assert out_interval.end == 80

def test_predict_interval_mean_aggregation(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    model1 = MockModel(value=2.0, output_len=60)
    model2 = MockModel(value=4.0, output_len=60)
    mock_model_manager.get_models.return_value = [model1, model2]
    
    output, out_interval = predict_interval(interval, mock_dataset, mock_model_manager, config, device="cpu")
    
    assert output.shape[-1] == 60
    assert np.allclose(output, np.ones((4, 60)) * 3.0)
    assert out_interval.start == 20

def test_predict_interval_tuple_output(mock_dataset, mock_model_manager):
    interval = Interval("chr1", 0, 100)
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    model1 = MockTupleModel(value1=2.0, value2=10.0, output_len=60)
    model2 = MockTupleModel(value1=4.0, value2=20.0, output_len=60)
    mock_model_manager.get_models.return_value = [model1, model2]
    
    output, out_interval = predict_interval(interval, mock_dataset, mock_model_manager, config, device="cpu")
    
    assert isinstance(output, tuple)
    assert output[0].shape[-1] == 60
    assert np.allclose(output[0], np.ones((4, 60)) * 3.0)
    assert np.allclose(output[1], np.ones((4, 60)) * 15.0)

# --- New Tests for predict_intervals ---

def test_predict_intervals_overlap(mock_dataset, mock_model_manager):
    # Output is length 60, offset 20.
    # Int1: 0-100 -> Output 20-80
    # Int2: 10-110 -> Output 30-90
    interval_1 = Interval("chr1", 0, 100)
    interval_2 = Interval("chr1", 10, 110)
    
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    # Mock model manager to return models that output constant 1 for Int1 and 2 for Int2
    # But get_models is called per interval.
    # We can use side_effect or just return a model that outputs 1 everywhere, 
    # but here we want different values to test averaging.
    # Let's mock predict_interval logic partially? No, let's just make the model return 1.
    # Wait, if I want to test averaging, I need different values.
    # I can mock dataset.get_interval to return different inputs for different intervals?
    # Int1 -> input 1s -> Model(x)=x -> output 1s
    # Int2 -> input 2s -> Model(x)=x -> output 2s
    
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
            # Input is (B, C, 100). We return (B, C, 60)
            # Just return ones * value. 
            # Note: get_interval returns 1s or 2s.
            # But here we just want to output 1s or 2s based on input value?
            # x has values 1 or 2.
            # return x[:, :, 20:80] ?
            # Yes, crop center.
            return x[:, :, 20:80] * self.value

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
    assert arr.shape == (1, 70)
    
    # [20, 30): From Int1 (val 1) -> Indices 0-10
    # [30, 80): Overlap (val 1 and 2) -> Indices 10-60 -> (1+2)/2 = 1.5
    # [80, 90): From Int2 (val 2) -> Indices 60-70
    
    assert np.allclose(arr[0, 0:10], 1.0)
    assert np.allclose(arr[0, 10:60], 1.5)
    assert np.allclose(arr[0, 60:70], 2.0)

def test_predict_intervals_scalar_broadcast(mock_dataset, mock_model_manager):
    # Scalar output
    interval_1 = Interval("chr1", 0, 100)
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    model = MockScalarModel(value=5.0)
    mock_model_manager.get_models.return_value = [model]
    
    results = predict_intervals([interval_1], mock_dataset, mock_model_manager, config, device="cpu")
    
    arr, merged_interval = results
    
    # Output length 60. Scalar should be broadcast to 60.
    assert arr.shape == (1, 60)
    assert np.allclose(arr, 5.0)
    assert merged_interval.start == 20
    assert merged_interval.end == 80

def test_predict_intervals_tuple_recursive(mock_dataset, mock_model_manager):
    # Tuple output (Profile, Scalar)
    # Output length 60 (for profile). Scalar broadcast.
    mock_dataset.data_config["output_bin_size"] = 1
    
    interval_1 = Interval("chr1", 0, 100)
    config = {"use_folds": ["test"], "aggregation": "mean"}
    
    # Profile: length 100 (input) -> cropped? 
    # MockTupleModel returns input size.
    # But predict_interval logic assumes model returns valid output.
    # Our MockTupleModel returns (x*v1, x*v2).
    # x is 100 long.
    # If output_len is 60, our mock model returns 100.
    # predict_intervals helper checks dimensions.
    # 100 != 60. So it might treat it as scalar? 
    # 100 * 1 != 60.
    # So it treats as scalar and broadcasts to 60?
    # This is tricky.
    # We should ensure our mock model returns output_len if we want it treated as profile.
    
    class MockCorrectProfileModel(nn.Module):
        def forward(self, x):
            # Return (Profile 60, Scalar)
            prof = torch.ones(x.shape[0], 1, 60) * 3.0
            scalar = torch.ones(x.shape[0], 1) * 7.0
            return (prof, scalar)

    model = MockCorrectProfileModel()
    mock_model_manager.get_models.return_value = [model]
    
    results = predict_intervals([interval_1], mock_dataset, mock_model_manager, config, device="cpu")
    
    # Result should be tuple (Tuple[Array, Array], Interval)
    # Because predict_intervals returns (aggregated_values, merged_interval)
    # aggregated_values mimics the recursive structure: (prof_array, scalar_array)
    
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
