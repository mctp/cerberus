from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cerberus.interval import Interval
from cerberus.signal import SignalExtractor


@pytest.fixture
def mock_pybigtools():
    with patch("cerberus.signal.pybigtools") as mock:
        yield mock

def test_signal_extractor_mocked(mock_pybigtools):
    # Setup mock bigwig file
    mock_bw = MagicMock()
    mock_pybigtools.open.return_value = mock_bw
    
    # Mock values
    # Return 1.0s
    mock_bw.values.return_value = np.ones(100, dtype=np.float32)
    
    from pathlib import Path
    paths = {"s1": Path("path/to/s1.bw")}
    extractor = SignalExtractor(paths)
    
    interval = Interval("chr1", 0, 100)
    
    signal = extractor.extract(interval)
    
    assert signal.shape == (1, 100)
    assert torch.all(signal == 1.0)
    
    mock_pybigtools.open.assert_called_with("path/to/s1.bw")
    mock_bw.values.assert_called_with("chr1", 0, 100)

def test_signal_extractor_multiple_channels(mock_pybigtools):
    # Setup mock bigwig files
    mock_bw1 = MagicMock()
    mock_bw2 = MagicMock()
    
    # Side effect for open to return different mocks
    def open_side_effect(path):
        if path == "s1.bw": return mock_bw1
        if path == "s2.bw": return mock_bw2
        return MagicMock()
        
    mock_pybigtools.open.side_effect = open_side_effect
    
    mock_bw1.values.return_value = np.ones(10) # 1s
    mock_bw2.values.return_value = np.zeros(10) # 0s
    
    from pathlib import Path
    paths = {"c1": Path("s1.bw"), "c2": Path("s2.bw")}
    extractor = SignalExtractor(paths)
    
    interval = Interval("chr1", 0, 10)
    signal = extractor.extract(interval)
    
    assert signal.shape == (2, 10)
    assert torch.all(signal[0] == 1.0)
    assert torch.all(signal[1] == 0.0)

def test_signal_extractor_padding(mock_pybigtools):
    mock_bw = MagicMock()
    mock_pybigtools.open.return_value = mock_bw
    
    # Return 50 values, but request 100
    mock_bw.values.return_value = np.ones(50, dtype=np.float32)
    
    from pathlib import Path
    extractor = SignalExtractor({"s1": Path("test.bw")})
    interval = Interval("chr1", 0, 100)
    
    signal = extractor.extract(interval)
    
    assert signal.shape == (1, 100)
    # First 50 should be 1
    assert torch.all(signal[0, :50] == 1.0)
    # Rest should be 0 (padding)
    assert torch.all(signal[0, 50:] == 0.0)

def test_signal_extractor_error_handling(mock_pybigtools):
    mock_bw = MagicMock()
    mock_pybigtools.open.return_value = mock_bw
    
    # Raise runtime error (e.g. chrom not found)
    mock_bw.values.side_effect = RuntimeError("Chrom not found")
    
    from pathlib import Path
    extractor = SignalExtractor({"s1": Path("test.bw")})
    interval = Interval("chr1", 0, 100)
    
    signal = extractor.extract(interval)
    
    # Should return zeros
    assert signal.shape == (1, 100)
    assert torch.all(signal == 0.0)

def test_signal_extractor_open_error(mock_pybigtools):
    mock_pybigtools.open.side_effect = Exception("Open failed")
    
    from pathlib import Path
    extractor = SignalExtractor({"s1": Path("test.bw")})
    
    # Lazy load happens on extract
    with pytest.raises(RuntimeError, match="Failed to open BigWig file"):
        extractor.extract(Interval("chr1", 0, 100))

def test_signal_extractor_base_exception_propagates(mock_pybigtools):
    """BaseException subclasses (e.g. pyo3 Rust panics) should propagate
    rather than being silently swallowed.  Only Exception subclasses are
    caught and result in zeros."""
    mock_bw = MagicMock()
    mock_pybigtools.open.return_value = mock_bw

    class FakePanicException(BaseException):
        pass

    mock_bw.values.side_effect = FakePanicException("called Result::unwrap() on an Err value: BadData")

    from pathlib import Path
    extractor = SignalExtractor({"s1": Path("test.bw")})
    interval = Interval("chr1", 0, 100)

    with pytest.raises(FakePanicException):
        extractor.extract(interval)


def test_signal_extractor_nan_inf_handling(mock_pybigtools):
    mock_bw = MagicMock()
    mock_pybigtools.open.return_value = mock_bw
    
    # Return NaN and Inf values
    # Note: np.nan_to_num converts Inf to MAX_FLOAT by default.
    # We test current behavior.
    mock_bw.values.return_value = np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32)
    
    from pathlib import Path
    extractor = SignalExtractor({"s1": Path("test.bw")})
    interval = Interval("chr1", 0, 4)
    
    signal = extractor.extract(interval)
    
    assert signal.shape == (1, 4)
    
    # 0: NaN -> 0
    assert signal[0, 0] == 0.0
    # 1: 1.0 -> 1.0
    assert signal[0, 1] == 1.0
    # 2: Inf -> Large Number (checking it's not Inf)
    assert not torch.isinf(signal[0, 2])
    assert signal[0, 2] > 1e30
    # 3: -Inf -> Small Number (checking it's not -Inf)
    assert not torch.isinf(signal[0, 3])
    assert signal[0, 3] < -1e30
