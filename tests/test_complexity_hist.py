import numpy as np
import pytest
from cerberus.complexity import get_bin_index, compute_hist

def test_get_bin_index_1d():
    """Test get_bin_index with 1D input (single metric)."""
    # Value 0.0 -> bin 0
    res = get_bin_index(np.array([0.0]), 10)
    assert res == (0,)
    
    # Value 0.5 -> bin 5
    res = get_bin_index(np.array([0.5]), 10)
    assert res == (5,)
    
    # Value 0.99 -> bin 9
    res = get_bin_index(np.array([0.99]), 10)
    assert res == (9,)
    
    # Value 1.0 -> bin 9 (clipped)
    res = get_bin_index(np.array([1.0]), 10)
    assert res == (9,)
    
    # Value > 1.0 -> bin 9 (clipped)
    res = get_bin_index(np.array([1.5]), 10)
    assert res == (9,)

def test_get_bin_index_nd():
    """Test get_bin_index with multidimensional input."""
    # (0.1, 0.9) with 10 bins -> (1, 9)
    res = get_bin_index(np.array([0.1, 0.9]), 10)
    assert res == (1, 9)

def test_get_bin_index_nan():
    """Test that get_bin_index returns None for inputs containing NaN."""
    res = get_bin_index(np.array([np.nan]), 10)
    assert res is None
    
    res = get_bin_index(np.array([0.5, np.nan]), 10)
    assert res is None

def test_compute_hist_1d():
    """Test compute_hist with 1D metrics array."""
    metrics = np.array([0.1, 0.1, 0.5, 0.9])
    # Bins=10: 0.1->1, 0.5->5, 0.9->9
    hist = compute_hist(metrics, 10)
    
    assert len(hist) == 3
    assert hist[(1,)] == 2
    assert hist[(5,)] == 1
    assert hist[(9,)] == 1

def test_compute_hist_nd():
    """Test compute_hist with multidimensional metrics array."""
    # 3 samples, 2 dimensions
    metrics = np.array([
        [0.1, 0.1], # (1, 1)
        [0.1, 0.1], # (1, 1)
        [0.5, 0.5]  # (5, 5)
    ])
    hist = compute_hist(metrics, 10)
    
    assert len(hist) == 2
    assert hist[(1, 1)] == 2
    assert hist[(5, 5)] == 1

def test_compute_hist_nan():
    """Test that compute_hist ignores rows with NaN values."""
    metrics = np.array([
        [0.1, 0.1],
        [np.nan, 0.5], # Should be ignored
        [0.5, 0.5]
    ])
    hist = compute_hist(metrics, 10)
    
    assert len(hist) == 2
    assert hist[(1, 1)] == 1
    assert hist[(5, 5)] == 1
