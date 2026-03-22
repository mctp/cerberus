import numpy as np
import torch

from cerberus.interval import Interval
from cerberus.transform import Jitter


def test_jitter_distribution():
    """
    Verify that Jitter produces random offsets within the expected range
    and that the distribution is roughly uniform (or at least covers the range).
    """
    # Setup
    input_len = 50
    # Slack needs to allow jitter. 
    # Total slack = input_tensor_len - input_len
    # Let's use input_tensor_len = 100. Slack = 50.
    # Center of slack is 25.
    # If max_jitter = 5, start range is [20, 30].
    # Range width = 11 values (20 to 30 inclusive).
    
    max_jitter = 5
    tensor_len = 100
    jitter = Jitter(input_len=input_len, max_jitter=max_jitter)
    
    # Create input tensor with unique values to identify start position
    inputs = torch.arange(tensor_len).unsqueeze(0).float() # (1, 100)
    targets = inputs.clone()
    
    start_indices = []
    num_trials = 1000
    
    dummy_interval = Interval("chr1", 0, 1000)
    
    for _ in range(num_trials):
        out_in, _, _ = jitter(inputs, targets, dummy_interval)
        first_val = out_in[0, 0].item()
        start_idx = int(first_val)
        start_indices.append(start_idx)
        
    # 1. Check Bounds
    # Expected range: center - max_jitter to center + max_jitter
    slack = tensor_len - input_len # 50
    center = slack // 2 # 25
    min_start = center - max_jitter # 20
    max_start = center + max_jitter # 30
    
    start_indices = np.array(start_indices)
    assert np.all(start_indices >= min_start), f"Found start index {start_indices.min()} < {min_start}"
    assert np.all(start_indices <= max_start), f"Found start index {start_indices.max()} > {max_start}"
    
    # 2. Check Coverage/Randomness
    # We expect to see all values in [20, 30] roughly equally.
    unique_starts = np.unique(start_indices)
    expected_starts = np.arange(min_start, max_start + 1)
    
    # Should check that we saw all possible start positions
    assert len(unique_starts) == len(expected_starts)
    assert np.array_equal(unique_starts, expected_starts)
    
    # 3. Check for Uniformity (Basic Chi-Squared or just range check)
    # With 1000 trials and 11 bins, expected count ~90 per bin.
    # Just check that no bin is empty (already done) or extremely rare.
    counts = np.bincount(start_indices)[min_start:max_start+1]
    assert np.all(counts > 0)
    
    # Optional: Check variance
    # Variance of uniform distribution U[a,b] is (n^2 - 1)/12 where n = b-a+1
    # n=11. Var = 120/12 = 10.
    # StdDev = sqrt(10) ~ 3.16
    
    std_dev = np.std(start_indices)
    assert 2.5 < std_dev < 3.8 # Approximate check

def test_jitter_full_slack():
    """
    Test Jitter when max_jitter is None (full slack usage).
    """
    input_len = 50
    tensor_len = 100
    jitter = Jitter(input_len=input_len, max_jitter=None)
    
    inputs = torch.arange(tensor_len).unsqueeze(0).float()
    targets = inputs.clone()
    
    start_indices = []
    dummy_interval = Interval("chr1", 0, 1000)
    for _ in range(500):
        out_in, _, _ = jitter(inputs, targets, dummy_interval)
        start_indices.append(int(out_in[0, 0].item()))
        
    start_indices = np.array(start_indices)
    
    # Bounds: 0 to slack (50)
    assert start_indices.min() >= 0
    assert start_indices.max() <= 50
    
    # Should cover full range (roughly)
    # Range is 0 to 50 inclusive (51 values). 500 trials might not hit every single one, 
    # but range should be close to 0-50.
    assert start_indices.min() < 5
    assert start_indices.max() > 45
    
    # Check randomness (not constant)
    assert len(np.unique(start_indices)) > 20
