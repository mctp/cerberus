import pytest
import numpy as np
import random
from collections import Counter
from cerberus.samplers import match_bin_counts

def test_match_bin_counts_1d_perfect_match():
    rng = random.Random(42)
    # Target: 10 values all 0.5 (mid bin)
    target_metrics = np.array([0.5] * 10)
    # Candidates: 10 values all 0.5
    candidate_metrics = np.array([0.5] * 10)
    
    indices = match_bin_counts(target_metrics, candidate_metrics, bins=10, match_ratio=1.0, rng=rng)
    
    assert len(indices) == 10
    # Should pick all candidates
    assert sorted(indices) == list(range(10))

def test_match_bin_counts_1d_distribution():
    rng = random.Random(42)
    # Target: 5 values in bin 0 (0.0-0.1), 5 values in bin 9 (0.9-1.0)
    target_metrics = np.array([0.05]*5 + [0.95]*5)
    
    # Candidates: 100 values in bin 0, 100 values in bin 9, 100 values in bin 5
    candidate_metrics = np.array([0.05]*100 + [0.95]*100 + [0.5]*100)
    
    indices = match_bin_counts(target_metrics, candidate_metrics, bins=10, match_ratio=1.0, rng=rng)
    
    assert len(indices) == 10
    
    selected_values = candidate_metrics[indices]
    
    # Should have 5 from bin 0 and 5 from bin 9, 0 from bin 5
    # Floating point comparison needs care, but exact values were used
    counts_low = np.sum(selected_values == 0.05)
    counts_high = np.sum(selected_values == 0.95)
    counts_mid = np.sum(selected_values == 0.5)
    
    assert counts_low == 5
    assert counts_high == 5
    assert counts_mid == 0

def test_match_bin_counts_2d():
    rng = random.Random(42)
    # Target: 10 points at (0.1, 0.1)
    target_metrics = np.array([[0.1, 0.1]] * 10)
    
    # Candidates:
    # 20 points at (0.1, 0.1) -> Match
    # 20 points at (0.9, 0.9) -> Ignore
    candidate_metrics = np.vstack([
        [[0.1, 0.1]] * 20,
        [[0.9, 0.9]] * 20
    ])
    
    indices = match_bin_counts(target_metrics, candidate_metrics, bins=10, match_ratio=1.0, rng=rng)
    
    assert len(indices) == 10
    # Indices should be from first 20
    assert all(i < 20 for i in indices)

def test_match_bin_counts_ratio():
    rng = random.Random(42)
    target_metrics = np.array([0.5] * 10)
    candidate_metrics = np.array([0.5] * 100)
    
    indices = match_bin_counts(target_metrics, candidate_metrics, bins=10, match_ratio=2.0, rng=rng)
    
    assert len(indices) == 20

def test_match_bin_counts_nan_handling():
    rng = random.Random(42)
    target_metrics = np.array([0.5, np.nan, 0.5])
    candidate_metrics = np.array([0.5, 0.5, np.nan])
    
    # 2 valid targets (0.5). 1 ignored.
    # 2 valid candidates (0.5). 1 ignored.
    
    indices = match_bin_counts(target_metrics, candidate_metrics, bins=10, match_ratio=1.0, rng=rng)
    
    # Should match 2
    assert len(indices) == 2
    # Indices 0 and 1 from candidates correspond to 0.5
    assert sorted(indices) == [0, 1]
