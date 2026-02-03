import pytest
import random
from pathlib import Path
from cerberus.samplers import (
    generate_sub_seeds,
    RandomSampler,
    ComplexityMatchedSampler,
)
from cerberus.interval import Interval

def test_generate_sub_seeds_uniqueness():
    """Verify that sub-seeds are distinct and deterministic."""
    seed = 42
    seeds1 = generate_sub_seeds(seed, 5)
    seeds2 = generate_sub_seeds(seed, 5)
    
    assert len(seeds1) == 5
    assert len(set(seeds1)) == 5, "Sub-seeds should be unique"
    assert seeds1 == seeds2, "Sub-seed generation should be deterministic"
    assert seeds1 != generate_sub_seeds(seed + 1, 5), "Different master seeds should produce different sub-seeds"

def test_generate_sub_seeds_none():
    """Verify behavior with None seed."""
    seeds = generate_sub_seeds(None, 5)
    assert seeds == [None] * 5

def test_random_sampler_resample():
    """Verify RandomSampler regenerates intervals on resample."""
    chrom_sizes = {"chr1": 1000}
    
    rs = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=10,
        num_intervals=5,
        seed=42
    )
    intervals1 = list(rs)
    
    # Resample with same seed -> same intervals
    rs.resample(42)
    intervals2 = list(rs)
    assert intervals1 == intervals2
    
    # Resample with different seed -> different intervals
    rs.resample(43)
    intervals3 = list(rs)
    assert intervals1 != intervals3
    
    # Resample with None -> should advance RNG
    rs.resample(None)
    intervals4 = list(rs)
    assert intervals3 != intervals4

def test_random_sampler_resample_clears_list():
    """Verify that resample clears the previous list of intervals."""
    chrom_sizes = {"chr1": 1000}
    rs = RandomSampler(chrom_sizes, 10, 5, seed=42)
    assert len(rs) == 5
    
    rs.resample(43)
    assert len(rs) == 5

def test_complexity_matched_sampler_propagation():
    """
    Verify that ComplexityMatchedSampler propagates resampling to its candidate_sampler.
    """
    class MockCandidateSampler:
        def __init__(self):
            self.seed_received = None
            self.chrom_sizes = {}
            self.exclude_intervals = {}
            self.folds = []
        
        def resample(self, seed: int | None = None):
            self.seed_received = seed
            
        def __iter__(self): yield from []
        def __len__(self): return 0
        def __getitem__(self, idx): raise IndexError
        def split_folds(self, t, v): return (self, self, self)

    # Subclass ComplexityMatchedSampler to mock dependencies
    class TestComplexityMatchedSampler(ComplexityMatchedSampler):
        def __init__(self, candidate_sampler, seed):
            self.candidate_sampler = candidate_sampler
            self.seed = seed
            self.match_ratio = 1.0
            self.bins = 100
            self.rng = random.Random(seed)
            self.target_metrics = []
            self.candidate_metrics = []
            self.fasta_path = "dummy.fa" 
            self.metrics = ["gc"]
            # Bypass heavy init of base class
            
    mock_candidate = MockCandidateSampler()
    sampler = TestComplexityMatchedSampler(mock_candidate, seed=42)
    
    # Patch compute_intervals_complexity to avoid actual file IO
    from unittest.mock import patch
    with patch("cerberus.samplers.compute_intervals_complexity") as mock_gc:
        mock_gc.return_value = []
        
        # Call resample
        sampler.resample(123)
    
        # Check if candidate received a seed
        assert mock_candidate.seed_received is not None
        # Ensure it's not 123 (should be derived)
        assert mock_candidate.seed_received != 123
        
        # Check if deterministic
        sampler.resample(123)
        seed_1 = mock_candidate.seed_received
        
        sampler.resample(123)
        seed_2 = mock_candidate.seed_received
        
        assert seed_1 == seed_2

def test_random_sampler_lazy_init():
    """Verify RandomSampler respects generate_on_init=False."""
    chrom_sizes = {"chr1": 1000}
    
    # Deferred init
    rs = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=10,
        num_intervals=5,
        seed=42,
        generate_on_init=False
    )
    
    # Should be empty initially
    assert len(rs) == 0
    assert rs._intervals == []
    
    # Resample should populate it
    rs.resample(42)
    assert len(rs) == 5
    assert len(rs._intervals) == 5
