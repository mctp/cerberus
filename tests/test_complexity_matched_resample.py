
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cerberus.samplers import ComplexityMatchedSampler, Interval, ListSampler


@pytest.fixture
def mock_fasta(tmp_path):
    # Minimal mock since we patch compute_intervals_complexity
    fasta_path = tmp_path / "genome.fa"
    fasta_path.touch()
    return fasta_path

def test_complexity_matched_resample_static_candidates(mock_fasta):
    """
    Verifies that calling resample() on ComplexityMatchedSampler does NOT
    trigger resample() on the candidate sampler (except during initialization).
    This ensures the candidate pool remains static for performance.
    """
    chrom_sizes = {"chr1": 2000}
    target_sampler = ListSampler(
        intervals=[Interval("chr1", 0, 100, "+")], 
        chrom_sizes=chrom_sizes
    )
    
    # Create a mock candidate sampler
    candidate_sampler = MagicMock()
    candidate_sampler.chrom_sizes = chrom_sizes
    candidate_sampler.folds = []
    candidate_sampler.exclude_intervals = {}
    candidate_sampler.__len__.return_value = 100
    candidate_sampler.__iter__.return_value = iter([Interval("chr1", i*10, i*10+10, "+") for i in range(100)])
    candidate_sampler.__getitem__.side_effect = lambda i: Interval("chr1", i*10, i*10+10, "+")
    
    # Mock complexity computation to return random values (matching input length)
    with patch("cerberus.samplers.compute_intervals_complexity") as mock_compute:
        mock_compute.side_effect = lambda intervals, *a, **kw: np.random.rand(len(list(intervals)), 1)
        
        sampler = ComplexityMatchedSampler(
            target_sampler=target_sampler,
            candidate_sampler=candidate_sampler,
            fasta_path=mock_fasta,
            chrom_sizes=chrom_sizes,
            metrics=["gc"],
            seed=42
        )
        
        # Initial call count (from init)
        initial_count = candidate_sampler.resample.call_count
        assert initial_count == 1
        
        # Resample
        sampler.resample(seed=None)
        
        # Count should NOT increase
        assert candidate_sampler.resample.call_count == initial_count

def test_complexity_matched_fallback(mock_fasta):
    """
    Verifies that if no candidates match the target bin, the sampler falls back
    to random sampling from the candidate pool.
    """
    chrom_sizes = {"chr1": 2000}
    
    # Target: High complexity (mocked to return 1.0)
    target_sampler = ListSampler(
        intervals=[Interval("chr1", 0, 100, "+")], 
        chrom_sizes=chrom_sizes
    )
    
    # Candidates: Low complexity (mocked to return 0.0)
    candidate_intervals = [Interval("chr1", 100, 200, "+"), Interval("chr1", 200, 300, "+")]
    candidate_sampler = ListSampler(intervals=candidate_intervals, chrom_sizes=chrom_sizes)
    
    with patch("cerberus.samplers.compute_intervals_complexity") as mock_compute:
        def side_effect(intervals, fasta, metrics, center_size=None):
            ivs = list(intervals)
            # Check if these are target intervals (high complexity) or candidate (low)
            if len(ivs) == 1 and ivs[0].start == 0:
                return np.array([[0.9]]) # High
            else:
                return np.array([[0.1]] * len(ivs)) # Low
        mock_compute.side_effect = side_effect
        
        sampler = ComplexityMatchedSampler(
            target_sampler=target_sampler,
            candidate_sampler=candidate_sampler,
            fasta_path=mock_fasta,
            chrom_sizes=chrom_sizes,
            metrics=["gc"],
            bins=2, # [0-0.5, 0.5-1.0]
            candidate_ratio=1.0,
            seed=42
        )
        
        # Target is in bin 1 (0.5-1.0). Candidates are in bin 0 (0-0.5).
        # No match. Should fallback to random choice from candidates.
        
        assert len(sampler) == 1
        selected = sampler[0]
        # It must be one of the candidates
        assert selected in candidate_intervals
