import pytest
from pathlib import Path
import random
import numpy as np
from cerberus.samplers import ComplexityMatchedSampler, ListSampler, Interval
from interlap import InterLap

@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    
    # Simple homogeneous sequence
    seq = "G" * 1000
    
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq + "\n")
        
    # Index it
    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))
    
    return fasta_path

def test_complexity_matched_exclusions(mock_fasta):
    """
    Ensure ComplexityMatchedSampler explicitly filters out excluded regions,
    even if the candidate sampler provides them.
    
    This verifies that candidates are checked against the sampler's exclude_intervals.
    """
    chrom_sizes = {"chr1": 1000}
    
    # Target: 1 interval
    target_intervals = [Interval("chr1", 0, 100, "+")]
    target_sampler = ListSampler(intervals=target_intervals, chrom_sizes=chrom_sizes)
    
    # Candidate: 2 intervals, one valid, one excluded
    # Both are on chr1 which matches target (and is homogeneous in mock_fasta)
    candidate_intervals = [
        Interval("chr1", 100, 200, "+"),  # Valid
        Interval("chr1", 200, 300, "+")   # Excluded
    ]
    candidate_sampler = ListSampler(intervals=candidate_intervals, chrom_sizes=chrom_sizes)
    
    # Define exclusions
    exclude_intervals = {"chr1": InterLap()}
    exclude_intervals["chr1"].add((200, 300))
    
    # Create sampler with exclusions
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        exclude_intervals=exclude_intervals,
        candidate_ratio=10.0, # Ask for many to force picking if available
        bins=1, # One bin, everything matches
        metrics=["gc"],
        seed=42
    )
    
    # It should only pick the valid one
    picked = list(sampler)
    assert len(picked) > 0 # Should pick something
    
    for interval in picked:
        assert interval.start != 200
        assert interval.start == 100
