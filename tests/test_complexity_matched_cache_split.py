
import pytest
from pathlib import Path
import random
import numpy as np
from cerberus.samplers import create_sampler, ComplexityMatchedSampler, ListSampler, Interval, RandomSampler
from interlap import InterLap

def test_cache_reuse_across_splits(tmp_path):
    # Mock FASTA
    fasta_path = tmp_path / "genome.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("A" * 10000 + "\n")
    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))

    chrom_sizes = {"chr1": 10000}
    
    # Target: 10 intervals
    target_intervals = [Interval("chr1", i*100, i*100+50, "+") for i in range(10)]
    target_sampler = ListSampler(intervals=target_intervals, chrom_sizes=chrom_sizes)
    
    # Candidate: RandomSampler
    # Use explicit seeds to track behavior
    candidate_sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=50,
        num_intervals=100,
        seed=42
    )
    
    # Shared cache
    metrics_cache = {}
    
    # Parent Sampler
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        bins=1,
        metrics=["gc"],
        seed=123,
        metrics_cache=metrics_cache
    )
    
    # Initialize parent -> populates cache with 100 random intervals
    sampler.resample()
    
    print(f"Cache size after parent init: {len(metrics_cache)}")
    # Should be targets (10) + candidates (100) = 110 (approx, allowing for duplicates)
    assert len(metrics_cache) >= 10
    parent_cache_size = len(metrics_cache)
    
    # Collect parent candidate intervals
    parent_candidates = set(str(iv) for iv in sampler.candidate_sampler)
    # assert len(parent_candidates) == 100 # Might have duplicates
    
    # Split folds
    # Define folds: chr1: 0-5000 (train), 5000-10000 (val)
    # But for simplicity, let's just use default folds (none provided) -> split logic?
    # RandomSampler.split_folds requires folds to be provided or it fails?
    # Wait, RandomSampler.split_folds raises ValueError if not self.folds
    
    folds = [
        {"chr1": InterLap([(0, 4999)])}, # Fold 0
        {"chr1": InterLap([(5000, 10000)])} # Fold 1
    ]
    
    # Re-create samplers with folds
    target_sampler = ListSampler(intervals=target_intervals, chrom_sizes=chrom_sizes, folds=folds)

    candidate_sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=50,
        num_intervals=100,
        folds=folds,
        seed=42
    )
    
    metrics_cache = {}
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        folds=folds,
        bins=1,
        metrics=["gc"],
        seed=123,
        metrics_cache=metrics_cache
    )
    sampler.resample()
    parent_cache_size = len(metrics_cache)
    print(f"Parent cache size: {parent_cache_size}")
    
    # Split: Test=0, Val=1
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)
    
    # Train should be empty (since we only have 2 folds and used 0 and 1 for test/val)
    # Wait, 3 splits: train, val, test.
    # If test_fold=0, val_fold=1.
    # Folds indices: 0, 1.
    # Train is whatever is NOT 0 or 1.
    # Since only folds 0 and 1 exist, Train should be empty or handle gracefully.
    
    # Let's verify test split (Fold 0)
    print("Initializing Test Split...")
    test.resample()
    
    # Test split candidate sampler should ideally reuse intervals from parent
    # If it generates NEW intervals, cache size will increase
    
    new_cache_size = len(metrics_cache)
    print(f"Cache size after split init: {new_cache_size}")
    
    # If cache size increased significantly, it means we generated new intervals
    # Parent had 100 candidates.
    # Test split will ask for candidates in Fold 0 (approx 50).
    # If it generates 50 NEW candidates, cache grows by 50.
    # If it reuses parent candidates, cache grows by 0.
    
    diff = new_cache_size - parent_cache_size
    print(f"Cache growth: {diff}")
    
    # Assert cache did NOT grow (allow small margin for weirdness)
    assert diff == 0, f"Cache grew by {diff} entries, implying regeneration of candidates!"

if __name__ == "__main__":
    import sys
    from pytest import ExitCode
    sys.exit(pytest.main(["-v", __file__]))
