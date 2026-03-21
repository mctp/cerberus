import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cerberus.samplers import ComplexityMatchedSampler, ListSampler
from cerberus.interval import Interval
from interlap import InterLap

# Mock compute_intervals_complexity
@pytest.fixture
def mock_compute():
    with patch("cerberus.samplers.compute_intervals_complexity") as m:
        def side_effect(intervals, fasta, metrics, center_size=None):
            # Return random metrics
            return np.random.rand(len(intervals), len(metrics))
        m.side_effect = side_effect
        yield m

def test_metrics_caching(mock_compute):
    """
    Test that metrics are computed once and reused via cache.
    """
    # Setup samplers
    intervals = [Interval("chr1", i*100, (i+1)*100) for i in range(100)]
    target_sampler = ListSampler(intervals[:20])
    candidate_sampler = ListSampler(intervals[20:])
    
    chrom_sizes = {"chr1": 10000}
    fasta_path = "mock.fa"
    
    folds = [
        {"chr1": InterLap([(0, 5000)])}, # Fold 0 (Test/Val/Train depending on args)
        {"chr1": InterLap([(5000, 10000)])}, # Fold 1
    ]
    # Re-init samplers with folds
    target_sampler = ListSampler(intervals[:20], chrom_sizes=chrom_sizes, folds=folds)
    candidate_sampler = ListSampler(intervals[20:], chrom_sizes=chrom_sizes, folds=folds)

    # Create sampler (Should trigger compute)
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        folds=folds,
        metrics=["gc"],
        generate_on_init=True
    )
    
    # Assert compute called
    assert mock_compute.call_count > 0
    
    # Check cache population
    # Cache key is str(interval)
    assert len(sampler.metrics_cache) == len(intervals) # 20 targets + 80 candidates = 100
    
    # Reset mock to check split behavior
    mock_compute.reset_mock()
    
    # Split
    splits = sampler.split_folds(test_fold=0, val_fold=1)
    
    # Splits are (Train, Val, Test)
    # Train should have initialized.
    # Val should have initialized.
    # Test should have initialized.
    
    # BUT, because they share cache, mock_compute should NOT be called.
    assert mock_compute.call_count == 0
    
    # Verify splits are valid
    train, val, test = splits
    assert isinstance(train, ComplexityMatchedSampler)
    assert train.metrics_cache is sampler.metrics_cache
    assert len(train.metrics_cache) == 100

def test_metrics_cache_partial_hit(mock_compute):
    """
    Test that new intervals trigger computation while existing ones don't.
    """
    intervals = [Interval("chr1", i*100, (i+1)*100) for i in range(10)]
    target_sampler = ListSampler(intervals[:5])
    candidate_sampler = ListSampler(intervals[5:])
    chrom_sizes = {"chr1": 1000}
    
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path="mock.fa",
        chrom_sizes=chrom_sizes,
        generate_on_init=True
    )
    
    initial_calls = mock_compute.call_count
    mock_compute.reset_mock()
    
    # New interval
    new_intervals = [Interval("chr1", 2000, 2100)] 
    new_target = ListSampler(new_intervals)
    
    # Create new sampler sharing cache
    sampler2 = ComplexityMatchedSampler(
        target_sampler=new_target,
        candidate_sampler=candidate_sampler, # Reused
        fasta_path="mock.fa",
        chrom_sizes=chrom_sizes,
        metrics_cache=sampler.metrics_cache,
        generate_on_init=True
    )
    
    # Should trigger computation ONLY for new_target
    # call_count should be > 0.
    # Note: _initialize calls _get_metrics twice.
    # 1. target (new) -> cache miss -> compute -> call_count++
    # 2. candidate (reused) -> cache hit -> no compute.
    assert mock_compute.call_count > 0
    
    # Verify ONLY new interval was computed
    # We can check arguments of call
    args, _ = mock_compute.call_args
    computed_intervals = args[0]
    assert len(computed_intervals) == 1
    assert computed_intervals[0] == new_intervals[0]
    
    # Check that cache updated
    assert str(new_intervals[0]) in sampler.metrics_cache
