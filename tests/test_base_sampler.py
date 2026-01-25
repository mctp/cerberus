import pytest
from cerberus.samplers import BaseSampler, ListSampler
from cerberus.interval import Interval
from interlap import InterLap

def test_list_sampler_init_iteration():
    intervals = [
        Interval("chr1", 100, 200, "+"),
        Interval("chr2", 300, 400, "-")
    ]
    sampler = ListSampler(intervals=intervals)
    
    assert len(sampler) == 2
    assert sampler[0] == intervals[0]
    assert list(sampler) == intervals

def test_list_sampler_subset():
    intervals = [
        Interval("chr1", 100, 200, "+"),
        Interval("chr2", 300, 400, "-"),
        Interval("chr3", 500, 600, "+")
    ]
    sampler = ListSampler(intervals=intervals)
    
    subset = sampler._subset([0, 2])
    assert isinstance(subset, ListSampler)
    assert len(subset) == 2
    assert subset[0] == intervals[0]
    assert subset[1] == intervals[2]

def test_list_sampler_split_folds():
    intervals = [
        Interval("chr1", 100, 200),
        Interval("chr2", 100, 200),
        Interval("chr3", 100, 200)
    ]
    
    # Define folds:
    # Fold 0: chr1
    # Fold 1: chr2
    # Fold 2: chr3 (implicit if not in others)
    
    folds = [
        {"chr1": InterLap([(0, 1000)])}, # Fold 0
        {"chr2": InterLap([(0, 1000)])}, # Fold 1
        {"chr3": InterLap([(0, 1000)])}  # Fold 2
    ]
    
    sampler = ListSampler(intervals=intervals, folds=folds)
    
    # Test fold 0, Val fold 1
    # Train: chr3 (not in 0 or 1)
    
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)
    
    assert len(test) == 1
    assert test[0].chrom == "chr1"
    
    assert len(val) == 1
    assert val[0].chrom == "chr2"
    
    assert len(train) == 1
    assert train[0].chrom == "chr3"

def test_base_sampler_exclude():
    # BaseSampler logic doesn't depend on intervals list, only on exclude_intervals dict
    exclude_intervals = {"chr1": InterLap([(150, 250)])}
    
    sampler = BaseSampler(exclude_intervals=exclude_intervals)
    
    # BaseSampler provides is_excluded
    assert sampler.is_excluded("chr1", 100, 200) # Overlaps 150
    assert not sampler.is_excluded("chr1", 300, 400)
    
    # If we wanted to filter, we'd do it manually or assume the user did it before passing intervals.
    # BaseSampler just holds what it's given.
