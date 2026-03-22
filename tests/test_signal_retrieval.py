import os

import numpy as np
import pyBigWig as pybigwig
import pytest

from cerberus.interval import Interval
from cerberus.signal import SignalExtractor


@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_compare_signal_extraction_with_pybigwig(mdapca2b_ar_dataset):
    """
    Compare SignalExtractor results with direct pybigwig usage
    on a set of random intervals.
    """
    BW_PATH = mdapca2b_ar_dataset["bigwig"]

    # 1. Setup
    bw_path = str(BW_PATH)
    
    # Initialize SignalExtractor
    extractor = SignalExtractor(bigwig_paths={"signal": BW_PATH})
    
    # Open with pybigwig to get chrom sizes for random sampling
    bw = pybigwig.open(bw_path)
    chroms = bw.chroms()
    
    # 2. Generate Random Intervals
    # We'll pick a few random chromosomes and intervals
    num_intervals = 20
    interval_len = 1000
    
    intervals = []
    
    # Sort chroms to be deterministic if possible, or just list keys
    chrom_names = list(chroms.keys())
    
    np.random.seed(42) # Ensure reproducibility
    
    for _ in range(num_intervals):
        chrom = np.random.choice(chrom_names)
        size = chroms[chrom]
        
        # Ensure we can pick a valid interval
        if size < interval_len:
            continue
            
        start = np.random.randint(0, size - interval_len)
        end = start + interval_len
        
        intervals.append(Interval(chrom, start, end, "+"))
        
    assert len(intervals) > 0, "No valid intervals generated"
    
    # 3. Compare Extractions
    for interval in intervals:
        # A. Cerberus SignalExtractor
        # Returns shape (Channels, Length) -> (1, Length) here
        cerberus_tensor = extractor.extract(interval)
        cerberus_vals = cerberus_tensor.numpy()[0] # Flatten channel dim
        
        # B. Direct pybigwig
        # pybigwig.values(chrom, start, end) returns list of floats (or nans)
        # We need to handle nan_to_num as SignalExtractor does
        pybw_vals = bw.values(interval.chrom, interval.start, interval.end)
        pybw_arr = np.array(pybw_vals, dtype=np.float32)
        pybw_arr = np.nan_to_num(pybw_arr)
        
        # C. Compare
        # Allow small floating point differences if any, but they should be identical
        # as we are reading same file and casting to float32.
        assert np.allclose(cerberus_vals, pybw_arr, equal_nan=True), \
            f"Mismatch at {interval}: Cerberus vs PyBigWig"

    bw.close()
