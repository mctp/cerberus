import pytest
import numpy as np
import pyBigWig
from pathlib import Path
from interlap import InterLap
from cerberus.samplers import IntervalSampler
from cerberus.signal import SignalExtractor
from cerberus.core import Interval

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def chrom_sizes():
    return {"chr1": 1000}

@pytest.fixture
def bed_file(temp_dir):
    path = temp_dir / "test.bed"
    with open(path, "w") as f:
        # chr1, start=100, end=200
        f.write("chr1\t100\t200\tname\t0\t+\n")
    return path

@pytest.fixture
def narrowpeak_file(temp_dir):
    path = temp_dir / "test.narrowPeak"
    with open(path, "w") as f:
        # chr1, start=100, end=200, ..., summit=50
        # summit is relative to start, so absolute summit is 150.
        # 10 columns required
        f.write("chr1\t100\t200\tname\t0\t+\t0.0\t0.0\t0.0\t50\n")
    return path

@pytest.fixture
def bigwig_file(temp_dir, chrom_sizes):
    path = temp_dir / "test.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()])
    
    # Add signal 1.0 at 100-200
    # pyBigWig addEntries: starts are 0-based.
    bw.addEntries("chr1", [100], values=[1.0], span=100)
    bw.close()
    return path

def test_bed_coordinates(bed_file, chrom_sizes):
    """
    Test that BED files are loaded with correct 0-based [start, end) coordinates.
    BED: chr1 100 200
    Expected: Interval(chr1, 100, 200)
    """
    sampler = IntervalSampler(
        file_path=bed_file,
        chrom_sizes=chrom_sizes,
        padded_size=None,
        exclude_intervals={},
        folds=[{}, {}, {}] # Dummy folds
    )
    
    intervals = list(sampler)
    assert len(intervals) == 1
    iv = intervals[0]
    
    assert iv.chrom == "chr1"
    assert iv.start == 100
    assert iv.end == 200
    assert iv.strand == "+"

def test_narrowpeak_coordinates(narrowpeak_file, chrom_sizes):
    """
    Test that narrowPeak files are loaded with correct 0-based [start, end) coordinates.
    narrowPeak: chr1 100 200
    Expected: Interval(chr1, 100, 200)
    """
    sampler = IntervalSampler(
        file_path=narrowpeak_file,
        chrom_sizes=chrom_sizes,
        padded_size=None,
        exclude_intervals={},
        folds=[{}, {}, {}]
    )
    
    intervals = list(sampler)
    assert len(intervals) == 1
    iv = intervals[0]
    
    assert iv.chrom == "chr1"
    assert iv.start == 100
    assert iv.end == 200
    assert iv.strand == "+"

def test_narrowpeak_centering(narrowpeak_file, chrom_sizes):
    """
    Test that narrowPeak centering works correctly.
    narrowPeak: chr1 100 200, summit=50 -> absolute summit 150
    If padded_size=20, expected interval: [150-10, 150+10) = [140, 160)
    """
    padded_size = 20
    sampler = IntervalSampler(
        file_path=narrowpeak_file,
        chrom_sizes=chrom_sizes,
        padded_size=padded_size,
        exclude_intervals={},
        folds=[{}, {}, {}]
    )
    
    intervals = list(sampler)
    assert len(intervals) == 1
    iv = intervals[0]
    
    assert iv.chrom == "chr1"
    # Summit is at 100 + 50 = 150
    # Start = 150 - 10 = 140
    # End = 140 + 20 = 160
    assert iv.start == 140
    assert iv.end == 160

def test_bigwig_coordinates(bigwig_file):
    """
    Test that BigWig signal is extracted correctly for given intervals.
    BigWig: 1.0 at 100-200.
    """
    extractor = SignalExtractor(bigwig_paths={"signal": bigwig_file})
    
    # 1. Exact match
    iv = Interval("chr1", 100, 200)
    signal = extractor.extract(iv)
    # shape: (Channels, Length) -> (1, 100)
    assert signal.shape == (1, 100)
    assert np.allclose(signal[0].numpy(), 1.0)
    
    # 2. Subset
    iv = Interval("chr1", 100, 101)
    signal = extractor.extract(iv)
    assert signal.shape == (1, 1)
    assert np.allclose(signal[0].numpy(), 1.0)
    
    # 3. Crossing boundary (left)
    iv = Interval("chr1", 90, 110)
    signal = extractor.extract(iv)
    assert signal.shape == (1, 20)
    # 90-100 should be 0 (default), 100-110 should be 1.0
    assert np.allclose(signal[0, :10].numpy(), 0.0)
    assert np.allclose(signal[0, 10:].numpy(), 1.0)
    
    # 4. Crossing boundary (right)
    iv = Interval("chr1", 190, 210)
    signal = extractor.extract(iv)
    assert signal.shape == (1, 20)
    # 190-200 should be 1.0, 200-210 should be 0
    assert np.allclose(signal[0, :10].numpy(), 1.0)
    assert np.allclose(signal[0, 10:].numpy(), 0.0)

def test_bigwig_one_base(bigwig_file):
    """
    Test single base extraction.
    """
    extractor = SignalExtractor(bigwig_paths={"signal": bigwig_file})
    
    # 100 corresponds to the first base of the signal region
    iv = Interval("chr1", 100, 101) 
    signal = extractor.extract(iv)
    assert signal[0, 0] == 1.0
    
    # 99 should be 0
    iv = Interval("chr1", 99, 100)
    signal = extractor.extract(iv)
    assert signal[0, 0] == 0.0
    
    # 199 corresponds to the last base of the signal region
    iv = Interval("chr1", 199, 200)
    signal = extractor.extract(iv)
    assert signal[0, 0] == 1.0
    
    # 200 should be 0
    iv = Interval("chr1", 200, 201)
    signal = extractor.extract(iv)
    assert signal[0, 0] == 0.0
