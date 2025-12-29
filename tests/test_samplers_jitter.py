import pytest
from pathlib import Path
from cerberus.interval import Interval
from cerberus.samplers import IntervalSampler
from cerberus.genome import create_genome_folds

@pytest.fixture
def mock_bed_file(tmp_path):
    p = tmp_path / "test.bed"
    p.write_text("chr1\t100\t200\nchr1\t1000\t1100\n")
    return p

def test_sampler_with_padding(mock_bed_file):
    # Original window 50 + 2*10 jitter = 70 padded_size
    # Original interval 1: [100, 200]. Center 150.
    # Target window: [125, 175] (length 50).
    # With padding (size 70): [115, 185] (length 70).
    
    chrom_sizes = {"chr1": 2000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(
        file_path=mock_bed_file,
        padded_size=70,
        chrom_sizes=chrom_sizes,
        exclude_intervals={},
        folds=folds
    )
    
    intervals = list(sampler)
    assert len(intervals) == 2
    
    # Check first interval
    i1 = intervals[0]
    # Center = 150
    # Start = 150 - 70//2 = 115
    # End = 115 + 70 = 185
    assert i1.start == 115
    assert i1.end == 185
    assert len(i1) == 70

def test_sampler_padding_validity_check(mock_bed_file):
    # Interval close to start: 100-200. Center 150.
    # padded_size 320. half 160.
    # Start = 150 - 160 = -10 (Invalid!)
    
    chrom_sizes = {"chr1": 2000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(
        file_path=mock_bed_file,
        padded_size=320,
        chrom_sizes=chrom_sizes,
        exclude_intervals={},
        folds=folds
    )
    
    intervals = list(sampler)
    # First interval should be filtered out
    assert len(intervals) == 1
    assert intervals[0].start > 0

@pytest.mark.parametrize("padded_size", [
    (100),
    (120),
    (250)
])
def test_sampler_output_length(mock_bed_file, padded_size):
    # Ensure chromosomes are large enough for these tests
    chrom_sizes = {"chr1": 10000}
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(
        file_path=mock_bed_file,
        padded_size=padded_size,
        chrom_sizes=chrom_sizes,
        exclude_intervals={},
        folds=folds
    )
    
    for interval in sampler:
        assert len(interval) == padded_size, \
            f"Expected length {padded_size}, got {len(interval)}"
