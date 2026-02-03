import pytest
import gzip
from pathlib import Path
from interlap import InterLap
from cerberus.samplers import (
    IntervalSampler,
    RandomSampler,
    ComplexityMatchedSampler,
    PeakSampler,
    MultiSampler,
    Sampler
)
from cerberus.interval import Interval
from cerberus.genome import create_genome_folds

# --- IntervalSampler Tests ---

def test_interval_sampler_gz(tmp_path):
    f = tmp_path / "test.bed.gz"
    content = "chr1\t100\t200\nchr1\t300\t400\n"
    with gzip.open(f, "wt") as gz:
        gz.write(content)
    
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    
    intervals = list(sampler)
    assert len(intervals) == 2
    assert intervals[0].start == 100

def test_interval_sampler_narrowpeak_gz(tmp_path):
    f = tmp_path / "test.narrowPeak.gz"
    # chr start end name score strand signal pval qval peak
    # Peak at 150 (offset 50)
    line = "chr1\t100\t200\t.\t0\t+\t0\t0\t0\t50\n"
    with gzip.open(f, "wt") as gz:
        gz.write(line)
        
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    
    # Padded size 20. Center 100+50=150. Start 150-10=140. End 160.
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=20, exclude_intervals={}, folds=folds)
    
    intervals = list(sampler)
    assert len(intervals) == 1
    assert intervals[0].start == 140
    assert intervals[0].end == 160

# --- RandomSampler Tests ---

def test_random_sampler_high_rejection():
    # Exclude 0-900. Size 1000. Padded 100.
    # Valid starts: 900 is the last valid point? 
    # If size 1000, indices 0..999.
    # Padded 100. Max start = 1000-100 = 900.
    # 0-900 excluded.
    # Only valid start is 900? range 900-1000.
    # Exclude is [0, 900). 
    
    chrom_sizes = {"chr1": 1000}
    exclude_intervals = {"chr1": InterLap()}
    exclude_intervals["chr1"].add((0, 899)) # Exclude 0-899. 
    
    # Valid starts: 900. (900-1000 does not overlap 0-899)
    # Actually RandomSampler checks: is_excluded(chrom, start, end).
    # Interval(900, 1000). overlap with (0, 899)? No.
    # So 900 is valid.
    # Any start < 900 will overlap.
    
    # We want to test that it eventually finds the spot or gives up.
    
    sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=100,
        num_intervals=5,
        exclude_intervals=exclude_intervals,
        folds=[],
        seed=42
    )
    
    assert len(sampler) <= 5
    for interval in sampler:
        assert interval.start >= 900

def test_random_sampler_impossible():
    # Exclude everything
    chrom_sizes = {"chr1": 100}
    exclude_intervals = {"chr1": InterLap()}
    exclude_intervals["chr1"].add((0, 100))
    
    sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=10,
        num_intervals=5,
        exclude_intervals=exclude_intervals,
        folds=[],
        seed=42
    )
    
    # Should result in 0 intervals and print warning (but not crash)
    assert len(sampler) == 0

# --- ComplexityMatchedSampler Tests ---

@pytest.fixture
def mock_fasta_simple(tmp_path):
    p = tmp_path / "genome.fa"
    with open(p, "w") as f:
        f.write(">chr1\n")
        f.write("G" * 1000 + "\n") # 100% GC
    import pyfaidx
    pyfaidx.Faidx(str(p))
    return p

def test_complexity_matched_sampler_gc_no_matches(mock_fasta_simple):
    # Target: chr1 (100% GC)
    # Candidate: chr1 (100% GC) - wait, this will match.
    
    # We need a candidate that has DIFFERENT GC.
    # But mock_fasta only has chr1.
    # Let's make candidate look at a region that we spoof? 
    # Or just use same fasta but different coords?
    # Actually if whole chr1 is G, then any interval is 100% GC.
    
    # Let's create a fasta with chr1 (100% GC) and chr2 (0% GC).
    # Target from chr1. Candidate from chr2.
    # They will never match.
    
    p = mock_fasta_simple.parent / "genome_diff.fa"
    with open(p, "w") as f:
        f.write(">chr1\n")
        f.write("G" * 100 + "\n")
        f.write(">chr2\n")
        f.write("A" * 100 + "\n")
    import pyfaidx
    pyfaidx.Faidx(str(p))
    
    chrom_sizes = {"chr1": 100, "chr2": 100}
    
    # Target: 1 interval on chr1
    target_sampler = MockSampler([Interval("chr1", 0, 10, "+")])
    
    # Candidate: 10 intervals on chr2
    candidate_sampler = MockSampler([Interval("chr2", 0, 10, "+")] * 10)
    
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=p,
        chrom_sizes=chrom_sizes,
        exclude_intervals={},
        folds=[],
        bins=10,
        match_ratio=1.0,
        seed=42,
        metrics=["gc"]
    )
    
    # Should find NO matches
    assert len(sampler) == 0

class MockSampler(Sampler):
    def __init__(self, intervals):
        self.intervals = intervals
        self.chrom_sizes = {}
        self.folds = []
        self.exclude_intervals = {}
    def __iter__(self): return iter(self.intervals)
    def __len__(self): return len(self.intervals)
    def __getitem__(self, idx): return self.intervals[idx]
    def resample(self, seed=None): pass
    def split_folds(self, test_fold=None, val_fold=None):
        return self, self, self

# --- PeakSampler Tests ---

def test_peak_sampler_folds(tmp_path, mock_fasta_simple):
    # Create dummy files
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t10\t20\nchr1\t30\t40\n")
    
    chrom_sizes = {"chr1": 100}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    
    sampler = PeakSampler(
        intervals_path=peaks,
        fasta_path=mock_fasta_simple,
        chrom_sizes=chrom_sizes,
        padded_size=10,
        exclude_intervals={},
        folds=folds,
        background_ratio=1.0,
        seed=42
    )
    
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)
    
    assert isinstance(train, MultiSampler)
    assert isinstance(val, MultiSampler)
    assert isinstance(test, MultiSampler)
    
    # Check that they are not empty (assuming distribution allows)
    # With 2 intervals and 2 folds:
    # Fold 0: 0-50
    # Fold 1: 50-100
    # Interval 10-20 -> Fold 0 (Test)
    # Interval 30-40 -> Fold 0 (Test)
    
    # So Test should have 2 peaks + matching backgrounds
    # Val should have 0
    # Train should have 0
    
    assert len(test.samplers[0]) == 2 # 2 peaks
    assert len(test.samplers[1]) >= 0 # some backgrounds
    
    assert len(val.samplers[0]) == 0
    assert len(train.samplers[0]) == 0

# --- MultiSampler Tests ---

def test_multi_sampler_empty():
    chrom_sizes = {"chr1": 100}
    ms = MultiSampler([], chrom_sizes=chrom_sizes, exclude_intervals={}, seed=42)
    assert len(ms) == 0
    assert list(ms) == []
    
    train, val, test = ms.split_folds()
    assert len(train) == 0

def test_random_sampler_split_dynamic_behavior(tmp_path):
    # Regression test: RandomSampler splits should remain dynamic RandomSamplers,
    # not static ListSamplers.
    
    chrom_sizes = {"chr1": 1000}
    # Folds: 0-500, 500-1000
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    
    sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=10,
        num_intervals=10,
        exclude_intervals={},
        folds=folds,
        seed=42
    )
    
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)
    
    # Train should be empty (since K=2, Fold 0 is Test, Fold 1 is Val)
    # Let's use K=5 so we have a train set.
    # Note: 'chrom_partition' puts entire chromosomes into folds. 
    # With 1 chrom, only 1 fold is populated.
    # We need multiple chroms to populate multiple folds.
    chrom_sizes_multi = {f"chr{i}": 1000 for i in range(1, 6)}
    folds = create_genome_folds(chrom_sizes_multi, fold_type="chrom_partition", fold_args={"k": 5})
    
    sampler = RandomSampler(
        chrom_sizes=chrom_sizes_multi,
        padded_size=10,
        num_intervals=100,
        exclude_intervals={},
        folds=folds,
        seed=42
    )
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)

    intervals_v1 = list(train)
    
    # Resample
    train.resample(seed=123)
    intervals_v2 = list(train)
    
    # Verify content changed
    # With static ListSampler, this will fail as intervals_v1 == intervals_v2
    # With dynamic RandomSampler, this should pass.
    
    # We check if at least one interval is different.
    # Note: highly unlikely to be identical by chance with 100 intervals.
    assert intervals_v1 != intervals_v2, "RandomSampler split should allow resampling, but intervals remained static."

