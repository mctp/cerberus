import numpy as np
import pytest

from cerberus.interval import Interval
from cerberus.samplers import ComplexityMatchedSampler


class MockSampler:
    def __init__(self, intervals):
        self._intervals = intervals
        self.chrom_sizes = {"chr1": 1000}
        self.folds = []
        self.exclude_intervals = {}

    def __iter__(self):
        return iter(self._intervals)

    def __len__(self):
        return len(self._intervals)

    def __getitem__(self, idx):
        return self._intervals[idx]

    def resample(self, seed=None):
        pass

    def split_folds(self, test_fold=None, val_fold=None):
        return (self, self, self)

    def get_interval_source(self, idx):
        return type(self).__name__


@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"

    # Generate random sequence
    rng = np.random.default_rng(42)
    bases = np.array(["A", "C", "G", "T"])
    seq = "".join(rng.choice(bases, size=1000))

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq + "\n")

    import pyfaidx

    pyfaidx.Faidx(str(fasta_path))
    return fasta_path


def test_complexity_matched_sampler_metrics(mock_fasta):
    target = MockSampler([Interval("chr1", 0, 100, "+")])
    candidate = MockSampler([Interval("chr1", 100, 200, "+")])
    chrom_sizes = {"chr1": 1000}

    # Test default mode -> 3 metrics
    s1 = ComplexityMatchedSampler(target, candidate, mock_fasta, chrom_sizes)
    assert s1.target_metrics.shape[1] == 3
    assert s1.metrics == ["gc", "dust", "cpg"]

    # Test "gc" mode -> 1 metric
    s2 = ComplexityMatchedSampler(
        target, candidate, mock_fasta, chrom_sizes, metrics=["gc"]
    )
    assert s2.target_metrics.shape[1] == 1
    assert s2.metrics == ["gc"]

    # Test list mode ["gc", "dust"] -> 2 metrics
    s3 = ComplexityMatchedSampler(
        target, candidate, mock_fasta, chrom_sizes, metrics=["gc", "dust"]
    )
    assert s3.target_metrics.shape[1] == 2
    assert s3.metrics == ["gc", "dust"]

    # Test invalid metric
    with pytest.raises(ValueError):
        ComplexityMatchedSampler(
            target, candidate, mock_fasta, chrom_sizes, metrics=["invalid"]
        )


def test_complexity_matched_sampler_center_size(mock_fasta):
    """center_size is stored and changes the cache key."""
    target = MockSampler([Interval("chr1", 0, 100, "+")])
    candidate = MockSampler([Interval("chr1", 100, 200, "+")])
    chrom_sizes = {"chr1": 1000}

    s_none = ComplexityMatchedSampler(target, candidate, mock_fasta, chrom_sizes)
    s_center = ComplexityMatchedSampler(
        target, candidate, mock_fasta, chrom_sizes, center_size=50
    )

    assert s_none.center_size is None
    assert s_center.center_size == 50

    # Cache keys should differ for the same interval
    iv = Interval("chr1", 0, 100, "+")
    assert s_none._cache_key(iv) == str(iv)
    assert s_center._cache_key(iv) == f"{iv}:cs=50"
