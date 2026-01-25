
import pytest
from cerberus.samplers import MultiSampler, IntervalSampler, ScaledSampler, create_sampler
from cerberus.genome import create_genome_folds

@pytest.fixture
def peaks_bed(tmp_path):
    p = tmp_path / "peaks.bed"
    p.write_text("chr1\t100\t200\nchr1\t300\t400\nchr1\t500\t600\n") # 3 peaks
    return p

@pytest.fixture
def negatives_bed(tmp_path):
    p = tmp_path / "negatives.bed"
    p.write_text("chr1\t1000\t1100\nchr1\t1200\t1300\nchr1\t1400\t1500\nchr1\t1600\t1700\n") # 4 negatives
    return p

@pytest.fixture
def chrom_sizes():
    return {"chr1": 2000}

def test_multi_sampler_basic(peaks_bed, negatives_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    s2 = IntervalSampler(negatives_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    
    # No scaling factors (takes all)
    ms = MultiSampler([s1, s2], chrom_sizes=chrom_sizes, exclude_intervals={}, seed=42)
    
    assert len(ms) == 3 + 4
    
    intervals = list(ms)
    assert len(intervals) == 7
    # Verify we have both types (by position)
    starts = [i.start for i in intervals]
    assert 100 in starts
    assert 1000 in starts

def test_scaled_sampler(peaks_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    # 3 items. Scale to 1 (subsample).
    ss = ScaledSampler(s1, num_samples=1, seed=42)
    assert len(ss) == 1
    assert len(list(ss)) == 1
    
    # Scale to 5 (oversample).
    ss2 = ScaledSampler(s1, num_samples=5, seed=42)
    assert len(ss2) == 5
    assert len(list(ss2)) == 5
    
def test_create_sampler_scaling(peaks_bed, negatives_bed, chrom_sizes):
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "exclude_intervals": {},
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(peaks_bed)},
                    "scaling": 1.0
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(negatives_bed)},
                    "scaling": 0.5
                }
            ]
        }
    }
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = create_sampler(config, chrom_sizes, exclude_intervals={}, folds=folds)
    assert isinstance(sampler, MultiSampler)
    # s1: 3 * 1.0 = 3
    # s2: 4 * 0.5 = 2
    # Total = 5
    assert len(sampler) == 5
    
    # Check wrappers
    # s1: 1.0 -> IntervalSampler (no wrapper needed logic)
    assert isinstance(sampler.samplers[0], IntervalSampler)
    
    # s2: 0.5 -> ScaledSampler wrapper
    assert isinstance(sampler.samplers[1], ScaledSampler)
    assert sampler.samplers[1].num_samples == 2

def test_scaled_sampler_resample(peaks_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    # 3 items
    
    # Scale to 1 item
    ss = ScaledSampler(s1, num_samples=1, seed=42)
    
    indices1 = ss._indices.copy()
    ss.resample(seed=43)
    indices2 = ss._indices.copy()
    
    assert len(indices1) == 1
    assert len(indices2) == 1
    # Different seeds *should* produce different indices (likely, but 1/3 chance of collision)
    # But checking that indices are valid is enough.
    assert 0 <= indices1[0] < 3

def test_multi_sampler_split_folds(peaks_bed, negatives_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    s2 = IntervalSampler(negatives_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    
    ms = MultiSampler([s1, s2], chrom_sizes=chrom_sizes, exclude_intervals={})
    
    train, val, test = ms.split_folds(test_fold=0, val_fold=1)
    
    assert isinstance(train, MultiSampler)
    assert isinstance(val, MultiSampler)
    assert isinstance(test, MultiSampler)
    
    assert len(train.samplers) == 2
    assert len(val.samplers) == 2
    assert len(test.samplers) == 2
    
    total_len = len(train) + len(val) + len(test)
    original_len = len(ms)
    assert total_len == original_len
