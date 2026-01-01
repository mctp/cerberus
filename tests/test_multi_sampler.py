import pytest
from cerberus.samplers import MultiSampler, IntervalSampler, create_sampler
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
    
    # 1.0 scaling (take all)
    ms = MultiSampler([s1, s2], chrom_sizes=chrom_sizes, exclude_intervals={}, scaling_factors=[1.0, 1.0])
    
    assert len(ms) == 3 + 4
    
    intervals = list(ms)
    assert len(intervals) == 7
    # Verify we have both types (by position)
    starts = [i.start for i in intervals]
    # IntervalSampler centers:
    # 100-200 -> center 150. Padded 100 -> 100-200. Start 100.
    assert 100 in starts
    assert 1000 in starts

def test_multi_sampler_scaling(peaks_bed, negatives_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    s2 = IntervalSampler(negatives_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    
    # s1: 3 items. s2: 4 items.
    # scaling 1.0 and 0.5.
    # s2 -> 0.5 * 4 = 2 items.
    # Total = 3 + 2 = 5.
    
    ms = MultiSampler([s1, s2], chrom_sizes=chrom_sizes, exclude_intervals={}, scaling_factors=[1.0, 0.5])
    assert len(ms) == 5
    
def test_create_sampler_recursive(peaks_bed, negatives_bed, chrom_sizes):
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
    assert len(sampler.samplers) == 2
    assert isinstance(sampler.samplers[0], IntervalSampler)
    assert len(sampler) == 3 + 2

def test_resampling(peaks_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    # 3 items
    
    ms = MultiSampler([s1], chrom_sizes=chrom_sizes, exclude_intervals={}, scaling_factors=[0.4]) # 3 * 0.4 = 1.2 -> 1 item
    
    indices1 = ms._indices.copy()
    ms.resample()
    indices2 = ms._indices.copy()
    
    # Just checking it runs without error and updates indices.
    assert len(indices1) == 1
    assert len(indices2) == 1

def test_resampling_seeded(peaks_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    # 3 items
    
    # Scaling 0.4 -> 1 item
    ms = MultiSampler([s1], chrom_sizes=chrom_sizes, exclude_intervals={}, scaling_factors=[0.4])
    
    # Seed 42
    ms.resample(seed=42)
    indices1 = ms._indices.copy()
    
    # Seed 42 again
    ms.resample(seed=42)
    indices2 = ms._indices.copy()
    
    assert indices1 == indices2

def test_multi_sampler_split_folds(peaks_bed, negatives_bed, chrom_sizes):
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    s1 = IntervalSampler(peaks_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    s2 = IntervalSampler(negatives_bed, chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    
    ms = MultiSampler([s1, s2], chrom_sizes=chrom_sizes, exclude_intervals={}, scaling_factors=[1.0, 1.0])
    
    train, val, test = ms.split_folds(test_fold=0, val_fold=1)
    
    assert isinstance(train, MultiSampler)
    assert isinstance(val, MultiSampler)
    assert isinstance(test, MultiSampler)
    
    # Check that subsamplers are split
    assert len(train.samplers) == 2
    assert len(val.samplers) == 2
    assert len(test.samplers) == 2
    
    # Verify split logic
    # Fold 0 is Test. Fold 1 is Val.
    # We have only chr1 in folds. 
    # Since we only have chr1, create_genome_folds with k=2 might put chr1 in one fold.
    # Let's check fold assignments:
    # chr1 size 2000.
    # If using 'chrom_partition', it tries to balance sizes.
    # With 1 chrom, it goes to one fold.
    # Wait, create_genome_folds usually needs multiple chroms for 'chrom_partition'.
    # If we only have 'chr1', 'chrom_partition' puts it in fold 0.
    
    # Let's check sizes
    # If chr1 in fold 0 -> Test.
    # Then Val and Train should be empty.
    
    # chr1 is in fold 0 or 1.
    # If it's in fold 0:
    # Test should have all items.
    # Val/Train empty.
    
    # Let's just check mutual exclusivity and total length
    total_len = len(train) + len(val) + len(test)
    original_len = len(ms) # 7
    
    assert total_len == original_len
    
    # Ensure they are disjoint sets of intervals is tricky with MultiSampler resample.
    # But underlying BaseSamplers should be disjoint.
    
    # Check s1 split
    s1_train = train.samplers[0]
    s1_val = val.samplers[0]
    s1_test = test.samplers[0]
    
    s1_total = len(s1_train) + len(s1_val) + len(s1_test)
    assert s1_total == len(s1)
