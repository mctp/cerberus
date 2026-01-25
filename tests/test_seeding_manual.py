from typing import cast
from cerberus.samplers import create_sampler, MultiSampler

def test_random_sampler_seeding():
    chrom_sizes = {"chr1": 10000}
    config = {
        "sampler_type": "random",
        "padded_size": 100,
        "sampler_args": {
            "num_intervals": 5
        }
    }
    
    # Test deterministic behavior
    s1 = create_sampler(config, chrom_sizes, {}, [], seed=42)
    s2 = create_sampler(config, chrom_sizes, {}, [], seed=42)
    
    # RandomSampler generates intervals in init
    intervals1 = list(s1)
    intervals2 = list(s2)
    
    assert len(intervals1) == 5
    assert len(intervals2) == 5
    
    # Check strict equality of generated intervals
    for i1, i2 in zip(intervals1, intervals2):
        assert i1.chrom == i2.chrom
        assert i1.start == i2.start
        assert i1.end == i2.end
        
    # Test difference with different seed
    s3 = create_sampler(config, chrom_sizes, {}, [], seed=43)
    intervals3 = list(s3)
    assert intervals1 != intervals3

def test_multi_sampler_seeding(tmp_path):
    # Manual instantiation of MultiSampler
    from cerberus.samplers import IntervalSampler, RandomSampler, ScaledSampler
    from interlap import InterLap
    
    chrom_sizes = {"chr1": 10000}
    p = tmp_path / "peaks.bed"
    p.write_text("chr1\t100\t200\nchr1\t300\t400\n")
    
    # Function to create MS with a seed
    def create_ms(seed):
        # We need to replicate the seeding logic that create_sampler used to do for children
        # s1: Interval (no seed needed really, but ScaledSampler wrapper needs it)
        s1 = IntervalSampler(p, chrom_sizes, 100, {}, [])
        # Scaling 0.5 (2 items -> 1 item)
        seed1 = seed + 1 if seed is not None else None
        s1_scaled = ScaledSampler(s1, num_samples=1, seed=seed1)
        
        # s2: Random
        seed2 = seed + 2 if seed is not None else None
        s2 = RandomSampler(chrom_sizes, 100, num_intervals=10, exclude_intervals={}, folds=[], seed=seed2)
        
        return MultiSampler([s1_scaled, s2], chrom_sizes, {}, seed=seed)

    ms1 = create_ms(42)
    ms2 = create_ms(42)
    
    # Check internal RandomSampler determinism
    rs1 = ms1.samplers[1]
    rs2 = ms2.samplers[1]
    assert list(rs1) == list(rs2)
    
    # Check ScaledSampler determinism
    ss1 = ms1.samplers[0]
    ss2 = ms2.samplers[0]
    assert list(ss1) == list(ss2)
    
    # Check MultiSampler mixing
    assert ms1._indices == ms2._indices
    
    # Check difference
    ms3 = create_ms(43)
    assert ms1._indices != ms3._indices
