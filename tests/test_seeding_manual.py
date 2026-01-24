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
    chrom_sizes = {"chr1": 10000}
    # Mock intervals file
    p = tmp_path / "peaks.bed"
    p.write_text("chr1\t100\t200\nchr1\t300\t400\n")
    
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(p)},
                    "scaling": 0.5 # Subsample
                },
                {
                    "type": "random",
                    "args": {"num_intervals": 10},
                    "scaling": 1.0
                }
            ]
        }
    }
    
    # With seed 42
    ms1 = create_sampler(config, chrom_sizes, {}, [], seed=42)
    ms1 = cast(MultiSampler, ms1)

    # With seed 42 again
    ms2 = create_sampler(config, chrom_sizes, {}, [], seed=42)
    ms2 = cast(MultiSampler, ms2)

    # Check internal RandomSampler determinism
    # ms1.samplers[1] is RandomSampler
    rs1 = ms1.samplers[1]
    rs2 = ms2.samplers[1]
    
    # Check generated intervals
    assert list(rs1) == list(rs2)
    
    # Check MultiSampler mixing (resample calls rng)
    # MultiSampler calls resample(seed) in init. 
    # With seed passed to create_sampler, it is passed to init.
    
    indices1 = ms1._indices
    indices2 = ms2._indices
    
    assert indices1 == indices2
