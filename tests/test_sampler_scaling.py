
import pytest
from pathlib import Path
from cerberus.samplers import create_sampler, MultiSampler, IntervalSampler, ScaledSampler

@pytest.fixture
def mock_intervals(tmp_path):
    # Create two bed files
    file_a = tmp_path / "a.bed"
    file_b = tmp_path / "b.bed"
    
    # File A has 10 lines
    with open(file_a, "w") as f:
        for i in range(10):
            # Shift start to avoid negative start after padding (padded_size=100 -> padding 50)
            start = i * 100 + 1000
            f.write(f"chr1\t{start}\t{start+50}\n")
            
    # File B has 100 lines
    with open(file_b, "w") as f:
        for i in range(100):
            start = i * 100 + 1000
            f.write(f"chr1\t{start}\t{start+50}\n")
            
    return file_a, file_b

def test_scaling_min(mock_intervals):
    file_a, file_b = mock_intervals
    chrom_sizes = {"chr1": 100000}
    exclude_intervals = {}
    folds = []
    
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_a)},
                    "scaling": 1.0
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_b)},
                    "scaling": "min"
                }
            ]
        }
    }
    
    sampler = create_sampler(config, chrom_sizes, exclude_intervals, folds)
    
    assert isinstance(sampler, MultiSampler)
    # A: 10, B: 100. Min: 10.
    
    # 0 (A): 1.0 -> 10 samples (IntervalSampler)
    assert isinstance(sampler.samplers[0], IntervalSampler)
    assert len(sampler.samplers[0]) == 10

    # 1 (B): "min" -> 10 samples (ScaledSampler)
    assert isinstance(sampler.samplers[1], ScaledSampler)
    assert len(sampler.samplers[1]) == 10
    
    # Total length: 10 + 10 = 20
    assert len(sampler) == 20

def test_scaling_max(mock_intervals):
    file_a, file_b = mock_intervals
    chrom_sizes = {"chr1": 100000}
    exclude_intervals = {}
    folds = []
    
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_a)},
                    "scaling": "max"
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_b)},
                    "scaling": 1.0
                }
            ]
        }
    }
    
    sampler = create_sampler(config, chrom_sizes, exclude_intervals, folds)
    assert isinstance(sampler, MultiSampler)
    
    # A: 10, B: 100. Max: 100.
    
    # 0 (A): "max" -> 100 samples (ScaledSampler)
    assert isinstance(sampler.samplers[0], ScaledSampler)
    assert len(sampler.samplers[0]) == 100
    
    # 1 (B): 1.0 -> 100 samples (IntervalSampler)
    assert isinstance(sampler.samplers[1], IntervalSampler)
    assert len(sampler.samplers[1]) == 100
    
    # Total length: 100 + 100 = 200
    assert len(sampler) == 200

def test_scaling_count(mock_intervals):
    file_a, file_b = mock_intervals
    chrom_sizes = {"chr1": 100000}
    exclude_intervals = {}
    folds = []
    
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_a)},
                    "scaling": "count:5"
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_b)},
                    "scaling": "count:50"
                }
            ]
        }
    }
    
    sampler = create_sampler(config, chrom_sizes, exclude_intervals, folds)
    assert isinstance(sampler, MultiSampler)
    
    # A: 10 -> 5 (ScaledSampler)
    assert isinstance(sampler.samplers[0], ScaledSampler)
    assert len(sampler.samplers[0]) == 5
    
    # B: 100 -> 50 (ScaledSampler)
    assert isinstance(sampler.samplers[1], ScaledSampler)
    assert len(sampler.samplers[1]) == 50
    
    # Total length: 5 + 50 = 55
    assert len(sampler) == 55

def test_scaling_mixed(mock_intervals):
    file_a, file_b = mock_intervals
    chrom_sizes = {"chr1": 100000}
    exclude_intervals = {}
    folds = []
    
    # A: 10
    # B: 100
    # C (A again): 10
    
    config = {
        "sampler_type": "multi",
        "padded_size": 100,
        "sampler_args": {
            "samplers": [
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_a)},
                    "scaling": 1.0
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_b)},
                    "scaling": "min"
                },
                {
                    "type": "interval",
                    "args": {"intervals_path": str(file_a)},
                    "scaling": "max"
                }
            ]
        }
    }
    
    sampler = create_sampler(config, chrom_sizes, exclude_intervals, folds)
    assert isinstance(sampler, MultiSampler)
    
    # Lengths: [10, 100, 10]
    # Min: 10. Max: 100.
    
    # 0 (A): 1.0 -> 10 samples (IntervalSampler)
    assert isinstance(sampler.samplers[0], IntervalSampler)
    assert len(sampler.samplers[0]) == 10
    
    # 1 (B): "min" -> 10 samples (ScaledSampler)
    assert isinstance(sampler.samplers[1], ScaledSampler)
    assert len(sampler.samplers[1]) == 10
    
    # 2 (C): "max" -> 100 samples (ScaledSampler)
    assert isinstance(sampler.samplers[2], ScaledSampler)
    assert len(sampler.samplers[2]) == 100
    
    # Total length: 10 + 10 + 100 = 120
    assert len(sampler) == 120
