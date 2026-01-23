
import pytest
from pathlib import Path
from cerberus.samplers import create_sampler, MultiSampler

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
    # B scaling: 10/100 = 0.1
    # Expected factors: [1.0, 0.1]
    
    factors = sampler.scaling_factors
    assert factors[0] == 1.0
    assert abs(factors[1] - 0.1) < 1e-6
    
    # Total length: 10 * 1 + 100 * 0.1 = 10 + 10 = 20
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
    # A scaling: 100/10 = 10.0
    # Expected factors: [10.0, 1.0]
    
    factors = sampler.scaling_factors
    assert abs(factors[0] - 10.0) < 1e-6
    assert factors[1] == 1.0
    
    # Total length: 10 * 10 + 100 * 1 = 100 + 100 = 200
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
    
    # A: 10 -> 5. Scaling 0.5
    # B: 100 -> 50. Scaling 0.5
    
    factors = sampler.scaling_factors
    assert abs(factors[0] - 0.5) < 1e-6
    assert abs(factors[1] - 0.5) < 1e-6
    
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
    
    # 0 (A): 1.0 -> 10 samples
    # 1 (B): "min" -> 10 samples. Scaling: 10/100 = 0.1
    # 2 (C): "max" -> 100 samples. Scaling: 100/10 = 10.0
    
    factors = sampler.scaling_factors
    assert factors[0] == 1.0
    assert abs(factors[1] - 0.1) < 1e-6
    assert abs(factors[2] - 10.0) < 1e-6
    
    # Total length: 10 + 10 + 100 = 120
    assert len(sampler) == 120
