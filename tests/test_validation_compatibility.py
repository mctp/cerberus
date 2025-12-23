import pytest
from cerberus.config import validate_data_and_sampler_compatibility

def test_validate_data_and_sampler_compatibility_valid():
    data_config = {
        "input_len": 100,
        "max_jitter": 50,
        "output_len": 50,
        "bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False,
        "inputs": {},
        "targets": {}
    }
    sampler_config = {
        "sampler_type": "interval",
        "padded_size": 200,
        "sampler_args": {}
    }
    
    # Should not raise
    validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_exact_boundary():
    # padded_size == input_len + 2 * max_jitter
    data_config = {
        "input_len": 100,
        "max_jitter": 50,
        # Other fields ignored by this validation function but required for typing if strictly checked (here we pass dict)
    }
    sampler_config = {
        "padded_size": 200
    }
    
    validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_invalid():
    # padded_size < input_len + 2 * max_jitter
    data_config = {
        "input_len": 100,
        "max_jitter": 50
    }
    sampler_config = {
        "padded_size": 199
    }
    
    with pytest.raises(ValueError, match="Sampler padded_size \(199\) is smaller than required size"):
        validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_large_jitter():
    data_config = {
        "input_len": 1000,
        "max_jitter": 500
    }
    # Required: 1000 + 1000 = 2000
    
    sampler_config = {
        "padded_size": 1500
    }
    
    with pytest.raises(ValueError, match="Sampler padded_size \(1500\) is smaller than required size"):
        validate_data_and_sampler_compatibility(data_config, sampler_config)
