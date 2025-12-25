import pytest
from typing import cast
from cerberus.config import validate_data_and_sampler_compatibility, DataConfig, SamplerConfig

def test_validate_data_and_sampler_compatibility_valid():
    data_config = cast(DataConfig, {
        "input_len": 100,
        "max_jitter": 50,
        "output_len": 50,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "inputs": {},
        "targets": {},
        "use_sequence": True,
    })
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 200,
        "sampler_args": {}
    })
    
    # Should not raise
    validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_exact_boundary():
    # padded_size == input_len + 2 * max_jitter
    data_config = cast(DataConfig, {
        "input_len": 100,
        "max_jitter": 50,
        # Other fields ignored by this validation function but required for typing if strictly checked (here we pass dict)
    })
    sampler_config = cast(SamplerConfig, {
        "padded_size": 200
    })
    
    validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_invalid():
    # padded_size < input_len + 2 * max_jitter
    data_config = cast(DataConfig, {
        "input_len": 100,
        "max_jitter": 50
    })
    sampler_config = cast(SamplerConfig, {
        "padded_size": 199
    })
    
    with pytest.raises(ValueError, match=r"Sampler padded_size \(199\) is smaller than required size"):
        validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_large_jitter():
    data_config = cast(DataConfig, {
        "input_len": 1000,
        "max_jitter": 500
    })
    # Required: 1000 + 1000 = 2000
    
    sampler_config = cast(SamplerConfig, {
        "padded_size": 1500
    })
    
    with pytest.raises(ValueError, match=r"Sampler padded_size \(1500\) is smaller than required size"):
        validate_data_and_sampler_compatibility(data_config, sampler_config)
