import pytest
from cerberus.samplers import create_sampler, Sampler
from cerberus.config import SamplerConfig, validate_sampler_config
from typing import cast

def test_create_sampler_dummy_removed():
    """
    Verify that 'dummy' sampler type is no longer supported and raises ValueError.
    """
    config = {
        "sampler_type": "dummy",
        "padded_size": 100,
        "sampler_args": {}
    }
    
    # Validation should pass? No, I removed it from validate_sampler_config too?
    # Let's check src/cerberus/config.py
    # If I removed the "elif type == 'dummy'" block, it falls through to return dict.
    # It doesn't raise error for unknown types in validate_sampler_config, it just validates fields.
    # Wait, let's verify validate_sampler_config behavior.
    
    # create_sampler raises ValueError for unsupported types.
    
    with pytest.raises(ValueError, match="Unsupported sampler type: dummy"):
        create_sampler(cast(SamplerConfig, config), chrom_sizes={}, exclude_intervals={}, folds=[])

def test_create_sampler_unknown_type():
    config = {
        "sampler_type": "unknown_type",
        "padded_size": 100,
        "sampler_args": {}
    }
    with pytest.raises(ValueError, match="Unsupported sampler type: unknown_type"):
        create_sampler(cast(SamplerConfig, config), chrom_sizes={}, exclude_intervals={}, folds=[])

def test_validate_sampler_config_unknown_type():
    # validate_sampler_config only validates required fields for known types, 
    # but doesn't strictly reject unknown types if they have basic structure?
    # Let's verify this behavior or enforce strictness if needed.
    # Actually, current implementation of validate_sampler_config only has specific checks for known types.
    # It allows others to pass through if they have "sampler_type", "padded_size", "sampler_args".
    
    config = {
        "sampler_type": "unknown_type",
        "padded_size": 100,
        "sampler_args": {}
    }
    # This should pass validation, but fail creation.
    validated = validate_sampler_config(cast(SamplerConfig, config))
    assert validated["sampler_type"] == "unknown_type"

def test_create_sampler_random(tmp_path):
    config = {
        "sampler_type": "random",
        "padded_size": 100,
        "sampler_args": {"num_intervals": 10}
    }
    chrom_sizes = {"chr1": 1000}
    sampler = create_sampler(cast(SamplerConfig, config), chrom_sizes=chrom_sizes, exclude_intervals={}, folds=[])
    assert len(sampler) == 10

