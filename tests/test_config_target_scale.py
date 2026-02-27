import pytest
from pathlib import Path
from typing import cast
from cerberus.config import validate_data_config, DataConfig


def _base_config(tmp_path: Path, **overrides) -> DataConfig:
    """Create a minimal valid data config for testing."""
    cons = tmp_path / "cons.bw"
    cons.touch()
    config = {
        "inputs": {"cons": str(cons)},
        "targets": {},
        "input_len": 1024,
        "output_len": 512,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    }
    config.update(overrides)
    return cast(DataConfig, config)


def test_target_scale_default_value(tmp_path):
    """target_scale=1.0 should validate successfully."""
    config = _base_config(tmp_path, target_scale=1.0)
    validated = validate_data_config(config)
    assert validated["target_scale"] == 1.0


def test_target_scale_large_value(tmp_path):
    """Large target_scale values (e.g. 1000.0) should be valid."""
    config = _base_config(tmp_path, target_scale=1000.0)
    validated = validate_data_config(config)
    assert validated["target_scale"] == 1000.0


def test_target_scale_fractional(tmp_path):
    """Fractional target_scale should be valid."""
    config = _base_config(tmp_path, target_scale=0.001)
    validated = validate_data_config(config)
    assert validated["target_scale"] == 0.001


def test_target_scale_zero_rejected(tmp_path):
    """target_scale=0.0 should be rejected."""
    config = _base_config(tmp_path, target_scale=0.0)
    with pytest.raises(ValueError, match="target_scale must be a positive number"):
        validate_data_config(config)


def test_target_scale_negative_rejected(tmp_path):
    """Negative target_scale should be rejected."""
    config = _base_config(tmp_path, target_scale=-1.0)
    with pytest.raises(ValueError, match="target_scale must be a positive number"):
        validate_data_config(config)


def test_target_scale_int_rejected(tmp_path):
    """Integer target_scale should be rejected (must be float)."""
    config = _base_config(tmp_path, target_scale=1)
    with pytest.raises(ValueError, match="target_scale must be a positive number"):
        validate_data_config(config)


def test_target_scale_missing_rejected(tmp_path):
    """Missing target_scale should be rejected (required field)."""
    config = _base_config(tmp_path)
    del config["target_scale"]  # type: ignore
    with pytest.raises(ValueError, match="missing required keys.*target_scale"):
        validate_data_config(config)


def test_target_scale_preserved_in_output(tmp_path):
    """Validated config should preserve the target_scale value."""
    config = _base_config(tmp_path, target_scale=42.5)
    validated = validate_data_config(config)
    assert validated["target_scale"] == 42.5
    assert isinstance(validated["target_scale"], float)
