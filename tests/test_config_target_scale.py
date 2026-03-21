import pytest
from pathlib import Path
from pydantic import ValidationError
from cerberus.config import DataConfig


def _base_config(tmp_path: Path, **overrides) -> dict:
    """Create a minimal valid DataConfig dict for testing."""
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
    return config


def test_target_scale_default_value(tmp_path):
    """target_scale=1.0 should validate successfully."""
    cfg = DataConfig(**_base_config(tmp_path, target_scale=1.0))
    assert cfg.target_scale == 1.0


def test_target_scale_large_value(tmp_path):
    """Large target_scale values (e.g. 1000.0) should be valid."""
    cfg = DataConfig(**_base_config(tmp_path, target_scale=1000.0))
    assert cfg.target_scale == 1000.0


def test_target_scale_fractional(tmp_path):
    """Fractional target_scale should be valid."""
    cfg = DataConfig(**_base_config(tmp_path, target_scale=0.001))
    assert cfg.target_scale == 0.001


def test_target_scale_zero_rejected(tmp_path):
    """target_scale=0.0 should be rejected."""
    with pytest.raises(ValidationError, match="target_scale"):
        DataConfig(**_base_config(tmp_path, target_scale=0.0))


def test_target_scale_negative_rejected(tmp_path):
    """Negative target_scale should be rejected."""
    with pytest.raises(ValidationError, match="target_scale"):
        DataConfig(**_base_config(tmp_path, target_scale=-1.0))


def test_target_scale_int_coerced(tmp_path):
    """Integer target_scale should be coerced to float by Pydantic."""
    cfg = DataConfig(**_base_config(tmp_path, target_scale=1))
    assert cfg.target_scale == 1.0
    assert isinstance(cfg.target_scale, float)


def test_target_scale_missing_rejected(tmp_path):
    """Missing target_scale should be rejected (required field)."""
    kw = _base_config(tmp_path)
    del kw["target_scale"]
    with pytest.raises(ValidationError, match="target_scale"):
        DataConfig(**kw)


def test_target_scale_preserved_in_output(tmp_path):
    """Validated config should preserve the target_scale value."""
    cfg = DataConfig(**_base_config(tmp_path, target_scale=42.5))
    assert cfg.target_scale == 42.5
    assert isinstance(cfg.target_scale, float)
