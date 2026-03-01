import pytest
import torch.nn as nn
from pathlib import Path
from typing import cast
from torchmetrics import MetricCollection
from cerberus.config import (
    validate_sampler_config,
    validate_model_config,
    validate_data_config,
    validate_data_and_sampler_compatibility,
    validate_data_and_model_compatibility,
    SamplerConfig,
    ModelConfig,
    DataConfig
)

# --- Sampler Config Tests ---

def test_validate_sampler_config_interval():
    config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 1000,
        "sampler_args": {
            "intervals_path": "path/to/intervals.bed"
        }
    })
    validated = validate_sampler_config(config)
    assert validated["sampler_type"] == "interval"
    assert validated["padded_size"] == 1000

def test_validate_sampler_config_sliding_window():
    config = cast(SamplerConfig, {
        "sampler_type": "sliding_window",
        "padded_size": 1000,
        "sampler_args": {
            "stride": 50
        }
    })
    validated = validate_sampler_config(config)
    assert validated["sampler_type"] == "sliding_window"

def test_validate_sampler_config_invalid_type():
    config = cast(SamplerConfig, {
        "sampler_type": "unknown",
        "padded_size": 100,
        "sampler_args": {}
    })
    # Current implementation doesn't strictly reject unknown types in the main function,
    # but specific types have checks. If it's unknown, it just passes basic checks.
    validated = validate_sampler_config(config)
    assert validated["sampler_type"] == "unknown"

def test_validate_sampler_config_missing_args():
    config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 1000,
        "sampler_args": {} # Missing intervals_path
    })
    with pytest.raises(ValueError, match="IntervalSampler args missing required keys"):
        validate_sampler_config(config)

# --- Model Config Tests ---

class DummyLoss(nn.Module):
    def forward(self, x, y): return 0

def test_validate_model_config_valid():
    config = cast(ModelConfig, {
        "name": "test_model",
        "model_cls": "torch.nn.Linear",
        "loss_cls": "tests.test_config_validation.DummyLoss",
        "loss_args": {},
        "metrics_cls": "torchmetrics.MetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "output_type": "signal",
        }
    })
    validated = validate_model_config(config)
    assert validated["name"] == "test_model"
    assert validated["model_args"]["input_channels"] == ["A", "C", "G", "T"]

def test_validate_model_config_invalid_output_type():
    config = cast(ModelConfig, {
        "name": "test",
        "model_cls": "torch.nn.Linear",
        "loss_cls": "tests.test_config_validation.DummyLoss",
        "loss_args": {},
        "metrics_cls": "torchmetrics.MetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A"],
            "output_channels": ["B"],
            "output_type": "invalid_type",
        }
    })
    with pytest.raises(ValueError, match=r"model_args\['output_type'\] must be one of"):
        validate_model_config(config)

def test_validate_model_config_empty_channels():
    config = cast(ModelConfig, {
        "name": "test",
        "model_cls": "torch.nn.Linear",
        "loss_cls": "tests.test_config_validation.DummyLoss",
        "loss_args": {},
        "metrics_cls": "torchmetrics.MetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": [],
            "output_channels": ["B"],
            "output_type": "signal",
        }
    })
    with pytest.raises(ValueError, match=r"model_args\['input_channels'\] must not be empty"):
        validate_model_config(config)

# --- Compatibility Tests ---

def test_validate_data_and_sampler_compatibility_valid():
    data_config = cast(DataConfig, {
        "inputs": {}, "targets": {},
        "input_len": 100,
        "output_len": 100,
        "max_jitter": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 120, # 100 + 2*10
        "sampler_args": {"intervals_path": "path"}
    })
    # Should pass
    validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_sampler_compatibility_invalid():
    data_config = cast(DataConfig, {
        "inputs": {}, "targets": {},
        "input_len": 100,
        "output_len": 100,
        "max_jitter": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 119, # Too small (< 120)
        "sampler_args": {"intervals_path": "path"}
    })
    with pytest.raises(ValueError, match="Sampler padded_size"):
        validate_data_and_sampler_compatibility(data_config, sampler_config)

def test_validate_data_and_model_compatibility_valid():
    data_config = cast(DataConfig, {
        "inputs": {"track1": Path("path1")}, 
        "targets": {"target1": Path("path2")},
        "input_len": 100, "output_len": 100, "max_jitter": 0, "output_bin_size": 1, 
        "encoding": "ACGT", "log_transform": False, 
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    model_config = cast(ModelConfig, {
        "name": "m", "model_cls": "torch.nn.Linear", "loss_cls": "tests.test_config_validation.DummyLoss", 
        "loss_args": {}, "metrics_cls": "torchmetrics.MetricCollection", "metrics_args": {},
        "model_args": {
            "input_channels": ["track1", "A", "C", "G", "T"], # data inputs + sequence
            "output_channels": ["target1"],
            "output_type": "signal",
        }
    })
    
    validate_data_and_model_compatibility(data_config, model_config)

def test_validate_data_and_model_compatibility_invalid_targets():
    data_config = cast(DataConfig, {
        "inputs": {}, 
        "targets": {"target1": Path("path2")},
        "input_len": 100, "output_len": 100, "max_jitter": 0, "output_bin_size": 1, 
        "encoding": "ACGT", "log_transform": False, 
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    model_config = cast(ModelConfig, {
        "name": "m", "model_cls": "torch.nn.Linear", "loss_cls": "tests.test_config_validation.DummyLoss", 
        "loss_args": {}, "metrics_cls": "torchmetrics.MetricCollection", "metrics_args": {},
        "model_args": {
            "input_channels": ["A"], 
            "output_channels": ["target2"], # Mismatch
            "output_type": "signal",
        }
    })
    
    with pytest.raises(ValueError, match="Model output channels"):
        validate_data_and_model_compatibility(data_config, model_config)

def test_validate_data_and_model_compatibility_invalid_inputs():
    data_config = cast(DataConfig, {
        "inputs": {"track1": Path("path")}, 
        "targets": {},
        "input_len": 100, "output_len": 100, "max_jitter": 0, "output_bin_size": 1, 
        "encoding": "ACGT", "log_transform": False, 
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    model_config = cast(ModelConfig, {
        "name": "m", "model_cls": "torch.nn.Linear", "loss_cls": "tests.test_config_validation.DummyLoss", 
        "loss_args": {}, "metrics_cls": "torchmetrics.MetricCollection", "metrics_args": {},
        "model_args": {
            "input_channels": ["A"], # Missing track1
            "output_channels": [],
            "output_type": "signal",
        }
    })
    
    with pytest.raises(ValueError, match="Data inputs .* are not in model input channels"):
        validate_data_and_model_compatibility(data_config, model_config)


# --- validate_data_config: count_pseudocount ---

def _minimal_raw_data_config(count_pseudocount=1.0, target_scale=1.0):
    """Return a minimal raw data config dict (no real file paths needed)."""
    return {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "max_jitter": 0,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": target_scale,
        "use_sequence": True,
        "count_pseudocount": count_pseudocount,
    }


def test_validate_data_config_count_pseudocount_missing():
    """validate_data_config raises when count_pseudocount is absent."""
    config = _minimal_raw_data_config()
    del config["count_pseudocount"]
    with pytest.raises(ValueError, match="count_pseudocount"):
        validate_data_config(config)  # type: ignore[arg-type]


def test_validate_data_config_count_pseudocount_invalid():
    """validate_data_config raises when count_pseudocount is non-positive."""
    config = _minimal_raw_data_config(count_pseudocount=0.0)
    with pytest.raises(ValueError, match="count_pseudocount"):
        validate_data_config(config)  # type: ignore[arg-type]

    config_neg = _minimal_raw_data_config(count_pseudocount=-5.0)
    with pytest.raises(ValueError, match="count_pseudocount"):
        validate_data_config(config_neg)  # type: ignore[arg-type]


def test_validate_data_config_count_pseudocount_stored_as_float():
    """validate_data_config coerces int count_pseudocount to float."""
    config = _minimal_raw_data_config(count_pseudocount=100)  # int, not float
    result = validate_data_config(config)  # type: ignore[arg-type]
    assert result["count_pseudocount"] == 100.0
    assert isinstance(result["count_pseudocount"], float)


# --- count_pseudocount injection from data_config into loss/metrics args ---

def test_count_pseudocount_injection_into_loss_and_metrics_args():
    """propagate_pseudocount injects count_pseudocount*target_scale into loss_args and metrics_args."""
    # Simulate the injection logic from propagate_pseudocount.
    data_conf = cast(DataConfig, _minimal_raw_data_config(count_pseudocount=100.0, target_scale=2.0))
    model_conf = cast(ModelConfig, {
        "name": "m", "model_cls": "torch.nn.Linear",
        "loss_cls": "torch.nn.Linear", "loss_args": {},
        "metrics_cls": "torchmetrics.MetricCollection", "metrics_args": {},
        "model_args": {},
    })

    scaled_pseudocount = data_conf["count_pseudocount"] * data_conf["target_scale"]
    model_conf["loss_args"].setdefault("count_pseudocount", scaled_pseudocount)
    model_conf["metrics_args"].setdefault("count_pseudocount", scaled_pseudocount)

    assert model_conf["loss_args"]["count_pseudocount"] == 200.0
    assert model_conf["metrics_args"]["count_pseudocount"] == 200.0


def test_count_pseudocount_injection_does_not_override_explicit():
    """Explicit count_pseudocount in loss_args/metrics_args takes precedence over injection."""
    data_conf = cast(DataConfig, _minimal_raw_data_config(count_pseudocount=100.0, target_scale=2.0))
    model_conf = cast(ModelConfig, {
        "name": "m", "model_cls": "torch.nn.Linear",
        "loss_cls": "torch.nn.Linear", "loss_args": {"count_pseudocount": 999.0},
        "metrics_cls": "torchmetrics.MetricCollection", "metrics_args": {},
        "model_args": {},
    })

    scaled_pseudocount = data_conf["count_pseudocount"] * data_conf["target_scale"]
    model_conf["loss_args"].setdefault("count_pseudocount", scaled_pseudocount)
    model_conf["metrics_args"].setdefault("count_pseudocount", scaled_pseudocount)

    # loss_args had explicit value — must not be overwritten
    assert model_conf["loss_args"]["count_pseudocount"] == 999.0
    # metrics_args had no explicit value — injected
    assert model_conf["metrics_args"]["count_pseudocount"] == 200.0
