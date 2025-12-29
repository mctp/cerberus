import pytest
from cerberus.config import validate_train_config, TrainConfig

def test_validate_train_config_valid():
    config: TrainConfig = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {"T_max": 10}
    }
    validated = validate_train_config(config)
    assert validated == config

def test_validate_train_config_missing_keys():
    config = {"batch_size": 32}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_invalid_types():
    config = {
        "batch_size": "32", # Invalid type
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True
    }
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_invalid_values():
    config = {
        "batch_size": -1, # Invalid value
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True
    }
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_defaults():
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True
        # Missing scheduler, should default
    }
    validated = validate_train_config(config) # type: ignore
    assert validated["scheduler_type"] == "default"
    assert validated["scheduler_args"] == {}
