import pytest
from typing import cast
from cerberus.config import validate_train_config, TrainConfig

def test_validate_train_config_valid():
    config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "scheduler_type": "default",
        "scheduler_args": {}
    })
    validated = validate_train_config(config)
    assert validated["batch_size"] == 32
    assert validated["optimizer"] == "adamw"
    assert validated["scheduler_type"] == "default"

def test_validate_train_config_valid_asap():
    config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adamw_asap",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {"warmup_steps": 100}
    })
    validated = validate_train_config(config)
    assert validated["optimizer"] == "adamw_asap"
    assert validated["scheduler_type"] == "cosine"
    assert validated["scheduler_args"]["warmup_steps"] == 100

def test_validate_train_config_valid_defaults():
    config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        # Missing scheduler_type and args should use defaults
    })
    validated = validate_train_config(config)
    assert validated["scheduler_type"] == "default"
    assert validated["scheduler_args"] == {}


def test_validate_train_config_missing_keys():
    config = cast(TrainConfig, {
        "batch_size": 32
        # Missing others
    })
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config)

def test_validate_train_config_invalid_types():
    config = cast(TrainConfig, {
        "batch_size": "32", # Should be int
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
    })
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_train_config(config)
