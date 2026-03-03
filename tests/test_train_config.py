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
        "scheduler_args": {"T_max": 10},
        "reload_dataloaders_every_n_epochs": 1,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }
    validated = validate_train_config(config)
    assert validated == config

def test_validate_train_config_missing_keys():
    config = {"batch_size": 32}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config) # type: ignore

def _complete_train_config(**overrides):
    """Helper to build a complete TrainConfig with overrides."""
    base = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "scheduler_type": "default",
        "scheduler_args": {},
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }
    base.update(overrides)
    return base

def test_validate_train_config_invalid_types():
    config = _complete_train_config(batch_size="32")  # Invalid type
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_invalid_values():
    config = _complete_train_config(batch_size=-1)  # Invalid value
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_missing_scheduler_keys():
    """Missing scheduler_type, scheduler_args, reload_dataloaders_every_n_epochs raises ValueError."""
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_reload_dataloaders_invalid():
    config = _complete_train_config(reload_dataloaders_every_n_epochs=-1)
    with pytest.raises(ValueError, match="reload_dataloaders_every_n_epochs must be a non-negative integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_adam_eps_invalid():
    config = _complete_train_config(adam_eps=-1e-8)
    with pytest.raises(ValueError, match="adam_eps must be a positive float"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_gradient_clip_val_invalid():
    config = _complete_train_config(gradient_clip_val=-1.0)
    with pytest.raises(ValueError, match="gradient_clip_val must be a positive float or None"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_missing_adam_eps():
    """Regression: missing adam_eps must raise ValueError, not KeyError."""
    config = _complete_train_config()
    del config["adam_eps"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config)  # type: ignore

def test_validate_train_config_missing_gradient_clip_val():
    """Regression: missing gradient_clip_val must raise ValueError, not KeyError."""
    config = _complete_train_config()
    del config["gradient_clip_val"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config)  # type: ignore

def test_validate_train_config_gradient_clip_val_valid_float():
    config: TrainConfig = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "scheduler_type": "default",
        "scheduler_args": {},
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": 1.0, # Valid positive float
    }
    validated = validate_train_config(config)
    assert validated["gradient_clip_val"] == 1.0
