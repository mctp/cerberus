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

def test_validate_train_config_invalid_types():
    config = {
        "batch_size": "32", # Invalid type
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
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
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
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
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
        # Missing scheduler and reload_dataloaders, should default
    }
    validated = validate_train_config(config) # type: ignore
    assert validated["scheduler_type"] == "default"
    assert validated["scheduler_args"] == {}
    assert validated["reload_dataloaders_every_n_epochs"] == 0

def test_validate_train_config_reload_dataloaders_invalid():
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
        "reload_dataloaders_every_n_epochs": -1, # Invalid
    }
    with pytest.raises(ValueError, match="reload_dataloaders_every_n_epochs must be a non-negative integer"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_adam_eps_invalid():
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "adam_eps": -1e-8, # Invalid: negative
        "gradient_clip_val": None,
    }
    with pytest.raises(ValueError, match="adam_eps must be a positive float"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_gradient_clip_val_invalid():
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": -1.0, # Invalid: negative
    }
    with pytest.raises(ValueError, match="gradient_clip_val must be a positive float or None"):
        validate_train_config(config) # type: ignore

def test_validate_train_config_missing_adam_eps():
    """Regression: missing adam_eps must raise ValueError, not KeyError."""
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "gradient_clip_val": None,
        # adam_eps intentionally missing
    }
    with pytest.raises(ValueError, match="missing required keys"):
        validate_train_config(config)  # type: ignore

def test_validate_train_config_missing_gradient_clip_val():
    """Regression: missing gradient_clip_val must raise ValueError, not KeyError."""
    config = {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        # gradient_clip_val intentionally missing
    }
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
