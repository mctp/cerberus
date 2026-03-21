import pytest
from pydantic import ValidationError
from cerberus.config import TrainConfig


def _complete_train_config(**overrides) -> dict:
    """Helper to build a complete TrainConfig dict with overrides."""
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


def test_validate_train_config_valid():
    cfg = TrainConfig(**_complete_train_config())
    assert cfg.batch_size == 32
    assert cfg.max_epochs == 10


def test_validate_train_config_missing_keys():
    with pytest.raises(ValidationError):
        TrainConfig(batch_size=32)  # type: ignore[call-arg]


def test_validate_train_config_invalid_types():
    with pytest.raises(ValidationError):
        TrainConfig(**_complete_train_config(batch_size=[1, 2, 3]))


def test_validate_train_config_invalid_values():
    with pytest.raises(ValidationError):
        TrainConfig(**_complete_train_config(batch_size=-1))


def test_validate_train_config_missing_scheduler_keys():
    """Missing scheduler_type, scheduler_args, reload_dataloaders_every_n_epochs raises ValidationError."""
    with pytest.raises(ValidationError):
        TrainConfig(
            batch_size=32,
            max_epochs=10,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=5,
            optimizer="adam",
            filter_bias_and_bn=True,
            adam_eps=1e-8,
            gradient_clip_val=None,
        )  # type: ignore[call-arg]


def test_validate_train_config_reload_dataloaders_invalid():
    with pytest.raises(ValidationError):
        TrainConfig(**_complete_train_config(reload_dataloaders_every_n_epochs=-1))


def test_validate_train_config_adam_eps_invalid():
    with pytest.raises(ValidationError):
        TrainConfig(**_complete_train_config(adam_eps=-1e-8))


def test_validate_train_config_gradient_clip_val_invalid():
    with pytest.raises(ValidationError):
        TrainConfig(**_complete_train_config(gradient_clip_val=-1.0))


def test_validate_train_config_missing_adam_eps():
    """Regression: missing adam_eps must raise ValidationError."""
    kw = _complete_train_config()
    del kw["adam_eps"]
    with pytest.raises(ValidationError):
        TrainConfig(**kw)


def test_validate_train_config_missing_gradient_clip_val_uses_default():
    """gradient_clip_val has a default of None, so omitting it is valid."""
    kw = _complete_train_config()
    del kw["gradient_clip_val"]
    cfg = TrainConfig(**kw)
    assert cfg.gradient_clip_val is None


def test_validate_train_config_gradient_clip_val_valid_float():
    cfg = TrainConfig(**_complete_train_config(gradient_clip_val=1.0))
    assert cfg.gradient_clip_val == 1.0
