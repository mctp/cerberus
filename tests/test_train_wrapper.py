from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.train import _train as train
from cerberus.config import TrainConfig, ModelConfig, DataConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def _make_train_config() -> TrainConfig:
    return cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 5,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    })


def _make_model_config(loss_args: dict | None = None) -> ModelConfig:
    return cast(ModelConfig, {
        "name": "TestModel",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.models.bpnet.BPNetLoss",
        "loss_args": loss_args if loss_args is not None else {"alpha": 1.0},
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
        },
    })


def _make_data_config() -> DataConfig:
    return cast(DataConfig, {
        "input_len": 2114,
        "output_len": 1000,
        "output_bin_size": 1,
        "targets": [],
        "inputs": [],
        "use_sequence": True,
        "target_scale": 1.0,
        "jitter": 0,
    })


def test_train_wrapper_calls_trainer_fit():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    model_config = _make_model_config()
    data_config = _make_data_config()
    train_config = _make_train_config()

    with patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
         patch("cerberus.train.instantiate", return_value=mock_module) as mock_instantiate, \
         patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm, **kw: mc):

        mock_trainer_instance = mock_trainer_cls.return_value

        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=2,
            in_memory=False,
            accelerator="cpu",
        )

        # Verify Trainer init
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs["max_epochs"] == 10
        assert call_kwargs["accelerator"] == "cpu"

        # Verify default callbacks
        callbacks = call_kwargs["callbacks"]
        assert len(callbacks) >= 3
        callback_types = [type(cb) for cb in callbacks]
        assert LearningRateMonitor in callback_types
        assert ModelCheckpoint in callback_types
        assert EarlyStopping in callback_types

        # Verify fit called with the mock module
        mock_trainer_instance.fit.assert_called_once_with(mock_module, datamodule=datamodule)

        # Verify datamodule setup called with runtime params
        datamodule.setup.assert_called_once_with(
            batch_size=32,
            val_batch_size=None,
            num_workers=2,
            in_memory=False,
        )

        # Verify instantiate was called with the resolved model_config
        mock_instantiate.assert_called_once()


def test_train_wrapper_custom_callbacks():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    model_config = _make_model_config()
    data_config = _make_data_config()
    train_config = _make_train_config()
    custom_cb = MagicMock(spec=pl.Callback)

    with patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
         patch("cerberus.train.instantiate", return_value=mock_module), \
         patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm, **kw: mc):

        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=2,
            in_memory=False,
            callbacks=[custom_cb],
        )

        call_kwargs = mock_trainer_cls.call_args[1]
        assert custom_cb in call_kwargs["callbacks"]
