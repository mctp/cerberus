from unittest.mock import MagicMock, patch

import pytorch_lightning as pl

from cerberus.config import DataConfig, ModelConfig, TrainConfig
from cerberus.train import _train as train


def _make_configs():
    train_config = TrainConfig(
        batch_size=32,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        patience=1,
        optimizer="adamw",
        scheduler_type="default",
        scheduler_args={},
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )
    model_config = ModelConfig(
        name="Test",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.models.bpnet.BPNetLoss",
        loss_args={"alpha": 1.0},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args={},
        pretrained=[],
    )
    data_config = MagicMock(spec=DataConfig)
    data_config.input_len = 2114
    data_config.output_len = 1000
    data_config.output_bin_size = 1
    data_config.targets = {}
    data_config.inputs = {}
    data_config.use_sequence = True
    data_config.target_scale = 1.0
    data_config.max_jitter = 0
    data_config.model_dump.return_value = {}
    return train_config, model_config, data_config


def test_train_wrapper_logger_setup():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    train_config, model_config, data_config = _make_configs()

    with (
        patch("pytorch_lightning.Trainer") as mock_trainer_cls,
        patch("pytorch_lightning.loggers.CSVLogger") as mock_csv_logger_cls,
        patch("cerberus.train.instantiate", return_value=mock_module),
        patch(
            "cerberus.train.resolve_adaptive_loss_args",
            side_effect=lambda mc, dm, **kw: mc,
        ),
    ):
        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=0,
            in_memory=False,
            logger=True,
            default_root_dir="/tmp/logs",
        )

        # Verify CSVLogger was instantiated with correct dir
        mock_csv_logger_cls.assert_called_once_with(save_dir="/tmp/logs")

        # Verify Trainer was called with the logger instance
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs["logger"] == mock_csv_logger_cls.return_value


def test_train_wrapper_logger_setup_default_dir():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    train_config, model_config, data_config = _make_configs()

    with (
        patch("pytorch_lightning.Trainer"),
        patch("pytorch_lightning.loggers.CSVLogger") as mock_csv_logger_cls,
        patch("cerberus.train.instantiate", return_value=mock_module),
        patch(
            "cerberus.train.resolve_adaptive_loss_args",
            side_effect=lambda mc, dm, **kw: mc,
        ),
    ):
        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=0,
            in_memory=False,
            logger=True,
        )

        # Verify CSVLogger was instantiated with "."
        mock_csv_logger_cls.assert_called_once_with(save_dir=".")
