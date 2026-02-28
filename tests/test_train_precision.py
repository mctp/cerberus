from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.train import _train as train
from cerberus.config import TrainConfig, ModelConfig, DataConfig
import pytorch_lightning as pl


def _make_configs():
    train_config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 1,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    })
    model_config = cast(ModelConfig, {
        "name": "Test",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.models.bpnet.BPNetLoss",
        "loss_args": {"alpha": 1.0},
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
        "metrics_args": {},
        "model_args": {},
    })
    data_config = cast(DataConfig, {
        "input_len": 2114, "output_len": 1000, "output_bin_size": 1,
        "targets": [], "inputs": [], "use_sequence": True,
        "target_scale": 1.0, "count_pseudocount": 1.0, "jitter": 0,
    })
    return train_config, model_config, data_config


def test_train_wrapper_matmul_precision():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    train_config, model_config, data_config = _make_configs()

    # Test default (highest)
    with patch("pytorch_lightning.Trainer"), \
         patch("cerberus.train.instantiate", return_value=mock_module), \
         patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm, **kw: mc), \
         patch("torch.set_float32_matmul_precision") as mock_set_precision:

        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=0,
            in_memory=False,
            logger=False,
        )

        mock_set_precision.assert_called_once_with("highest")

    # Test medium
    with patch("pytorch_lightning.Trainer"), \
         patch("cerberus.train.instantiate", return_value=mock_module), \
         patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm, **kw: mc), \
         patch("torch.set_float32_matmul_precision") as mock_set_precision:

        train(
            model_config=model_config,
            data_config=data_config,
            datamodule=datamodule,
            train_config=train_config,
            num_workers=0,
            in_memory=False,
            logger=False,
            matmul_precision="medium",
        )

        mock_set_precision.assert_called_once_with("medium")
