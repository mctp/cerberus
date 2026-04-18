from unittest.mock import MagicMock, patch

import pytorch_lightning as pl

from cerberus.config import DataConfig, ModelConfig, TrainConfig
from cerberus.train import _train as train
from cerberus.utils import get_precision_kwargs


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


def test_train_wrapper_matmul_precision():
    mock_module = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    train_config, model_config, data_config = _make_configs()

    # Test default (highest)
    with (
        patch("pytorch_lightning.Trainer"),
        patch("cerberus.train.instantiate", return_value=mock_module),
        patch(
            "cerberus.train.resolve_adaptive_loss_args",
            side_effect=lambda mc, dm, **kw: mc,
        ),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
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
    with (
        patch("pytorch_lightning.Trainer"),
        patch("cerberus.train.instantiate", return_value=mock_module),
        patch(
            "cerberus.train.resolve_adaptive_loss_args",
            side_effect=lambda mc, dm, **kw: mc,
        ),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
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


def test_get_precision_kwargs_full_mode():
    kwargs = get_precision_kwargs("full", "gpu", 1)

    assert kwargs == {
        "precision": "32-true",
        "matmul_precision": "highest",
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "compile": False,
    }


def test_get_precision_kwargs_mps_mode():
    kwargs = get_precision_kwargs("mps", "mps", 1)

    assert kwargs == {
        "precision": "16-mixed",
        "accelerator": "mps",
        "devices": 1,
        "strategy": "auto",
        "compile": False,
    }


def test_get_precision_kwargs_bf16_single_device():
    kwargs = get_precision_kwargs("bf16", "gpu", 1)

    assert kwargs == {
        "precision": "bf16-mixed",
        "matmul_precision": "medium",
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "benchmark": True,
        "compile": True,
    }


def test_get_precision_kwargs_bf16_multi_gpu_uses_ddp_override():
    kwargs = get_precision_kwargs("bf16", "gpu", 2)

    assert kwargs["strategy"] == "ddp_find_unused_parameters_false"
    assert kwargs["precision"] == "bf16-mixed"
    assert kwargs["matmul_precision"] == "medium"


def test_get_precision_kwargs_can_disable_ddp_override():
    kwargs = get_precision_kwargs(
        "bf16",
        "gpu",
        2,
        use_ddp_find_unused_parameters_false=False,
    )

    assert kwargs["strategy"] == "auto"
