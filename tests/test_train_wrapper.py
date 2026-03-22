import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import torch
import json
from cerberus.train import _train as train, _save_model_pt, train_single
from cerberus.config import TrainConfig, ModelConfig, DataConfig, GenomeConfig, SamplerConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def _make_train_config() -> TrainConfig:
    return TrainConfig(
        batch_size=32,
        max_epochs=10,
        learning_rate=1e-3,
        weight_decay=0.01,
        patience=5,
        optimizer="adamw",
        scheduler_type="default",
        scheduler_args={},
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )

def _make_model_config(loss_args: dict | None = None) -> ModelConfig:
    return ModelConfig(
        name="TestModel",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.models.bpnet.BPNetLoss",
        loss_args=loss_args if loss_args is not None else {"alpha": 1.0},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args={
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
        },
        pretrained=[],
    )

def _make_data_config() -> DataConfig:
    """Create a minimal DataConfig for testing.

    Uses MagicMock because DataConfig validators require real file paths.
    The mock supports attribute access for all fields used in _train.
    """
    dc = MagicMock(spec=DataConfig)
    dc.input_len = 2114
    dc.output_len = 1000
    dc.output_bin_size = 1
    dc.targets = {}
    dc.inputs = {}
    dc.use_sequence = True
    dc.target_scale = 1.0
    dc.max_jitter = 0
    dc.encoding = "ACGT"
    dc.log_transform = False
    dc.reverse_complement = False
    dc.model_dump.return_value = {
        "input_len": 2114, "output_len": 1000, "output_bin_size": 1,
        "targets": {}, "inputs": {}, "use_sequence": True,
        "target_scale": 1.0, "max_jitter": 0, "encoding": "ACGT",
        "log_transform": False, "reverse_complement": False,
    }
    return dc

def _make_genome_config(k: int = 3) -> MagicMock:
    """Create a MagicMock GenomeConfig with fold_args as a plain dict."""
    gc = MagicMock(spec=GenomeConfig)
    gc.fold_args = {"k": k, "test_fold": None, "val_fold": None}
    gc.model_copy = lambda update: gc  # fold_args override returns self-like mock
    gc.model_dump.return_value = {"fold_args": {"k": k}}
    return gc

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

def test_save_model_pt_strips_prefix():
    """_save_model_pt writes model.pt with 'model.' prefix stripped."""
    fake_state = {
        "model.conv.weight": torch.zeros(4, 4),
        "model.conv.bias": torch.zeros(4),
        "criterion.weight": torch.ones(1),  # non-model key — should be excluded
    }
    fake_ckpt = {"state_dict": fake_state}

    ckpt_callback = MagicMock(spec=ModelCheckpoint)
    ckpt_callback.best_model_path = "/fake/path/best.ckpt"

    trainer = MagicMock(spec=pl.Trainer)
    trainer.checkpoint_callback = ckpt_callback

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("cerberus.train.torch.load", return_value=fake_ckpt), \
             patch("cerberus.train.torch.save") as mock_save:
            _save_model_pt(trainer, tmp_dir)

        mock_save.assert_called_once()
        saved_sd, saved_path = mock_save.call_args[0]
        assert saved_path == Path(tmp_dir) / "model.pt"
        assert set(saved_sd.keys()) == {"conv.weight", "conv.bias"}

def test_save_model_pt_strips_compile_prefix():
    """_save_model_pt strips _orig_mod. prefix when model was torch.compiled."""
    fake_state = {
        "model._orig_mod.layer.weight": torch.zeros(2, 2),
        "model._orig_mod.layer.bias": torch.zeros(2),
    }
    fake_ckpt = {"state_dict": fake_state}

    ckpt_callback = MagicMock(spec=ModelCheckpoint)
    ckpt_callback.best_model_path = "/fake/path/best.ckpt"

    trainer = MagicMock(spec=pl.Trainer)
    trainer.checkpoint_callback = ckpt_callback

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("cerberus.train.torch.load", return_value=fake_ckpt), \
             patch("cerberus.train.torch.save") as mock_save:
            _save_model_pt(trainer, tmp_dir)

        saved_sd, _ = mock_save.call_args[0]
        assert set(saved_sd.keys()) == {"layer.weight", "layer.bias"}

def test_save_model_pt_skips_when_no_checkpoint():
    """_save_model_pt logs a warning and exits when best_model_path is empty."""
    ckpt_callback = MagicMock(spec=ModelCheckpoint)
    ckpt_callback.best_model_path = ""

    trainer = MagicMock(spec=pl.Trainer)
    trainer.checkpoint_callback = ckpt_callback

    with patch("cerberus.train.torch.save") as mock_save:
        _save_model_pt(trainer, "/some/dir")
        mock_save.assert_not_called()

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

def test_train_single_run_test_false():
    """run_test=False (default) must NOT call trainer.test()."""
    with patch("cerberus.train._train") as mock_train, \
         patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.update_ensemble_metadata"):

        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_train.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_single(
                genome_config=_make_genome_config(k=3),
                data_config=_make_data_config(),
                sampler_config=MagicMock(spec=SamplerConfig),
                model_config=_make_model_config(),
                train_config=_make_train_config(),
                test_fold=0,
                root_dir=tmp_dir,
                run_test=False,
            )

        mock_trainer.test.assert_not_called()

def test_train_single_run_test_true():
    """run_test=True calls trainer.test(datamodule=..., ckpt_path='best') when a best checkpoint exists."""
    with patch("cerberus.train._train") as mock_train, \
         patch("cerberus.train.CerberusDataModule") as mock_dm_cls, \
         patch("cerberus.train.update_ensemble_metadata"):

        mock_trainer = MagicMock(spec=pl.Trainer)
        # Simulate a ModelCheckpoint callback with a valid best path
        mock_ckpt_cb = MagicMock(spec=ModelCheckpoint)
        mock_ckpt_cb.best_model_path = "/path/to/best.ckpt"
        mock_trainer.checkpoint_callback = mock_ckpt_cb
        mock_train.return_value = mock_trainer
        mock_datamodule = mock_dm_cls.return_value

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_single(
                genome_config=_make_genome_config(k=3),
                data_config=_make_data_config(),
                sampler_config=MagicMock(spec=SamplerConfig),
                model_config=_make_model_config(),
                train_config=_make_train_config(),
                test_fold=0,
                root_dir=tmp_dir,
                run_test=True,
            )

        mock_trainer.test.assert_called_once_with(
            datamodule=mock_datamodule, ckpt_path="best"
        )

def test_train_single_run_test_skips_when_no_checkpoint():
    """run_test=True with no best checkpoint logs a warning and does not call trainer.test()."""
    with patch("cerberus.train._train") as mock_train, \
         patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.update_ensemble_metadata"):

        mock_trainer = MagicMock(spec=pl.Trainer)
        # Simulate no ModelCheckpoint (e.g. enable_checkpointing=False)
        mock_trainer.checkpoint_callback = None
        mock_train.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_single(
                genome_config=_make_genome_config(k=3),
                data_config=_make_data_config(),
                sampler_config=MagicMock(spec=SamplerConfig),
                model_config=_make_model_config(),
                train_config=_make_train_config(),
                test_fold=0,
                root_dir=tmp_dir,
                run_test=True,
            )

        mock_trainer.test.assert_not_called()
