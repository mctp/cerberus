"""Coverage tests for cerberus.train — untested code paths."""
import pytest
import json
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from pytorch_lightning.callbacks import ModelCheckpoint

from cerberus.config import ModelConfig, DataConfig, TrainConfig, GenomeConfig, SamplerConfig
from cerberus.train import _dump_config, _save_model_pt


def _make_model_config(name: str = "test") -> ModelConfig:
    return ModelConfig(
        name=name,
        model_cls="x.Y",
        loss_cls="x.L",
        loss_args={},
        metrics_cls="x.M",
        metrics_args={},
        model_args={},
        pretrained=[],
    )


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


def _make_data_config_mock() -> MagicMock:
    """Mock DataConfig with model_dump support."""
    dc = MagicMock(spec=DataConfig)
    dc.input_len = 1000
    dc.model_dump.return_value = {"input_len": 1000}
    return dc


def _make_genome_config_mock() -> MagicMock:
    """Mock GenomeConfig with model_dump support."""
    gc = MagicMock(spec=GenomeConfig)
    gc.name = "hg38"
    gc.model_dump.return_value = {"name": "hg38"}
    return gc


def _make_sampler_config_mock() -> MagicMock:
    """Mock SamplerConfig with model_dump support."""
    sc = MagicMock(spec=SamplerConfig)
    sc.sampler_type = "random"
    sc.model_dump.return_value = {"sampler_type": "random"}
    return sc


# ---------------------------------------------------------------------------
# _dump_config
# ---------------------------------------------------------------------------

class TestDumpConfig:

    def test_writes_expected_json(self, tmp_path):
        model_config = _make_model_config()
        data_config = _make_data_config_mock()
        train_config = _make_train_config()

        _dump_config(tmp_path, model_config, data_config, train_config)

        config_path = tmp_path / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            data = json.load(f)
        assert data["model_config"]["name"] == "test"
        assert data["data_config"]["input_len"] == 1000
        assert data["train_config"]["batch_size"] == 32
        # genome_config and sampler_config should not be present when not provided
        assert "genome_config" not in data
        assert "sampler_config" not in data

    def test_includes_optional_configs(self, tmp_path):
        _dump_config(
            tmp_path,
            model_config=_make_model_config(name="m"),
            data_config=_make_data_config_mock(),
            train_config=_make_train_config(),
            genome_config=_make_genome_config_mock(),
            sampler_config=_make_sampler_config_mock(),
        )
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data["genome_config"]["name"] == "hg38"
        assert data["sampler_config"]["sampler_type"] == "random"

    def test_handles_path_objects(self, tmp_path):
        """Path objects should be serialized as strings via model_dump(mode='json')."""
        model_config = _make_model_config()
        data_config = _make_data_config_mock()
        train_config = _make_train_config()

        _dump_config(tmp_path, model_config, data_config, train_config)

        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        # model_dump(mode="json") serializes all fields properly
        assert isinstance(data["model_config"], dict)

    def test_handles_exception_gracefully(self, tmp_path):
        """If writing fails, _dump_config should not raise."""
        # Make root_dir a file (not a directory) to cause failure
        bad_path = tmp_path / "existing_file"
        bad_path.touch()
        # Trying to create a directory inside a file should fail gracefully
        _dump_config(
            bad_path / "sub",
            model_config=_make_model_config(),
            data_config=_make_data_config_mock(),
            train_config=_make_train_config(),
        )
        # Should not raise — just logs a warning


# ---------------------------------------------------------------------------
# _save_model_pt
# ---------------------------------------------------------------------------

class TestSaveModelPt:

    def test_no_best_checkpoint_skips(self, tmp_path):
        """If no best checkpoint, skip export without error."""
        trainer = MagicMock()
        trainer.checkpoint_callback = None
        _save_model_pt(trainer, tmp_path)
        assert not (tmp_path / "model.pt").exists()

    def test_saves_stripped_state_dict(self, tmp_path):
        """Normal model: strips 'model.' prefix."""
        # Create a fake checkpoint
        ckpt_path = tmp_path / "best.ckpt"
        state_dict = {
            "model.layer.weight": torch.randn(10, 10),
            "model.layer.bias": torch.randn(10),
            "criterion.param": torch.randn(5),  # should be excluded
        }
        torch.save({"state_dict": state_dict}, ckpt_path)

        ckpt_callback = MagicMock(spec=ModelCheckpoint)
        ckpt_callback.best_model_path = str(ckpt_path)

        trainer = MagicMock()
        trainer.checkpoint_callback = ckpt_callback

        _save_model_pt(trainer, tmp_path)

        pt_path = tmp_path / "model.pt"
        assert pt_path.exists()
        loaded = torch.load(pt_path, weights_only=True)
        assert "layer.weight" in loaded
        assert "layer.bias" in loaded
        assert "criterion.param" not in loaded

    def test_strips_compile_prefix(self, tmp_path):
        """Compiled model: strips both 'model.' and '_orig_mod.' prefixes."""
        ckpt_path = tmp_path / "best.ckpt"
        state_dict = {
            "model._orig_mod.layer.weight": torch.randn(10, 10),
            "model._orig_mod.layer.bias": torch.randn(10),
        }
        torch.save({"state_dict": state_dict}, ckpt_path)

        ckpt_callback = MagicMock(spec=ModelCheckpoint)
        ckpt_callback.best_model_path = str(ckpt_path)

        trainer = MagicMock()
        trainer.checkpoint_callback = ckpt_callback

        _save_model_pt(trainer, tmp_path)

        pt_path = tmp_path / "model.pt"
        assert pt_path.exists()
        loaded = torch.load(pt_path, weights_only=True)
        assert "layer.weight" in loaded
        assert "layer.bias" in loaded
        # No _orig_mod prefix
        assert not any(k.startswith("_orig_mod.") for k in loaded)
