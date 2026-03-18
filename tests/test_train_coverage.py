"""Coverage tests for cerberus.train — untested code paths."""
import pytest
import json
import torch
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch
from pytorch_lightning.callbacks import ModelCheckpoint

from cerberus.config import ModelConfig, DataConfig, TrainConfig, GenomeConfig, SamplerConfig
from cerberus.train import _dump_config, _save_model_pt


# ---------------------------------------------------------------------------
# _dump_config
# ---------------------------------------------------------------------------

class TestDumpConfig:

    def test_writes_expected_json(self, tmp_path):
        model_config = cast(ModelConfig, {"name": "test", "model_cls": "x.Y", "pretrained": []})
        data_config = cast(DataConfig, {"input_len": 1000})
        train_config = cast(TrainConfig, {"batch_size": 32})

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
            model_config=cast(ModelConfig, {"name": "m"}),
            data_config=cast(DataConfig, {"x": 1}),
            train_config=cast(TrainConfig, {"y": 2}),
            genome_config=cast(GenomeConfig, {"name": "hg38"}),
            sampler_config=cast(SamplerConfig, {"sampler_type": "random"}),
        )
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data["genome_config"]["name"] == "hg38"
        assert data["sampler_config"]["sampler_type"] == "random"

    def test_handles_path_objects(self, tmp_path):
        """Path objects should be serialized as strings via default=str."""
        _dump_config(
            tmp_path,
            model_config=cast(ModelConfig, {"path": Path("/some/path")}),
            data_config=cast(DataConfig, {}),
            train_config=cast(TrainConfig, {}),
        )
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data["model_config"]["path"] == "/some/path"

    def test_handles_exception_gracefully(self, tmp_path):
        """If writing fails, _dump_config should not raise."""
        # Make root_dir a file (not a directory) to cause failure
        bad_path = tmp_path / "existing_file"
        bad_path.touch()
        # Trying to create a directory inside a file should fail gracefully
        _dump_config(
            bad_path / "sub",
            model_config=cast(ModelConfig, {}),
            data_config=cast(DataConfig, {}),
            train_config=cast(TrainConfig, {}),
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
