from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from cerberus.config import (
    DataConfig,
    GenomeConfig,
    ModelConfig,
    TrainConfig,
)
from cerberus.model_ensemble import (
    _ModelManager as ModelManager,
    _extract_backbone_state_dict_from_lightning,
    _parse_val_loss,
    find_latest_hparams,
    load_backbone_weights_from_checkpoint,
    load_backbone_weights_from_fold_dir,
    select_best_checkpoint,
)


# Dummy Configs
@pytest.fixture
def mock_configs():
    model_config = ModelConfig.model_construct(
        name="dummy",
        model_cls=MagicMock(),
        loss_cls=MagicMock(),
        loss_args={},
        metrics_cls=MagicMock(),
        metrics_args={},
        model_args={},
        pretrained=[],
        count_pseudocount=0.0,
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    train_config = TrainConfig.model_construct(
        batch_size=1,
        max_epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        patience=1,
        optimizer="adam",
        filter_bias_and_bn=False,
        scheduler_type="default",
        scheduler_args={},
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )
    genome_config = GenomeConfig.model_construct(
        name="test",
        fasta_path=Path("test.fa"),
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1000},
        fold_type="chrom_partition",
        fold_args={"k": 2},
    )
    return model_config, data_config, train_config, genome_config


@pytest.fixture
def model_manager(mock_configs, tmp_path):
    m_conf, d_conf, t_conf, g_conf = mock_configs

    # Create required metadata
    import yaml

    with open(tmp_path / "ensemble_metadata.yaml", "w") as f:
        yaml.dump({"folds": ["single"]}, f)

    return ModelManager(
        checkpoint_path=tmp_path,
        model_config=m_conf,
        data_config=d_conf,
        genome_config=g_conf,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Tests for module-level helpers
# ---------------------------------------------------------------------------


class TestParseValLoss:
    def test_standard_filename(self, tmp_path: Path) -> None:
        p = tmp_path / "checkpoint-epoch=01-val_loss=0.50.ckpt"
        assert _parse_val_loss(p) == 0.5

    def test_underscore_separator(self, tmp_path: Path) -> None:
        p = tmp_path / "checkpoint-epoch=01-val_loss_0.30.ckpt"
        assert _parse_val_loss(p) == 0.3

    def test_no_match_returns_inf(self, tmp_path: Path) -> None:
        p = tmp_path / "model.ckpt"
        assert _parse_val_loss(p) == float("inf")


class TestSelectBestCheckpoint:
    def test_selects_lowest_val_loss(self, tmp_path: Path) -> None:
        p1 = tmp_path / "checkpoint-epoch=01-val_loss=0.50.ckpt"
        p2 = tmp_path / "checkpoint-epoch=02-val_loss=0.30.ckpt"
        p3 = tmp_path / "checkpoint-epoch=03-val_loss=0.40.ckpt"
        p4 = tmp_path / "checkpoint-no-metric.ckpt"
        for p in [p1, p2, p3, p4]:
            p.touch()
        assert select_best_checkpoint([p1, p2, p3, p4]) == p2

    def test_tiebreak_by_name(self, tmp_path: Path) -> None:
        p5 = tmp_path / "a.ckpt"
        p6 = tmp_path / "b.ckpt"
        p5.touch()
        p6.touch()
        assert select_best_checkpoint([p6, p5]) == p5

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No checkpoints"):
            select_best_checkpoint([])


class TestFindLatestHparams:
    def test_finds_nested(self, tmp_path: Path) -> None:
        import time

        p1 = tmp_path / "hparams.yaml"
        p1.touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        time.sleep(0.05)
        p2 = subdir / "hparams.yaml"
        p2.touch()
        assert find_latest_hparams(tmp_path).resolve() == p2.resolve()

    def test_raises_if_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No hparams.yaml"):
            find_latest_hparams(tmp_path)


class TestExtractBackboneStateDict:
    def test_strips_model_prefix(self) -> None:
        raw = {
            "model.layer.weight": torch.tensor([1.0]),
            "model.layer.bias": torch.tensor([0.0]),
            "optimizer.state": torch.tensor([99.0]),
        }
        result = _extract_backbone_state_dict_from_lightning(raw)
        assert set(result.keys()) == {"layer.weight", "layer.bias"}

    def test_strips_orig_mod_prefix(self) -> None:
        raw = {"model._orig_mod.layer.weight": torch.tensor([1.0])}
        result = _extract_backbone_state_dict_from_lightning(raw)
        assert "layer.weight" in result

    def test_empty_when_no_model_keys(self) -> None:
        raw = {"optimizer.lr": torch.tensor([0.001])}
        assert _extract_backbone_state_dict_from_lightning(raw) == {}


class TestLoadBackboneWeightsFromCheckpoint:
    def test_loads_pt(self, tmp_path: Path) -> None:
        model = nn.Linear(2, 2)
        pt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), pt_path)

        fresh = nn.Linear(2, 2)
        load_backbone_weights_from_checkpoint(fresh, pt_path, "cpu")
        for p1, p2 in zip(model.parameters(), fresh.parameters()):
            assert torch.equal(p1, p2)

    def test_loads_ckpt(self, tmp_path: Path) -> None:
        model = nn.Linear(2, 2)
        ckpt_path = tmp_path / "model.ckpt"
        state = {"model." + k: v for k, v in model.state_dict().items()}
        torch.save({"state_dict": state}, ckpt_path)

        fresh = nn.Linear(2, 2)
        load_backbone_weights_from_checkpoint(fresh, ckpt_path, "cpu")
        for p1, p2 in zip(model.parameters(), fresh.parameters()):
            assert torch.equal(p1, p2)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.bin"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            load_backbone_weights_from_checkpoint(nn.Linear(2, 2), bad, "cpu")


class TestLoadBackboneWeightsFromFoldDir:
    def test_prefers_pt(self, tmp_path: Path) -> None:
        model = nn.Linear(2, 2)
        pt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), pt_path)
        # Also place a ckpt that would fail if loaded
        (tmp_path / "bad.ckpt").touch()

        fresh = nn.Linear(2, 2)
        used = load_backbone_weights_from_fold_dir(fresh, tmp_path, "cpu")
        assert used == pt_path

    def test_falls_back_to_ckpt(self, tmp_path: Path) -> None:
        model = nn.Linear(2, 2)
        ckpt_path = tmp_path / "checkpoint-val_loss=0.1.ckpt"
        state = {"model." + k: v for k, v in model.state_dict().items()}
        torch.save({"state_dict": state}, ckpt_path)

        fresh = nn.Linear(2, 2)
        used = load_backbone_weights_from_fold_dir(fresh, tmp_path, "cpu")
        assert used == ckpt_path

    def test_no_checkpoints_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No model.pt or .ckpt"):
            load_backbone_weights_from_fold_dir(nn.Linear(2, 2), tmp_path, "cpu")


# ---------------------------------------------------------------------------
# Tests for _ModelManager (integration with extracted helpers)
# ---------------------------------------------------------------------------


def test_load_model_from_fold(model_manager, tmp_path):
    """_load_model_from_fold delegates to load_backbone_weights_from_fold_dir."""
    fold_dir = tmp_path / "fold_test"
    fold_dir.mkdir()
    pt_path = fold_dir / "model.pt"
    pt_path.touch()

    with (
        patch("cerberus.model_ensemble.instantiate_model") as mock_instantiate,
        patch(
            "cerberus.model_ensemble.load_backbone_weights_from_fold_dir"
        ) as mock_load,
    ):
        mock_model = MagicMock()
        mock_instantiate.return_value = mock_model
        mock_load.return_value = pt_path

        model = model_manager._load_model_from_fold("test_key", fold_dir)

        mock_load.assert_called_once_with(
            model=mock_model,
            fold_dir=fold_dir,
            device=model_manager.device,
            strict=True,
        )
        assert model == mock_model


def test_load_model_caching(model_manager, tmp_path):
    cached_model = MagicMock()
    model_manager.cache["cached_key"] = cached_model

    fold_dir = tmp_path / "fold_cached"
    fold_dir.mkdir()

    with patch("cerberus.model_ensemble.instantiate_model") as mock_instantiate:
        model = model_manager._load_model_from_fold("cached_key", fold_dir)
        assert model == cached_model
        mock_instantiate.assert_not_called()


def test_load_models_and_folds_delegates(model_manager, tmp_path):
    """load_models_and_folds delegates to _load_model_from_fold per fold."""
    fold_dir = tmp_path / "fold_single"
    fold_dir.mkdir()
    (fold_dir / "model.pt").touch()

    with patch.object(model_manager, "_load_model_from_fold") as mock_load:
        mock_load.return_value = MagicMock(spec=nn.Module)

        models, folds = model_manager.load_models_and_folds()

        assert isinstance(models, dict)
        assert len(models) == 1
        assert "single" in models
        mock_load.assert_called_once_with("fold_single", fold_dir)


def test_load_models_and_folds_missing_dir(model_manager, tmp_path):
    """Missing fold directory is logged and skipped."""
    # fold_single dir does not exist
    models, folds = model_manager.load_models_and_folds()
    assert len(models) == 0
