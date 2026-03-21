import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import (
    ModelConfig, DataConfig, GenomeConfig, SamplerConfig, TrainConfig,
    CerberusConfig, FoldArgs, RandomSamplerArgs,
)
import time


def _make_cerberus_config(**overrides):
    """Create a minimal CerberusConfig for hparams tests."""
    dc = DataConfig.model_construct(
        inputs={}, targets={}, input_len=100, output_len=100,
        output_bin_size=1, max_jitter=0, encoding="ACGT",
        log_transform=False, reverse_complement=False,
        target_scale=1.0, use_sequence=True,
    )
    gc = GenomeConfig.model_construct(
        name="parsed_genome", fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000}, allowed_chroms=["chr1"],
        exclude_intervals={}, fold_type="chrom_partition",
        fold_args=FoldArgs.model_construct(k=5, test_fold=None, val_fold=None),
    )
    sc = SamplerConfig.model_construct(
        sampler_type="random", padded_size=100,
        sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
    )
    tc = TrainConfig.model_construct(
        batch_size=1, max_epochs=1, learning_rate=1e-3, weight_decay=0.0,
        patience=5, optimizer="adam", scheduler_type="default", scheduler_args={},
        filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0, adam_eps=1e-8,
        gradient_clip_val=None,
    )
    mc = ModelConfig.model_construct(
        name="parsed_model", model_cls="torch.nn.Linear",
        loss_cls="torch.nn.MSELoss", loss_args={},
        metrics_cls="torchmetrics.MeanSquaredError", metrics_args={},
        model_args={}, pretrained=[], count_pseudocount=0.0,
    )
    return CerberusConfig.model_construct(
        data_config=dc, genome_config=gc, sampler_config=sc,
        train_config=tc, model_config_=mc,
    )


@pytest.fixture
def mock_model_manager():
    with patch("cerberus.model_ensemble._ModelManager") as mock:
        instance = mock.return_value
        instance.load_models_and_folds.return_value = ({}, [])
        yield mock

def test_init_check_directory(tmp_path):
    p = tmp_path / "file.ckpt"
    p.touch()
    with pytest.raises(ValueError, match="must be a directory"):
        ModelEnsemble(checkpoint_path=p)

def test_find_hparams(tmp_path):
    root = tmp_path
    p1 = root / "hparams.yaml"
    p1.touch()

    subdir = root / "subdir"
    subdir.mkdir()
    p2 = subdir / "hparams.yaml"

    time.sleep(0.1)
    p2.touch()

    cerberus_config = _make_cerberus_config()
    with patch("cerberus.model_ensemble._ModelManager") as mock_mgr, \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value=cerberus_config):
        instance = mock_mgr.return_value
        instance.load_models_and_folds.return_value = ({}, [])

        ens = ModelEnsemble(checkpoint_path=tmp_path)
        found = ens._find_hparams(root)

    assert found.resolve() == p2.resolve()

def test_init_resolves_configs(tmp_path, mock_model_manager):
    hparams_path = tmp_path / "hparams.yaml"
    hparams_path.touch()

    cerberus_config = _make_cerberus_config()
    with patch("cerberus.model_ensemble.parse_hparams_config") as mock_parse:
        mock_parse.return_value = cerberus_config

        ens = ModelEnsemble(checkpoint_path=tmp_path)

        mock_parse.assert_called_once()
        args, _ = mock_parse.call_args
        assert args[0].resolve() == hparams_path.resolve()

        # Verify _ModelManager init arguments
        mock_model_manager.assert_called_once()
        args, _ = mock_model_manager.call_args
        assert args[1].name == "parsed_model"
        assert args[2].output_len == 100

def test_init_overrides_configs(tmp_path, mock_model_manager):
    hparams_path = tmp_path / "hparams.yaml"
    hparams_path.touch()

    cerberus_config = _make_cerberus_config()
    with patch("cerberus.model_ensemble.parse_hparams_config") as mock_parse:
        mock_parse.return_value = cerberus_config

        user_model_config = ModelConfig.model_construct(
            name="user_model", model_cls="torch.nn.Linear",
            loss_cls="torch.nn.MSELoss", loss_args={},
            metrics_cls="torchmetrics.MeanSquaredError", metrics_args={},
            model_args={}, pretrained=[], count_pseudocount=0.0,
        )

        ens = ModelEnsemble(
            checkpoint_path=tmp_path,
            model_config=user_model_config,
        )

        mock_parse.assert_called_once()

        args, _ = mock_model_manager.call_args
        assert args[1].name == "user_model"  # Overridden
        assert args[2].output_len == 100  # From hparams

def test_init_no_hparams_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No hparams.yaml found"):
        ModelEnsemble(checkpoint_path=tmp_path)
