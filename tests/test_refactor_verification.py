import pytest
from pathlib import Path
import yaml
from pydantic import ValidationError
from cerberus.config import (
    import_class,
    ModelConfig,
    DataConfig,
)
from cerberus.module import instantiate_model
from cerberus.model_ensemble import ModelEnsemble
import cerberus.model_ensemble
import torch
from unittest.mock import MagicMock, patch

# --- Config Tests ---

def test_model_dump_json_paths(tmp_path):
    """Test that model_dump(mode='json') converts Path objects to strings."""
    cons = tmp_path / "cons.bw"
    cons.touch()
    cfg = DataConfig(
        inputs={"cons": cons},
        targets={},
        input_len=100,
        output_len=50,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        use_sequence=True,
        target_scale=1.0,
    )
    sanitized = cfg.model_dump(mode="json")
    assert sanitized["inputs"]["cons"] == str(cons)
    assert isinstance(sanitized["inputs"]["cons"], str)
    assert sanitized["input_len"] == 100

def test_model_dump_json_scalars():
    """Test that model_dump(mode='json') preserves scalars."""
    cfg = ModelConfig(
        name="test_model",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.loss.PoissonMultinomialLoss",
        loss_args={},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={
            "input_channels": ["seq"],
            "output_channels": ["out"],
        },
        pretrained=[],
    )
    sanitized = cfg.model_dump(mode="json")
    assert sanitized["model_cls"] == "cerberus.models.bpnet.BPNet"
    assert sanitized["name"] == "test_model"
    assert isinstance(sanitized["name"], str)

def test_model_config_validation_valid():
    """Test construction of ModelConfig with valid string class names."""
    cfg = ModelConfig(
        name="test_model",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.loss.PoissonMultinomialLoss",
        loss_args={},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={
            "input_channels": ["seq"],
            "output_channels": ["out"],
        },
        pretrained=[],
    )
    assert cfg.model_cls == "cerberus.models.bpnet.BPNet"

def test_model_config_validation_extra_field_rejected():
    """Test that extra fields are rejected by Pydantic (extra='forbid')."""
    with pytest.raises(ValidationError):
        ModelConfig(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.PoissonMultinomialLoss",
            loss_args={},
            metrics_cls="torchmetrics.MetricCollection",
            metrics_args={},
            model_args={},
            pretrained=[],
            unexpected_field="bad",  # type: ignore[call-arg]
        )

def test_import_class_success():
    """Test dynamic import of a known class."""
    cls = import_class("pathlib.Path")
    assert cls is Path

    cls = import_class("cerberus.config.ModelConfig")
    assert cls is ModelConfig

def test_import_class_failure():
    """Test dynamic import failure cases."""
    with pytest.raises(ImportError):
        import_class("nonexistent.module.Class")

    with pytest.raises(ImportError):
        import_class("cerberus.config.NonExistentClass")

    with pytest.raises(ImportError):
        import_class(123) # type: ignore

# --- Entrypoints Tests ---

class DummyModel(torch.nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.kwargs = kwargs

def test_instantiate_model_with_pydantic_configs():
    """Test instantiate_model using Pydantic config objects."""

    with patch("cerberus.module.import_class") as mock_import:
        mock_import.return_value = DummyModel

        model_cfg = ModelConfig(
            name="test",
            model_cls="dummy.DummyModel",
            loss_cls="dummy.Loss",
            loss_args={},
            metrics_cls="dummy.Metrics",
            metrics_args={},
            model_args={"hidden_dim": 64},
            pretrained=[],
        )

        data_cfg = DataConfig.model_construct(
            inputs={},
            targets={},
            input_len=1000,
            output_len=200,
            output_bin_size=1,
            encoding="ACGT",
            max_jitter=0,
            log_transform=False,
            reverse_complement=False,
            target_scale=1.0,
            use_sequence=True,
        )

        model = instantiate_model(model_cfg, data_cfg)

        assert isinstance(model, DummyModel)
        assert model.input_len == 1000
        assert model.kwargs["hidden_dim"] == 64

        mock_import.assert_called_with("dummy.DummyModel")

# --- ModelEnsemble Tests ---

@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Creates a mock experiment directory structure."""
    root = tmp_path / "experiment"
    root.mkdir()

    # Metadata
    metadata = {"folds": [0, 1]}
    with open(root / "ensemble_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    # Fold 0
    (root / "fold_0").mkdir()
    (root / "fold_0" / "best-val_loss=0.1.ckpt").touch()

    # Fold 1
    (root / "fold_1").mkdir()
    (root / "fold_1" / "best-val_loss=0.2.ckpt").touch()

    # hparams.yaml (needed for config parsing if not provided)
    hparams = {
        "model_config": {"name": "test"},
        "data_config": {"output_len": 100, "output_bin_size": 1},
        "genome_config": {"fold_type": "chrom_partition", "chrom_sizes": {}, "fold_args": {}}
    }
    with open(root / "fold_0" / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f)

    return root

def _mock_cerberus_config():
    """Return a MagicMock that behaves like a CerberusConfig for ensemble tests."""
    mock_cfg = MagicMock()
    mock_cfg.model_config_ = MagicMock()
    mock_cfg.data_config = MagicMock()
    mock_cfg.data_config.output_len = 100
    mock_cfg.data_config.output_bin_size = 1
    mock_cfg.data_config.input_len = 1000
    mock_cfg.genome_config = MagicMock()
    mock_cfg.model_copy.return_value = mock_cfg
    return mock_cfg


def test_model_ensemble_init_missing_metadata(tmp_path):
    """Test initialization fails without ensemble_metadata.yaml."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="ensemble_metadata.yaml"):
        with patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
             patch("cerberus.model_ensemble.parse_hparams_config", return_value=_mock_cerberus_config()):
            ModelEnsemble(
                checkpoint_path=empty_dir,
                model_config={}, # type: ignore
                data_config={"output_len": 100, "output_bin_size": 1}, # type: ignore
                genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
            )

def test_model_ensemble_init_success(mock_checkpoint_dir):
    """Test successful initialization with mocked models."""

    with patch("cerberus.model_ensemble._ModelManager._load_model_ckpt") as mock_load, \
         patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value=_mock_cerberus_config()):
        mock_load.return_value = torch.nn.Linear(1, 1)

        with patch("cerberus.model_ensemble.create_genome_folds") as mock_folds:
            mock_folds.return_value = [{}, {}]

            ensemble = ModelEnsemble(
                checkpoint_path=mock_checkpoint_dir,
                model_config={}, # type: ignore
                data_config={"output_len": 100, "output_bin_size": 1}, # type: ignore
                genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
            )

            assert len(ensemble) == 2 # 2 folds
            assert "0" in ensemble
            assert "1" in ensemble

def test_model_ensemble_predict_empty_intervals(mock_checkpoint_dir):
    """Test predict_intervals with empty input."""
    with patch("cerberus.model_ensemble._ModelManager._load_model_ckpt") as mock_load, \
         patch("cerberus.model_ensemble.create_genome_folds") as mock_folds, \
         patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value=_mock_cerberus_config()):
        mock_load.return_value = torch.nn.Linear(1, 1)
        mock_folds.return_value = [{}, {}]

        ensemble = ModelEnsemble(
            checkpoint_path=mock_checkpoint_dir,
            model_config={}, # type: ignore
            data_config={"output_len": 100, "output_bin_size": 1}, # type: ignore
            genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
        )

        dataset = MagicMock()
        mock_data_config = MagicMock()
        mock_data_config.input_len = 100
        mock_data_config.output_len = 100
        mock_data_config.output_bin_size = 1
        dataset.data_config = mock_data_config

        with pytest.raises(RuntimeError, match="No results generated"):
            ensemble.predict_intervals([], dataset)

def test_model_ensemble_predict_output_intervals_empty(mock_checkpoint_dir):
    """Test predict_output_intervals with empty input."""
    with patch("cerberus.model_ensemble._ModelManager._load_model_ckpt") as mock_load, \
         patch("cerberus.model_ensemble.create_genome_folds") as mock_folds, \
         patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
         patch("cerberus.model_ensemble.parse_hparams_config", return_value=_mock_cerberus_config()):
        mock_load.return_value = torch.nn.Linear(1, 1)
        mock_folds.return_value = [{}, {}]

        ensemble = ModelEnsemble(
            checkpoint_path=mock_checkpoint_dir,
            model_config={}, # type: ignore
            data_config={"output_len": 100, "output_bin_size": 1, "input_len": 1000}, # type: ignore
            genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
        )

        dataset = MagicMock()
        mock_data_config = MagicMock()
        mock_data_config.input_len = 1000
        mock_data_config.output_len = 100
        mock_data_config.output_bin_size = 1
        dataset.data_config = mock_data_config

        results = ensemble.predict_output_intervals([], dataset)
        assert results == []

# --- Internal Method Tests ---

def test_model_manager_select_best_checkpoint():
    """Test _select_best_checkpoint logic."""
    with patch("cerberus.model_ensemble._ModelManager.__init__", return_value=None):
        manager = cerberus.model_ensemble._ModelManager() # type: ignore

        ckpts = [
            Path("ckpt-val_loss=0.5.ckpt"),
            Path("ckpt-val_loss=0.1.ckpt"), # Best
            Path("ckpt-val_loss=0.3.ckpt"),
            Path("ckpt-no_loss_info.ckpt")  # inf
        ]

        best = manager._select_best_checkpoint(ckpts)
        assert best.name == "ckpt-val_loss=0.1.ckpt"

def test_model_manager_select_best_checkpoint_tiebreaker():
    """Test tie-breaking by name for deterministic selection."""
    with patch("cerberus.model_ensemble._ModelManager.__init__", return_value=None):
        manager = cerberus.model_ensemble._ModelManager() # type: ignore

        ckpts = [
            Path("b-val_loss=0.1.ckpt"),
            Path("a-val_loss=0.1.ckpt"), # Should win by name
        ]

        best = manager._select_best_checkpoint(ckpts)
        assert best.name == "a-val_loss=0.1.ckpt"

def test_load_model_strips_prefix():
    """Test that _load_model_ckpt strips 'model.' prefix from state_dict keys."""
    with patch("cerberus.model_ensemble._ModelManager.__init__", return_value=None):
        manager = cerberus.model_ensemble._ModelManager() # type: ignore
        manager.device = torch.device("cpu")
        manager.model_config = {} # type: ignore
        manager.data_config = {} # type: ignore
        manager.cache = {}

        mock_model = MagicMock()

        state_dict = {
            "model.layer1.weight": torch.tensor([1.0]),
            "model.layer1.bias": torch.tensor([0.0]),
            "other_param": torch.tensor([2.0]) # Dropped: only "model." prefixed keys are kept
        }

        with patch("cerberus.model_ensemble.instantiate_model", return_value=mock_model), \
             patch("torch.load", return_value={"state_dict": state_dict}):

            manager._load_model_ckpt("fold_0", Path("dummy.ckpt"))

            # Verify load_state_dict call
            args, _ = mock_model.load_state_dict.call_args
            loaded_dict = args[0]

            assert "layer1.weight" in loaded_dict
            assert "layer1.bias" in loaded_dict
            assert "model.layer1.weight" not in loaded_dict
            assert "other_param" not in loaded_dict

def test_find_hparams_newest(tmp_path):
    """Test _find_hparams selects the most recently modified file."""
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()

    old_hparams = tmp_path / "old" / "hparams.yaml"
    old_hparams.touch()

    import time
    time.sleep(0.01)

    new_hparams = tmp_path / "new" / "hparams.yaml"
    new_hparams.touch()

    with patch("cerberus.model_ensemble.ModelEnsemble.__init__", return_value=None):
        ens = ModelEnsemble(None) # type: ignore
        found = ens._find_hparams(tmp_path)
        assert found.resolve() == new_hparams.resolve()

def test_find_hparams_missing(tmp_path):
    with patch("cerberus.model_ensemble.ModelEnsemble.__init__", return_value=None):
        ens = ModelEnsemble(None) # type: ignore
        with pytest.raises(FileNotFoundError, match="No hparams.yaml found"):
            ens._find_hparams(tmp_path)
