import pytest
from pathlib import Path
import yaml
from cerberus.config import (
    _sanitize_config,
    validate_model_config,
    import_class,
    ModelConfig
)
from cerberus.module import instantiate_model
from cerberus.model_ensemble import ModelEnsemble
import cerberus.model_ensemble
import torch
from unittest.mock import MagicMock, patch

# --- Config Tests ---

def test_sanitize_config_paths():
    """Test that _sanitize_config converts Path objects to strings."""
    config = {
        "path": Path("/tmp/test"),
        "list": [Path("a"), Path("b")],
        "nested": {"p": Path("c")},
        "str": "already_string",
        "int": 123
    }
    
    sanitized = _sanitize_config(config)
    
    assert sanitized["path"] == "/tmp/test"
    assert sanitized["list"] == ["a", "b"]
    assert sanitized["nested"]["p"] == "c"
    assert sanitized["str"] == "already_string"
    assert sanitized["int"] == 123

def test_validate_model_config_strings():
    """Test validation of ModelConfig with string class names."""
    config: ModelConfig = {
        "name": "test_model",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.loss.PoissonLoss",
        "metrics_cls": "torchmetrics.MetricCollection",
        "metrics_args": {},
        "loss_args": {},
        "model_args": {
            "input_channels": ["seq"],
            "output_channels": ["out"]
        }
    }
    
    validated = validate_model_config(config)
    assert validated["model_cls"] == "cerberus.models.bpnet.BPNet"

def test_validate_model_config_invalid_types():
    """Test that validation fails if class objects are passed instead of strings."""
    class DummyClass: pass
    
    config = {
        "name": "test_model",
        "model_cls": DummyClass, # Invalid
        "loss_cls": "loss",
        "metrics_cls": "metrics",
        "metrics_args": {},
        "loss_args": {},
        "model_args": {}
    }
    
    with pytest.raises(TypeError, match="model_cls must be a string"):
        validate_model_config(config) # type: ignore

def test_import_class_success():
    """Test dynamic import of a known class."""
    # Using a class from the standard library or local project to test
    cls = import_class("pathlib.Path")
    assert cls is Path
    
    # Using a class from cerberus
    cls = import_class("cerberus.config.ModelConfig")
    assert cls is ModelConfig

def test_import_class_failure():
    """Test dynamic import failure cases."""
    with pytest.raises(ImportError):
        import_class("nonexistent.module.Class")
        
    with pytest.raises(ImportError):
        import_class("cerberus.config.NonExistentClass")
        
    with pytest.raises(TypeError):
        import_class(123) # type: ignore

# --- Entrypoints Tests ---

class DummyModel(torch.nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, **kwargs):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.kwargs = kwargs

def test_instantiate_model_with_strings():
    """Test instantiate_model using string configuration."""
    
    # We mock import_class to return our DummyModel
    # Note: instantiate_model moved to cerberus.module, so we patch import_class there
    with patch("cerberus.module.import_class") as mock_import:
        mock_import.return_value = DummyModel
        
        model_config: ModelConfig = {
            "name": "test",
            "model_cls": "dummy.DummyModel", # String
            "loss_cls": "dummy.Loss",
            "metrics_cls": "dummy.Metrics",
            "loss_args": {},
            "metrics_args": {},
            "model_args": {"hidden_dim": 64}
        }
        
        data_config = {
            "input_len": 1000,
            "output_len": 200,
            "output_bin_size": 1,
            "inputs": {},
            "targets": {},
            "encoding": "ACGT",
            "max_jitter": 0,
            "log_transform": False,
            "reverse_complement": False
        }
        
        model = instantiate_model(model_config, data_config) # type: ignore
        
        assert isinstance(model, DummyModel)
        assert model.input_len == 1000
        assert model.kwargs["hidden_dim"] == 64
        
        # Verify import_class was called
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

def test_model_ensemble_init_missing_metadata(tmp_path):
    """Test initialization fails without ensemble_metadata.yaml."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    # We mock _find_hparams and parse_hparams_config to bypass config loading issues
    # but verify it fails at metadata loading
    
    with pytest.raises(FileNotFoundError, match="ensemble_metadata.yaml"):
        # We need to provide configs to avoid it searching for hparams.yaml first
        ModelEnsemble(
            checkpoint_path=empty_dir,
            model_config={}, # type: ignore
            data_config={"output_len": 100, "output_bin_size": 1}, # type: ignore
            genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
        )

def test_model_ensemble_init_success(mock_checkpoint_dir):
    """Test successful initialization with mocked models."""
    
    # Mock _load_model to avoid actual checkpoint loading
    with patch("cerberus.model_ensemble._ModelManager._load_model") as mock_load:
        mock_load.return_value = torch.nn.Linear(1, 1)
        
        # Mock create_genome_folds to avoid complex logic
        with patch("cerberus.model_ensemble.create_genome_folds") as mock_folds:
            mock_folds.return_value = [{}, {}]
            
            # Mock parse_hparams_config if needed, but we provided hparams.yaml
            # Actually, let's provide configs explicitly to isolate metadata logic
            
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
    # Initialize ensemble (mocked)
    with patch("cerberus.model_ensemble._ModelManager._load_model") as mock_load, \
         patch("cerberus.model_ensemble.create_genome_folds") as mock_folds:
        mock_load.return_value = torch.nn.Linear(1, 1)
        mock_folds.return_value = [{}, {}]
        
        ensemble = ModelEnsemble(
            checkpoint_path=mock_checkpoint_dir,
            model_config={}, # type: ignore
            data_config={"output_len": 100, "output_bin_size": 1}, # type: ignore
            genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
        )
        
        dataset = MagicMock()
        dataset.data_config = {"input_len": 100, "output_len": 100}
        
        with pytest.raises(RuntimeError, match="No results generated"):
            ensemble.predict_intervals([], dataset)

def test_model_ensemble_predict_output_intervals_empty(mock_checkpoint_dir):
    """Test predict_output_intervals with empty input."""
    # Initialize ensemble (mocked)
    with patch("cerberus.model_ensemble._ModelManager._load_model") as mock_load, \
         patch("cerberus.model_ensemble.create_genome_folds") as mock_folds:
        mock_load.return_value = torch.nn.Linear(1, 1)
        mock_folds.return_value = [{}, {}]
        
        ensemble = ModelEnsemble(
            checkpoint_path=mock_checkpoint_dir,
            model_config={}, # type: ignore
            data_config={"output_len": 100, "output_bin_size": 1, "input_len": 1000}, # type: ignore
            genome_config={"fold_type": "x", "chrom_sizes": {}, "fold_args": {}}, # type: ignore
        )
        
        dataset = MagicMock()
        dataset.data_config = {"input_len": 1000, "output_len": 100}
        
        # Should return empty list, not raise error (since it's a list of results per interval)
        results = ensemble.predict_output_intervals([], dataset)
        assert results == []

# --- Internal Method Tests ---

def test_model_manager_select_best_checkpoint():
    """Test _select_best_checkpoint logic."""
    # We can test this by instantiating _ModelManager (mocking init or just using the method if static/bound)
    # It's an instance method but doesn't use self, so we can call it if we have an instance.
    
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
    """Test that _load_model strips 'model.' prefix from state_dict keys."""
    # Logic:
    # 1. Mock torch.load to return state_dict with "model.layer1.weight"
    # 2. Mock instantiate_model to return a mock model
    # 3. Call _load_model
    # 4. Verify model.load_state_dict was called with "layer1.weight"
    
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
            "other_param": torch.tensor([2.0]) # Should be kept? The code says: "if k.startswith('model.'): new[k[6:]] = v". It iterates state_dict. So "other_param" is IGNORED?
            # Code:
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     if k.startswith("model."):
            #         new_state_dict[k[6:]] = v
            # So "other_param" is DROPPED.
        }
        
        with patch("cerberus.model_ensemble.instantiate_model", return_value=mock_model), \
             patch("torch.load", return_value={"state_dict": state_dict}):
            
            manager._load_model("fold_0", Path("dummy.ckpt"))
            
            # Verify load_state_dict call
            args, _ = mock_model.load_state_dict.call_args
            loaded_dict = args[0]
            
            assert "layer1.weight" in loaded_dict
            assert "layer1.bias" in loaded_dict
            assert "model.layer1.weight" not in loaded_dict
            assert "other_param" not in loaded_dict

def test_find_hparams_newest(tmp_path):
    """Test _find_hparams selects the most recently modified file."""
    # Create structure
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    
    old_hparams = tmp_path / "old" / "hparams.yaml"
    old_hparams.touch()
    
    # Ensure time difference
    import time
    time.sleep(0.01)
    
    new_hparams = tmp_path / "new" / "hparams.yaml"
    new_hparams.touch()
    
    # Mock ModelEnsemble init just to get an instance, or use a dummy class if method allows
    # _find_hparams is an instance method but doesn't use self.
    
    with patch("cerberus.model_ensemble.ModelEnsemble.__init__", return_value=None):
        ens = ModelEnsemble(None) # type: ignore
        found = ens._find_hparams(tmp_path)
        assert found.resolve() == new_hparams.resolve()

def test_find_hparams_missing(tmp_path):
    with patch("cerberus.model_ensemble.ModelEnsemble.__init__", return_value=None):
        ens = ModelEnsemble(None) # type: ignore
        with pytest.raises(FileNotFoundError, match="No hparams.yaml found"):
            ens._find_hparams(tmp_path)
