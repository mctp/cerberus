import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from cerberus.model_ensemble import ModelEnsemble
import time

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
    # Setup directory structure
    root = tmp_path
    p1 = root / "hparams.yaml"
    p1.touch()
    
    subdir = root / "subdir"
    subdir.mkdir()
    p2 = subdir / "hparams.yaml"
    
    # Ensure p2 is newer
    time.sleep(0.1)
    p2.touch()
    
    # Helper to access method by creating instance with dummy configs
    with patch("cerberus.model_ensemble._ModelManager") as mock_mgr:
        instance = mock_mgr.return_value
        instance.load_models_and_folds.return_value = ({}, [])
        
        ens = ModelEnsemble(
             checkpoint_path=tmp_path,
             model_config={}, 
             data_config={"output_len": 1, "output_bin_size": 1}, 
             genome_config={}
        )
        found = ens._find_hparams(root)
        
    assert found.resolve() == p2.resolve()

def test_init_resolves_configs(tmp_path, mock_model_manager):
    # Setup hparams
    hparams_path = tmp_path / "hparams.yaml"
    hparams_path.touch()
    
    # Mock parse_hparams_config
    with patch("cerberus.model_ensemble.parse_hparams_config") as mock_parse:
        mock_parse.return_value = {
            "model_config": {"name": "parsed_model"},
            "data_config": {"output_len": 100, "output_bin_size": 1},
            "genome_config": {"name": "parsed_genome"},
        }
        
        ens = ModelEnsemble(checkpoint_path=tmp_path)
        
        # Verify parse called with found path
        mock_parse.assert_called_once()
        args, _ = mock_parse.call_args
        assert args[0].resolve() == hparams_path.resolve()
        
        # Verify _ModelManager init arguments
        mock_model_manager.assert_called_once()
        args, _ = mock_model_manager.call_args
        # args: (path, model_config, data_config, genome_config, device)
        assert args[1]["name"] == "parsed_model"
        assert args[2]["output_len"] == 100

def test_init_overrides_configs(tmp_path, mock_model_manager):
    hparams_path = tmp_path / "hparams.yaml"
    hparams_path.touch()
    
    with patch("cerberus.model_ensemble.parse_hparams_config") as mock_parse:
        mock_parse.return_value = {
            "model_config": {"name": "parsed_model"},
            "data_config": {"output_len": 100, "output_bin_size": 1},
            "genome_config": {"name": "parsed_genome"},
        }
        
        # Override model_config
        user_model_config = {"name": "user_model"}
        
        ens = ModelEnsemble(
            checkpoint_path=tmp_path,
            model_config=user_model_config
        )
        
        # Verify parse called (since others are missing)
        mock_parse.assert_called_once()
        
        # Verify configs passed to manager
        args, _ = mock_model_manager.call_args
        assert args[1]["name"] == "user_model" # Overridden
        assert args[2]["output_len"] == 100 # From hparams

def test_init_no_hparams_error(tmp_path):
    # Empty dir
    with pytest.raises(FileNotFoundError, match="No hparams.yaml found"):
        ModelEnsemble(checkpoint_path=tmp_path)
