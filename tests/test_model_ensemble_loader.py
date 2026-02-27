import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch.nn as nn
import torch

from cerberus.model_ensemble import _ModelManager as ModelManager, ModelEnsemble
from cerberus.config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
    GenomeConfig,
)

# Dummy Configs
@pytest.fixture
def mock_configs():
    model_config: ModelConfig = {
        "name": "dummy",
        "model_cls": MagicMock(),
        "loss_cls": MagicMock(),
        "loss_args": {},
        "metrics_cls": MagicMock(),
        "metrics_args": {},
        "model_args": {}
    } # type: ignore
    data_config: DataConfig = {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 100,
        "max_jitter": 0,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True
    }
    train_config: TrainConfig = {
        "batch_size": 1,
        "max_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "optimizer": "adam",
        "filter_bias_and_bn": False,
        "scheduler_type": "default",
        "scheduler_args": {}
    }
    genome_config: GenomeConfig = {
        "name": "test",
        "fasta_path": Path("test.fa"),
        "exclude_intervals": {},
        "allowed_chroms": ["chr1"],
        "chrom_sizes": {"chr1": 1000},
        "fold_type": "chrom_partition",
        "fold_args": {"k": 2}
    }
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
        device=torch.device("cpu")
    )

def test_select_best_checkpoint(model_manager, tmp_path):
    # Setup dummy files
    p1 = tmp_path / "checkpoint-epoch=01-val_loss=0.50.ckpt"
    p2 = tmp_path / "checkpoint-epoch=02-val_loss=0.30.ckpt" # Best
    p3 = tmp_path / "checkpoint-epoch=03-val_loss=0.40.ckpt"
    p4 = tmp_path / "checkpoint-no-metric.ckpt" # Should be treated as inf
    
    for p in [p1, p2, p3, p4]:
        p.touch()
        
    checkpoints = [p1, p2, p3, p4]
    
    # Run selection
    best = model_manager._select_best_checkpoint(checkpoints)
    
    # Assert
    assert best == p2
    
    # Test case with only no-metric files (should pick by name sorting)
    p5 = tmp_path / "a.ckpt"
    p6 = tmp_path / "b.ckpt"
    p5.touch()
    p6.touch()
    
    best_no_metric = model_manager._select_best_checkpoint([p6, p5])
    assert best_no_metric == p5 # 'a.ckpt' comes before 'b.ckpt'

def test_load_model_file(model_manager, tmp_path):
    # Setup file
    p2 = tmp_path / "ckpt2-val_loss=0.1.ckpt" 
    p2.touch()
    
    with patch("cerberus.model_ensemble.instantiate_model") as mock_instantiate_model, \
         patch("torch.load") as mock_load:
        
        mock_model = MagicMock()
        mock_instantiate_model.return_value = mock_model
        mock_load.return_value = {"state_dict": {}}
        
        # Run loading from FILE
        model = model_manager._load_model("test_dir", p2)
        
        # Should have loaded p2
        mock_load.assert_called_with(p2, map_location=model_manager.device, weights_only=False)
        assert model == mock_model

def test_load_model_caching(model_manager, tmp_path):
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()
    
    # Pre-populate cache
    cached_model = MagicMock()
    model_manager.cache["cached_key"] = cached_model
    
    # Run
    # Should return cached without calling instantiate_model/load
    with patch("cerberus.model_ensemble.instantiate_model") as mock_instantiate_model:
        model = model_manager._load_model("cached_key", ckpt_path)
        
        assert model == cached_model
        mock_instantiate_model.assert_not_called()

def test_load_models_and_folds(model_manager, tmp_path):
    # Setup dummy single fold
    # We configured "folds": ["single"] in fixture
    fold_dir = tmp_path / "fold_single"
    fold_dir.mkdir()
    ckpt_path = fold_dir / "model.ckpt"
    ckpt_path.touch()
    
    # Mock _load_model to avoid instantiating real models
    with patch.object(model_manager, "_load_model") as mock_load:
        mock_load.return_value = MagicMock(spec=nn.Module)
        
        models, folds = model_manager.load_models_and_folds()
        
        assert isinstance(models, dict)
        assert len(models) == 1
        assert "single" in models
        # It finds the checkpoint
        mock_load.assert_called_with("fold_single", ckpt_path)

def test_load_models_and_folds_multifold(model_manager, tmp_path):
    # Update metadata for multifold
    import yaml
    with open(tmp_path / "ensemble_metadata.yaml", "w") as f:
        yaml.dump({"folds": [0]}, f)
        
    # Reload metadata (hack since model_manager is already init)
    model_manager.fold_indices = [0]
    
    # Setup fold directories
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    p1 = fold_dir / "model.ckpt"
    p1.touch()
    
    with patch.object(model_manager, "_load_model") as mock_load:
        mock_load.return_value = MagicMock(spec=nn.Module)
        
        models, folds = model_manager.load_models_and_folds()
        
        assert isinstance(models, dict)
        assert len(models) == 1
        # Should attempt to load fold 0
        mock_load.assert_called()
        # Verify it passed the checkpoint path selected
        args, _ = mock_load.call_args
        assert args[0] == "fold_0" # Key used for caching/loading
        assert args[1] == p1
