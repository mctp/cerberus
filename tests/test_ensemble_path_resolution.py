
import pytest
from pathlib import Path
import yaml
from unittest.mock import MagicMock, patch
from cerberus.train import train_single, update_ensemble_metadata

def test_train_single_updates_metadata(tmp_path):
    """Test that train_single updates ensemble_metadata.yaml correctly."""
    
    # Mock dependencies to avoid actual training
    with patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.instantiate"), \
         patch("cerberus.train._train") as mock_train:
             
        root_dir = tmp_path / "exp"
        
        # Call train_single for fold 0
        train_single(
            genome_config={"fold_args": {"k": 5}}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={}, # type: ignore
            model_config={}, # type: ignore
            train_config={}, # type: ignore
            test_fold=0,
            root_dir=root_dir
        )
        
        # Check metadata
        meta_path = root_dir / "ensemble_metadata.yaml"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0]
        
        # Call train_single for fold 1
        train_single(
            genome_config={"fold_args": {"k": 5}}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={}, # type: ignore
            model_config={}, # type: ignore
            train_config={}, # type: ignore
            test_fold=1,
            root_dir=root_dir
        )
        
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0, 1]
