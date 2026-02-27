import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch
import yaml
from typing import cast
from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import ModelConfig, DataConfig, GenomeConfig
from cerberus.module import CerberusModule

# Define a simple dummy model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size, hidden_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_len * 4, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        # x: (B, 4, L)
        b, c, l = x.shape
        x_flat = x.view(b, -1)
        h = self.linear(x_flat)
        return self.output(h)

@pytest.fixture
def mock_ensemble_dir(tmp_path):
    ensemble_dir = tmp_path / "test_ensemble"
    ensemble_dir.mkdir()
    
    # 1. Create ensemble_metadata.yaml
    metadata = {"folds": [0]}
    with open(ensemble_dir / "ensemble_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
        
    # 2. Create fold_0 directory
    fold_dir = ensemble_dir / "fold_0"
    fold_dir.mkdir()
    
    # 3. Create a dummy checkpoint
    # We manually create a state_dict simulating CerberusModule (keys prefixed with "model.")
    model = SimpleModel(input_len=10, output_len=1, output_bin_size=1)
    # Set known weights
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)
        
    # Wrap in dict as if it was CerberusModule
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[f"model.{k}"] = v
        
    checkpoint = {"state_dict": state_dict}
    torch.save(checkpoint, fold_dir / "val_loss=0.01.ckpt")
    
    return ensemble_dir

def test_model_ensemble_loads_stripped_weights(mock_ensemble_dir):
    # Configs
    model_config = {
        "name": "SimpleModel",
        "model_cls": "tests.test_model_loading_optimization.SimpleModel", # This needs to be resolvable
        "loss_cls": "torch.nn.MSELoss", # Dummy
        "metrics_cls": "torchmetrics.MeanSquaredError", # Dummy
        "loss_args": {},
        "metrics_args": {},
        "model_args": {"hidden_dim": 10}
    }
    
    data_config = {
        "inputs": {},
        "targets": {},
        "input_len": 10,
        "output_len": 1,
        "output_bin_size": 1,
        "max_jitter": 0,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True
    }
    
    genome_config = {
        "name": "hg38",
        "fasta_path": Path("genome.fa"),
        "exclude_intervals": {},
        "allowed_chroms": ["chr1"],
        "chrom_sizes": {"chr1": 1000},
        "fold_type": "chrom_partition",
        "fold_args": {"k": 1}
    }
    
    # Note: import_class relies on importlib. 
    # To make "tests.test_model_loading_optimization.SimpleModel" resolvable, 
    # we rely on pytest adding current dir to path, but "tests" might not be a package.
    # We can use a real class available in cerberus or patch import_class.
    # For robust testing, let's use a mock or a real simple class if available.
    # But here I want to test MY SimpleModel.
    
    # Let's try to monkeypatch import_class for this test to return our SimpleModel
    from cerberus import module
    
    original_import = module.import_class
    
    def mock_import(name):
        if name == "tests.test_model_loading_optimization.SimpleModel":
            return SimpleModel
        return original_import(name)
        
    module.import_class = mock_import
    
    try:
        with patch("cerberus.model_ensemble.ModelEnsemble._find_hparams", return_value=Path("hparams.yaml")), \
             patch("cerberus.model_ensemble.parse_hparams_config", return_value={}):
            ensemble = ModelEnsemble(
                mock_ensemble_dir,
                model_config=cast(ModelConfig, model_config),
                data_config=cast(DataConfig, data_config),
                genome_config=cast(GenomeConfig, genome_config),
                device="cpu"
            )
        
        # Verify model loaded
        assert "0" in ensemble
        loaded_model = ensemble["0"]
        
        # Check if weights are 1.0 (as set in fixture)
        # Cast to SimpleModel to access .linear
        simple_model = cast(SimpleModel, loaded_model)
        assert torch.all(simple_model.linear.weight == 1.0)
        
        # Verify it is NOT a CerberusModule
        assert isinstance(loaded_model, SimpleModel)
        assert not isinstance(loaded_model, CerberusModule)
        
    finally:
        # Restore
        module.import_class = original_import
