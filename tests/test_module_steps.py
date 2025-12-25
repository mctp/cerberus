import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from cerberus.module import CerberusModule
from cerberus.loss import get_default_loss, get_default_metrics

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    def forward(self, x):
        # Output (Batch, 1, 1) to match (Batch, Channels, Length) expectation
        return torch.abs(self.layer(x)).unsqueeze(-1)

@pytest.fixture
def base_config():
    return {
        "batch_size": 10,
        "max_epochs": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 2,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "num_workers": 2,
        "filter_bias_and_bn": True,
        "in_memory": False,
    }

def test_training_step(base_config):
    model = DummyModel()
    module = CerberusModule(model, base_config, criterion=get_default_loss(), metrics=get_default_metrics())
    module.log = MagicMock()
    
    # Batch: inputs (B, 10), targets (B, 1, 1)
    # Poisson loss requires non-negative targets
    batch = {
        "inputs": torch.randn(5, 10),
        "targets": torch.abs(torch.randn(5, 1)).unsqueeze(-1)
    }
    
    loss = module.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    # Check that log was called
    # Note: PL logs with on_step=True by default in training_step usually, but here we call self.log explicitly
    module.log.assert_called_with("train_loss", loss, prog_bar=True)

def test_validation_step(base_config):
    model = DummyModel()
    module = CerberusModule(model, base_config, criterion=get_default_loss(), metrics=get_default_metrics())
    module.log = MagicMock()
    
    batch = {
        "inputs": torch.randn(5, 10),
        "targets": torch.abs(torch.randn(5, 1)).unsqueeze(-1)
    }
    
    loss = module.validation_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    module.log.assert_called_with("val_loss", loss, prog_bar=True)

def test_on_validation_epoch_end(base_config):
    model = DummyModel()
    module = CerberusModule(model, base_config, criterion=get_default_loss(), metrics=get_default_metrics())
    module.log_dict = MagicMock()
    
    # Simulate some updates
    # Use (Batch, Channels=1, Length=1) shape
    preds = torch.tensor([1.0, 2.0]).view(2, 1, 1)
    targets = torch.tensor([1.0, 2.0]).view(2, 1, 1)
    module.val_metrics.update(preds, targets)
    
    module.on_validation_epoch_end()
    
    module.log_dict.assert_called()
    # Check if metrics are logged (pearson, mse)
    call_args = module.log_dict.call_args[0][0]
    assert "val_pearson" in call_args
    assert "val_mse" in call_args
