import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from cerberus.module import CerberusModule
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.metrics import DefaultMetricCollection
from cerberus.output import ProfileLogRates

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    def forward(self, x):
        # Output (Batch, 1, 1) to match (Batch, Channels, Length) expectation
        logits = torch.abs(self.layer(x)).unsqueeze(-1)
        return ProfileLogRates(log_rates=logits)

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
        "filter_bias_and_bn": True,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }

def test_training_step(base_config):
    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=base_config)
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
    module.log.assert_any_call("train_loss", loss, prog_bar=True, batch_size=5, sync_dist=False)

def test_validation_step(base_config):
    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=base_config)
    module.log = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    module._trainer = mock_trainer  # attach a mock trainer so self.trainer resolves

    batch = {
        "inputs": torch.randn(5, 10),
        "targets": torch.abs(torch.randn(5, 1)).unsqueeze(-1)
    }

    loss = module.validation_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    module.log.assert_any_call("val_loss", loss, prog_bar=True, batch_size=5, sync_dist=True)

def test_on_validation_epoch_end(base_config):
    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=base_config)
    module.log_dict = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = False
    mock_trainer.current_epoch = 0
    module._trainer = mock_trainer

    # Simulate some updates
    # Use larger batch and length to avoid zero variance in Pearson calculations
    # Small batch size causes instability in LogCountsPearsonCorrCoef (vector length = batch size)
    preds = torch.randn(10, 1, 10)
    targets = torch.abs(torch.randn(10, 1, 10))
    # Use ProfileLogRates as DummyModel produces them, and metrics now support them
    module.val_metrics.update(ProfileLogRates(log_rates=preds), targets)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_trainer.logger.log_dir = tmp_dir
        module.on_validation_epoch_end()

    module.log_dict.assert_called()

    # Check arguments
    args, kwargs = module.log_dict.call_args
    metrics_arg = args[0]
    assert "val_pearson" in metrics_arg
    assert "val_mse_profile" in metrics_arg
    assert "val_mse_log_counts" in metrics_arg
    assert kwargs.get("sync_dist") is True
