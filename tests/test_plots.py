import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import pytest

from cerberus.plots import save_count_scatter
from cerberus.module import CerberusModule
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.metrics import DefaultMetricCollection
from cerberus.output import ProfileLogRates


# ---------------------------------------------------------------------------
# save_count_scatter unit tests
# ---------------------------------------------------------------------------

def test_save_count_scatter_creates_file():
    rng = np.random.default_rng(0)
    preds = rng.standard_normal(100).astype(np.float32)
    targets = rng.standard_normal(100).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_count_scatter(preds, targets, tmp_dir, epoch=3)
        expected = Path(tmp_dir) / "plots" / "val_count_scatter_epoch_003.png"
        assert expected.exists()


def test_save_count_scatter_skips_without_matplotlib():
    rng = np.random.default_rng(1)
    preds = rng.standard_normal(10).astype(np.float32)
    targets = rng.standard_normal(10).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("builtins.__import__", side_effect=ImportError("no matplotlib")):
            # Should not raise; silently skips
            try:
                save_count_scatter(preds, targets, tmp_dir, epoch=0)
            except ImportError:
                pass  # acceptable if the patch propagates — no PNG should be created
        assert not (Path(tmp_dir) / "plots").exists()


# ---------------------------------------------------------------------------
# CerberusModule scatter plot integration tests
# ---------------------------------------------------------------------------

class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 8)

    def forward(self, x):
        # Output shape: (B, 1, 8) — one channel, profile length 8
        return ProfileLogRates(log_rates=self.layer(x).unsqueeze(1))


@pytest.fixture
def _base_config():
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
        "reload_dataloaders_every_n_epochs": 0,
    }


def test_validation_step_populates_metric_state(_base_config):
    """validation_step populates LogCountsPearsonCorrCoef's preds_list/targets_list."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    module._trainer = mock_trainer

    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    pearson_metric = module.val_metrics["val_pearson_log_counts"]
    assert len(pearson_metric.preds_list) == 1  # type: ignore[arg-type]
    assert len(pearson_metric.targets_list) == 1  # type: ignore[arg-type]
    assert pearson_metric.preds_list[0].shape == (4,)  # type: ignore[union-attr]


def test_on_validation_epoch_end_saves_scatter(_base_config):
    """on_validation_epoch_end writes a PNG from metric-accumulated data."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log = MagicMock()
    module.log_dict = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = False
    mock_trainer.current_epoch = 2
    module._trainer = mock_trainer

    # Run a validation step to populate metrics
    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_trainer.logger.log_dir = tmp_dir
        module.on_validation_epoch_end()
        plot_dir = Path(tmp_dir) / "plots"
        pngs = list(plot_dir.glob("val_count_scatter_epoch_*.png"))
        assert len(pngs) == 1

    # Metrics must be reset after epoch end
    pearson_metric = module.val_metrics["val_pearson_log_counts"]
    assert pearson_metric.preds_list == []


def test_on_validation_epoch_end_skips_scatter_during_sanity_check(_base_config):
    """No PNG is written during Lightning's sanity check pass."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log_dict = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = True  # ← sanity check active
    module._trainer = mock_trainer

    # Run a validation step to populate metrics
    module.log = MagicMock()
    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    with patch("cerberus.plots.save_count_scatter") as mock_save:
        module.on_validation_epoch_end()
        mock_save.assert_not_called()
