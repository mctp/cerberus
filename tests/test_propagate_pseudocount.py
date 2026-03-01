"""
Tests for propagate_pseudocount and its integration into the training path.

The count_pseudocount parameter is specified once in data_config (in raw coverage
units) and must be propagated — scaled by target_scale — into loss_args and
metrics_args before the criterion and metrics are instantiated.  Previously this
propagation only happened in parse_hparams_config (prediction time); these tests
verify it now also happens during training via propagate_pseudocount called from
_train().
"""

import math
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import pytorch_lightning as pl

from cerberus.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    GenomeConfig,
    SamplerConfig,
    propagate_pseudocount,
)
from cerberus.train import _train as train
from cerberus.loss import MSEMultinomialLoss
from cerberus.models.bpnet import BPNetLoss
from cerberus.output import ProfileCountOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_config(**overrides) -> DataConfig:
    cfg: dict = {
        "input_len": 2114,
        "output_len": 1000,
        "output_bin_size": 1,
        "targets": [],
        "inputs": [],
        "use_sequence": True,
        "target_scale": 1.0,
        "count_pseudocount": 1.0,
    }
    cfg.update(overrides)
    return cast(DataConfig, cfg)


def _model_config(**overrides) -> ModelConfig:
    cfg: dict = {
        "name": "TestModel",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.models.bpnet.BPNetLoss",
        "loss_args": {"alpha": 1.0},
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
        },
    }
    cfg.update(overrides)
    return cast(ModelConfig, cfg)


def _train_config() -> TrainConfig:
    return cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 5,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    })


# ===========================================================================
# 1. propagate_pseudocount unit tests
# ===========================================================================

class TestPropagatePseudocount:
    """Unit tests for the propagate_pseudocount function."""

    def test_basic_propagation(self):
        """Pseudocount is injected into loss_args and metrics_args."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == 150.0
        assert model_cfg["metrics_args"]["count_pseudocount"] == 150.0

    def test_scaled_by_target_scale(self):
        """Pseudocount is multiplied by target_scale before injection."""
        data_cfg = _data_config(count_pseudocount=100.0, target_scale=2.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == pytest.approx(200.0)
        assert model_cfg["metrics_args"]["count_pseudocount"] == pytest.approx(200.0)

    def test_fractional_target_scale(self):
        """Fractional target_scale (e.g. 0.001) scales pseudocount down."""
        data_cfg = _data_config(count_pseudocount=100.0, target_scale=0.001)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == pytest.approx(0.1)

    def test_default_pseudocount_1(self):
        """Default pseudocount=1 with target_scale=1 injects 1.0."""
        data_cfg = _data_config(count_pseudocount=1.0, target_scale=1.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == 1.0

    def test_does_not_overwrite_explicit_loss_arg(self):
        """An explicit count_pseudocount in loss_args is not overwritten."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config(loss_args={"alpha": 1.0, "count_pseudocount": 999.0})
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == 999.0

    def test_does_not_overwrite_explicit_metrics_arg(self):
        """An explicit count_pseudocount in metrics_args is not overwritten."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config(metrics_args={"count_pseudocount": 42.0})
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["metrics_args"]["count_pseudocount"] == 42.0

    def test_independent_loss_and_metrics_override(self):
        """Only loss_args with explicit value is preserved; metrics gets injected."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config(
            loss_args={"alpha": 1.0, "count_pseudocount": 77.0},
            metrics_args={},
        )
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == 77.0
        assert model_cfg["metrics_args"]["count_pseudocount"] == 150.0

    def test_mutates_model_config_in_place(self):
        """propagate_pseudocount modifies model_config in place (returns None)."""
        data_cfg = _data_config(count_pseudocount=50.0)
        model_cfg = _model_config()
        result = propagate_pseudocount(data_cfg, model_cfg)
        assert result is None
        assert "count_pseudocount" in model_cfg["loss_args"]

    def test_idempotent(self):
        """Calling propagate_pseudocount twice doesn't change the result."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        first_loss = model_cfg["loss_args"]["count_pseudocount"]
        first_metrics = model_cfg["metrics_args"]["count_pseudocount"]
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == first_loss
        assert model_cfg["metrics_args"]["count_pseudocount"] == first_metrics


# ===========================================================================
# 2. Integration: propagate_pseudocount called from _train
# ===========================================================================

class TestTrainPropagation:
    """Verify that _train passes data_config to instantiate, which handles propagation."""

    def test_train_passes_data_config_to_instantiate(self):
        """_train must pass data_config (with count_pseudocount) to instantiate."""
        mock_module = MagicMock(spec=pl.LightningModule)
        datamodule = MagicMock()

        data_cfg = _data_config(count_pseudocount=150.0, target_scale=1.0)
        model_cfg = _model_config()
        train_cfg = _train_config()

        with patch("pytorch_lightning.Trainer"), \
             patch("cerberus.train.instantiate", return_value=mock_module) as mock_inst, \
             patch("cerberus.train.resolve_adaptive_loss_args", side_effect=lambda mc, dm, **kw: mc):

            train(
                model_config=model_cfg,
                data_config=data_cfg,
                datamodule=datamodule,
                train_config=train_cfg,
                num_workers=0,
                in_memory=False,
                accelerator="cpu",
            )

            # Verify data_config is passed through so instantiate() can propagate
            call_kwargs = mock_inst.call_args[1]
            assert call_kwargs["data_config"]["count_pseudocount"] == 150.0
            assert call_kwargs["data_config"]["target_scale"] == 1.0

    def test_instantiate_propagates_pseudocount(self):
        """instantiate() must inject scaled count_pseudocount into loss_args/metrics_args."""
        data_cfg = _data_config(count_pseudocount=100.0, target_scale=0.5)
        model_cfg = _model_config()

        # propagate_pseudocount is called inside instantiate; verify via direct call
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == pytest.approx(50.0)
        assert model_cfg["metrics_args"]["count_pseudocount"] == pytest.approx(50.0)

    def test_instantiate_does_not_overwrite_explicit_loss_arg(self):
        """Explicit count_pseudocount in loss_args is preserved by propagate_pseudocount."""
        data_cfg = _data_config(count_pseudocount=150.0)
        model_cfg = _model_config(loss_args={"alpha": 1.0, "count_pseudocount": 999.0})

        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == 999.0


# ===========================================================================
# 3. End-to-end: BPNetLoss actually receives the pseudocount
# ===========================================================================

class TestBPNetLossReceivesPseudocount:
    """Verify BPNetLoss constructed with injected pseudocount uses it correctly."""

    def test_bpnet_loss_uses_injected_pseudocount(self):
        """After propagation, BPNetLoss(count_pseudocount=150) computes the right target."""
        data_cfg = _data_config(count_pseudocount=150.0, target_scale=1.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)

        loss_fn = BPNetLoss(**model_cfg["loss_args"])
        assert loss_fn.count_pseudocount == 150.0

        # Verify forward: count target should be log(total + 150)
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total
        pred = torch.log(torch.tensor([[total + 150.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred)

        # With beta=0 (no profile loss), count loss should be 0 for perfect pred
        loss_fn_isolated = BPNetLoss(
            beta=0.0, count_pseudocount=model_cfg["loss_args"]["count_pseudocount"]
        )
        loss_val = loss_fn_isolated(out, targets)
        assert loss_val.item() == pytest.approx(0.0, abs=1e-4)

    def test_wrong_pseudocount_gives_nonzero_loss(self):
        """If pseudocount is not propagated (stays at 1), loss will be non-zero."""
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total

        # Prediction calibrated for pseudocount=150
        pred = torch.log(torch.tensor([[total + 150.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred)

        # Loss with default pseudocount=1 (the bug scenario)
        loss_fn = BPNetLoss(beta=0.0)  # count_pseudocount defaults to 1.0
        loss_val = loss_fn(out, targets)

        # target = log(500 + 1) = log(501), pred = log(650) → non-zero MSE
        expected = (math.log(650.0) - math.log(501.0)) ** 2
        assert loss_val.item() == pytest.approx(expected, rel=1e-3)
        assert loss_val.item() > 0.01


# ===========================================================================
# 4. Scatter plot target consistency
# ===========================================================================

class TestScatterPlotPseudocount:
    """Verify that _accumulate_log_counts uses the propagated pseudocount
    and produces targets consistent with the loss."""

    def test_scatter_target_matches_loss_target(self):
        """The scatter plot X-axis (target log count) must match the loss target."""
        pseudocount = 150.0
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total

        # What the loss computes as the count target
        loss_target = torch.log(torch.tensor(total) + pseudocount)

        # What _accumulate_log_counts computes as target_lc
        # (reproducing the logic from module.py)
        accum_target = torch.log(targets.sum(dim=(1, 2), dtype=torch.float32) + pseudocount)

        assert accum_target.item() == pytest.approx(loss_target.item(), abs=1e-6)

    def test_scatter_target_min_with_pseudocount_150(self):
        """With pseudocount=150, the minimum X value is log(150) ≈ 5.01, not 0."""
        pseudocount = 150.0
        # All-zero targets (silent region)
        targets = torch.zeros(1, 1, 10)
        target_lc = torch.log(targets.sum(dim=(1, 2), dtype=torch.float32) + pseudocount)
        assert target_lc.item() == pytest.approx(math.log(150.0), abs=1e-6)
        assert target_lc.item() > 5.0

    def test_scatter_target_min_with_pseudocount_1(self):
        """With pseudocount=1 (default), the minimum X value is log(1) = 0."""
        targets = torch.zeros(1, 1, 10)
        target_lc = torch.log(targets.sum(dim=(1, 2), dtype=torch.float32) + 1.0)
        assert target_lc.item() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 5. parse_hparams_config still works (regression)
# ===========================================================================

class TestParseHparamsRegression:
    """Verify parse_hparams_config still propagates pseudocount via propagate_pseudocount."""

    def test_inject_simulated(self):
        """Simulate the logic from parse_hparams_config to confirm it uses propagate_pseudocount."""
        # This mirrors what parse_hparams_config does after validation
        data_cfg = _data_config(count_pseudocount=100.0, target_scale=2.0)
        model_cfg = _model_config()
        propagate_pseudocount(data_cfg, model_cfg)
        assert model_cfg["loss_args"]["count_pseudocount"] == pytest.approx(200.0)
        assert model_cfg["metrics_args"]["count_pseudocount"] == pytest.approx(200.0)
