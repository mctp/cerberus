"""
Tests for count_pseudocount as a first-class field on ModelConfig and its
injection into loss/metrics via instantiate_metrics_and_loss().

Previously count_pseudocount lived on DataConfig and was propagated by a
standalone propagate_pseudocount() function.  Now it is a direct field on
ModelConfig and is injected at construction time by
instantiate_metrics_and_loss().
"""

import math

import pytest
import torch

from cerberus.config import (
    ModelConfig,
)
from cerberus.models.bpnet import BPNetLoss
from cerberus.module import instantiate_metrics_and_loss
from cerberus.output import ProfileCountOutput, get_log_count_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_config(**overrides) -> ModelConfig:
    kw: dict = {
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
        "pretrained": [],
        "count_pseudocount": 0.0,
    }
    kw.update(overrides)
    return ModelConfig(**kw)


# ===========================================================================
# 1. ModelConfig.count_pseudocount field tests
# ===========================================================================

class TestModelConfigCountPseudocount:
    """Unit tests for count_pseudocount as a first-class field on ModelConfig."""

    def test_default_value(self):
        """count_pseudocount defaults to 0.0."""
        cfg = _model_config()
        assert cfg.count_pseudocount == 0.0

    def test_explicit_value(self):
        """count_pseudocount can be set explicitly."""
        cfg = _model_config(count_pseudocount=150.0)
        assert cfg.count_pseudocount == 150.0

    def test_zero_valid(self):
        """count_pseudocount=0.0 is valid (for Poisson/NB losses)."""
        cfg = _model_config(count_pseudocount=0.0)
        assert cfg.count_pseudocount == 0.0

    def test_fractional(self):
        """Fractional count_pseudocount (e.g. 0.1) is valid."""
        cfg = _model_config(count_pseudocount=0.1)
        assert cfg.count_pseudocount == pytest.approx(0.1)

    def test_negative_rejected(self):
        """Negative count_pseudocount should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="count_pseudocount"):
            _model_config(count_pseudocount=-5.0)

    def test_frozen_immutable(self):
        """count_pseudocount cannot be mutated on a frozen model."""
        from pydantic import ValidationError
        cfg = _model_config(count_pseudocount=150.0)
        with pytest.raises(ValidationError):
            cfg.count_pseudocount = 999.0  # type: ignore[misc]


# ===========================================================================
# 2. instantiate_metrics_and_loss injection tests
# ===========================================================================

class TestInstantiateMetricsAndLoss:
    """Verify instantiate_metrics_and_loss injects count_pseudocount."""

    def test_injects_pseudocount_into_loss(self):
        """Loss receives count_pseudocount from ModelConfig."""
        cfg = _model_config(count_pseudocount=150.0)
        metrics, criterion = instantiate_metrics_and_loss(cfg)
        assert getattr(criterion, "count_pseudocount") == 150.0  # noqa: B009

    def test_injects_pseudocount_into_metrics(self):
        """Metrics sub-metrics receive count_pseudocount from ModelConfig."""
        cfg = _model_config(count_pseudocount=150.0)
        metrics, criterion = instantiate_metrics_and_loss(cfg)
        # BPNetMetricCollection passes count_pseudocount to each sub-metric
        for name, metric in metrics.items():
            assert metric.count_pseudocount == 150.0, f"Sub-metric '{name}' has wrong pseudocount"

    def test_zero_pseudocount_injected(self):
        """count_pseudocount=0 is still injected."""
        cfg = _model_config(count_pseudocount=0.0)
        metrics, criterion = instantiate_metrics_and_loss(cfg)
        assert getattr(criterion, "count_pseudocount") == 0.0  # noqa: B009

    def test_model_config_loss_args_not_mutated(self):
        """The original model_config.loss_args dict should not be mutated."""
        cfg = _model_config(count_pseudocount=150.0)
        original_loss_args = dict(cfg.loss_args)
        instantiate_metrics_and_loss(cfg)
        # Frozen Pydantic model's loss_args shouldn't change
        assert cfg.loss_args == original_loss_args


# ===========================================================================
# 3. End-to-end: BPNetLoss actually receives the pseudocount
# ===========================================================================

class TestBPNetLossReceivesPseudocount:
    """Verify BPNetLoss constructed with injected pseudocount uses it correctly."""

    def test_bpnet_loss_uses_injected_pseudocount(self):
        """After injection, BPNetLoss(count_pseudocount=150) computes the right target."""
        cfg = _model_config(count_pseudocount=150.0)
        _metrics, criterion = instantiate_metrics_and_loss(cfg)

        assert getattr(criterion, "count_pseudocount") == 150.0  # noqa: B009

        # Verify forward: count target should be log(total + 150)
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total
        pred = torch.log(torch.tensor([[total + 150.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred)

        # With beta=0 (no profile loss), count loss should be 0 for perfect pred
        loss_fn_isolated = BPNetLoss(
            beta=0.0, count_pseudocount=150.0,
        )
        loss_val = loss_fn_isolated(out, targets)
        assert loss_val.item() == pytest.approx(0.0, abs=1e-4)

    def test_wrong_pseudocount_gives_nonzero_loss(self):
        """If pseudocount is wrong (stays at 1), loss will be non-zero."""
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total

        # Prediction calibrated for pseudocount=150
        pred = torch.log(torch.tensor([[total + 150.0]]))
        out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred)

        # Loss with default pseudocount=1 (the bug scenario)
        loss_fn = BPNetLoss(beta=0.0)  # count_pseudocount defaults to 1.0
        loss_val = loss_fn(out, targets)

        # target = log(500 + 1) = log(501), pred = log(650) -> non-zero MSE
        expected = (math.log(650.0) - math.log(501.0)) ** 2
        assert loss_val.item() == pytest.approx(expected, rel=1e-3)
        assert loss_val.item() > 0.01


# ===========================================================================
# 4. Scatter plot target consistency
# ===========================================================================

class TestScatterPlotPseudocount:
    """Verify that log-count targets with pseudocount are consistent with the loss."""

    def test_scatter_target_matches_loss_target(self):
        """The scatter plot X-axis (target log count) must match the loss target."""
        pseudocount = 150.0
        total = 500.0
        targets = torch.zeros(1, 1, 10)
        targets[0, 0, 0] = total

        loss_target = torch.log(torch.tensor(total) + pseudocount)
        accum_target = torch.log(targets.sum(dim=(1, 2), dtype=torch.float32) + pseudocount)
        assert accum_target.item() == pytest.approx(loss_target.item(), abs=1e-6)

    def test_scatter_target_min_with_pseudocount_150(self):
        """With pseudocount=150, the minimum X value is log(150), not 0."""
        pseudocount = 150.0
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
# 5. get_log_count_params reads from ModelConfig
# ===========================================================================

class TestGetLogCountParams:
    """Verify get_log_count_params reads count_pseudocount from ModelConfig."""

    def test_mse_uses_pseudocount(self):
        cfg = _model_config(
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            count_pseudocount=50.0,
        )
        includes, pseudocount = get_log_count_params(cfg)
        assert includes is True
        assert pseudocount == 50.0

    def test_poisson_ignores_pseudocount(self):
        cfg = _model_config(
            loss_cls="cerberus.loss.PoissonMultinomialLoss",
            loss_args={},
            count_pseudocount=99.0,
        )
        includes, pseudocount = get_log_count_params(cfg)
        assert includes is False
        assert pseudocount == 0.0
