"""
Regression tests for log_counts_include_pseudocount propagation through MetricCollections.

Bug: PomeranianMetricCollection and BPNetMetricCollection did not accept
log_counts_include_pseudocount, causing TypeError when instantiated via
instantiate_metrics_and_loss() with MSE-style losses (which set
uses_count_pseudocount=True in config.propagate_pseudocount).

These tests ensure:
1. All MetricCollections accept log_counts_include_pseudocount.
2. The flag propagates to inner LogCounts* sub-metrics.
3. instantiate_metrics_and_loss() works end-to-end with the flag.
4. The flag actually affects metric computation for multi-channel outputs.
"""
import pytest
import torch
from typing import Any, cast
from torchmetrics import MetricCollection

from cerberus.metrics import (
    DefaultMetricCollection,
    LogCountsMeanSquaredError,
    LogCountsPearsonCorrCoef,
)
from cerberus.models.bpnet import BPNetMetricCollection
from cerberus.models.pomeranian import PomeranianMetricCollection
from cerberus.config import ModelConfig
from cerberus.module import instantiate_metrics_and_loss
from cerberus.output import ProfileCountOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_METRIC_COLLECTIONS = [
    DefaultMetricCollection,
    BPNetMetricCollection,
    PomeranianMetricCollection,
]

LOG_COUNTS_METRIC_NAMES = {"mse_log_counts", "pearson_log_counts"}


def _make_model_config(**overrides: Any) -> ModelConfig:
    """Minimal model config for instantiate_metrics_and_loss tests."""
    config: dict[str, Any] = {
        "name": "test_model",
        "model_cls": "torch.nn.Linear",
        "loss_cls": "torch.nn.MSELoss",
        "loss_args": {},
        "metrics_cls": "torchmetrics.MeanSquaredError",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "output_type": "signal",
        },
        "pretrained": [],
    }
    config.update(overrides)
    return cast(ModelConfig, config)


# ===========================================================================
# 1. Construction: all MetricCollections accept log_counts_include_pseudocount
# ===========================================================================

class TestMetricCollectionAcceptsLogCountsIncludePseudocount:
    """Every MetricCollection must accept log_counts_include_pseudocount without error."""

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_default_false(self, cls: type[Any]) -> None:
        mc = cls(log_counts_include_pseudocount=False)
        assert isinstance(mc, MetricCollection)

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_explicit_true(self, cls: type[Any]) -> None:
        mc = cls(log_counts_include_pseudocount=True)
        assert isinstance(mc, MetricCollection)

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_all_kwargs_together(self, cls: type[Any]) -> None:
        """Combination of all kwargs that propagate_pseudocount may inject."""
        mc = cls(
            log1p_targets=True,
            count_pseudocount=50.0,
            log_counts_include_pseudocount=True,
        )
        assert isinstance(mc, MetricCollection)


# ===========================================================================
# 2. Propagation: flag reaches inner LogCounts* sub-metrics
# ===========================================================================

class TestLogCountsIncludePseudocountPropagation:
    """log_counts_include_pseudocount must propagate to every LogCounts* sub-metric."""

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_false_propagated(self, cls: type[Any]) -> None:
        mc = cls(log_counts_include_pseudocount=False)
        for name, metric in mc.items():
            if name in LOG_COUNTS_METRIC_NAMES:
                assert hasattr(metric, "log_counts_include_pseudocount"), (
                    f"{cls.__name__}.{name} missing log_counts_include_pseudocount attr"
                )
                assert metric.log_counts_include_pseudocount is False

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_true_propagated(self, cls: type[Any]) -> None:
        mc = cls(log_counts_include_pseudocount=True)
        for name, metric in mc.items():
            if name in LOG_COUNTS_METRIC_NAMES:
                assert metric.log_counts_include_pseudocount is True

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_non_log_counts_metrics_unaffected(self, cls: type[Any]) -> None:
        """Metrics that don't use log_counts_include_pseudocount should not have the attr."""
        mc = cls(log_counts_include_pseudocount=True)
        for name, metric in mc.items():
            if name not in LOG_COUNTS_METRIC_NAMES:
                # These metrics don't use the flag — they may or may not have the attr,
                # but they should not be LogCounts* types.
                assert not isinstance(metric, (LogCountsMeanSquaredError, LogCountsPearsonCorrCoef))


# ===========================================================================
# 3. End-to-end: instantiate_metrics_and_loss with log_counts_include_pseudocount
# ===========================================================================

class TestInstantiateWithLogCountsIncludePseudocount:
    """instantiate_metrics_and_loss must work when metrics_args contains the flag."""

    @pytest.mark.parametrize("metrics_cls_path", [
        "cerberus.metrics.DefaultMetricCollection",
        "cerberus.models.bpnet.BPNetMetricCollection",
        "cerberus.models.pomeranian.PomeranianMetricCollection",
    ])
    @pytest.mark.parametrize("flag_value", [True, False])
    def test_instantiation_succeeds(self, metrics_cls_path: str, flag_value: bool) -> None:
        config = _make_model_config(
            metrics_cls=metrics_cls_path,
            metrics_args={
                "log1p_targets": False,
                "count_pseudocount": 42.0,
                "log_counts_include_pseudocount": flag_value,
            },
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MetricCollection)

    @pytest.mark.parametrize("metrics_cls_path", [
        "cerberus.metrics.DefaultMetricCollection",
        "cerberus.models.bpnet.BPNetMetricCollection",
        "cerberus.models.pomeranian.PomeranianMetricCollection",
    ])
    def test_flag_propagated_through_instantiate(self, metrics_cls_path: str) -> None:
        """The flag value from config reaches inner sub-metrics."""
        config = _make_model_config(
            metrics_cls=metrics_cls_path,
            metrics_args={
                "log1p_targets": False,
                "log_counts_include_pseudocount": True,
            },
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        metrics, _ = instantiate_metrics_and_loss(config)
        for name, metric in metrics.items():
            if name in LOG_COUNTS_METRIC_NAMES:
                assert metric.log_counts_include_pseudocount is True


# ===========================================================================
# 4. Functional: flag affects multi-channel log_counts computation
# ===========================================================================

class TestLogCountsIncludePseudocountFunctional:
    """Verify the flag actually changes metric behavior for multi-channel outputs."""

    def _make_multi_channel_data(self, pseudocount: float = 1.0) -> tuple[ProfileCountOutput, torch.Tensor]:
        """Create pred (ProfileCountOutput) and target (raw counts tensor) for multi-channel."""
        # 2 channels, batch=4, length=64
        B, C, L = 4, 2, 64
        logits = torch.randn(B, C, L)
        # log_counts in log(count + pseudocount) space (MSE-style)
        log_counts = torch.log(torch.rand(B, C).abs() * 100 + pseudocount)
        pred = ProfileCountOutput(logits=logits, log_counts=log_counts)
        # Target is a raw count tensor (B, C, L)
        target = torch.rand(B, C, L).abs() * 100
        return pred, target

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_flag_changes_mse_log_counts_value(self, cls: type[Any]) -> None:
        """With multi-channel outputs, True vs False should give different MSE values."""
        pseudocount = 50.0
        pred, target = self._make_multi_channel_data(pseudocount=pseudocount)

        mc_false = cls(count_pseudocount=pseudocount, log_counts_include_pseudocount=False)
        mc_true = cls(count_pseudocount=pseudocount, log_counts_include_pseudocount=True)

        mc_false.update(pred, target)
        mc_true.update(pred, target)

        result_false = mc_false.compute()
        result_true = mc_true.compute()

        # The mse_log_counts should differ because the aggregation path differs
        assert not torch.allclose(
            result_false["mse_log_counts"], result_true["mse_log_counts"]
        ), "Multi-channel mse_log_counts should differ between flag=True and flag=False"

    @pytest.mark.parametrize("cls", ALL_METRIC_COLLECTIONS)
    def test_single_channel_flag_has_no_effect(self, cls: type[Any]) -> None:
        """With single-channel outputs, the flag should not matter."""
        B, L = 4, 64
        logits = torch.randn(B, 1, L)
        log_counts = torch.randn(B, 1)
        pred = ProfileCountOutput(logits=logits, log_counts=log_counts)
        # Target is a raw count tensor (B, 1, L)
        target = torch.rand(B, 1, L).abs() * 100

        mc_false = cls(log_counts_include_pseudocount=False)
        mc_true = cls(log_counts_include_pseudocount=True)

        mc_false.update(pred, target)
        mc_true.update(pred, target)

        result_false = mc_false.compute()
        result_true = mc_true.compute()

        assert torch.allclose(
            result_false["mse_log_counts"], result_true["mse_log_counts"]
        ), "Single-channel mse_log_counts should be identical regardless of flag"
