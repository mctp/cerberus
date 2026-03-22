from typing import Any

import pytest
import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from cerberus.config import ModelConfig
from cerberus.module import instantiate_metrics_and_loss


def _make_model_config(**overrides: Any) -> ModelConfig:
    """Create a minimal model config for testing.

    Uses Cerberus-native loss/metrics classes so that count_pseudocount
    injection (performed by instantiate_metrics_and_loss) works correctly.
    """
    kwargs: dict[str, Any] = {
        "name": "test_model",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.loss.MSEMultinomialLoss",
        "loss_args": {},
        "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "output_type": "signal",
        },
        "pretrained": [],
        "count_pseudocount": 0.0,
    }
    kwargs.update(overrides)
    return ModelConfig.model_construct(**kwargs)


class TestInstantiateMetricsAndLoss:
    def test_basic_instantiation(self):
        """Should return (metrics, criterion) tuple."""
        config = _make_model_config()
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MetricCollection)
        assert isinstance(criterion, nn.Module)

    def test_with_args(self):
        """Should pass args to constructors."""
        config = _make_model_config(
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        _, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(criterion, nn.Module)

    def test_bpnet_metrics(self):
        """Should work with BPNet metric collection."""
        config = _make_model_config(
            metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
            metrics_args={"log1p_targets": False},
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MetricCollection)

    def test_pomeranian_metrics(self):
        """Should work with Pomeranian metric collection."""
        config = _make_model_config(
            metrics_cls="cerberus.models.pomeranian.PomeranianMetricCollection",
            metrics_args={"log1p_targets": False},
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MetricCollection)

    def test_device_placement(self):
        """When device is specified, metrics and loss should be moved to it."""
        config = _make_model_config()
        device = torch.device("cpu")
        metrics, criterion = instantiate_metrics_and_loss(config, device=device)
        # On CPU, just verify no error
        assert metrics is not None
        assert criterion is not None

    def test_no_device(self):
        """When device=None, should not attempt .to()."""
        config = _make_model_config()
        metrics, criterion = instantiate_metrics_and_loss(config, device=None)
        assert metrics is not None
        assert criterion is not None

    def test_missing_metrics_cls_raises(self):
        """Missing metrics_cls should raise AttributeError."""
        config = ModelConfig.model_construct(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
            # metrics_cls intentionally omitted
        )
        with pytest.raises(AttributeError):
            instantiate_metrics_and_loss(config)

    def test_missing_loss_cls_raises(self):
        """Missing loss_cls should raise AttributeError."""
        config = ModelConfig.model_construct(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_args={},
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
            # loss_cls intentionally omitted
        )
        with pytest.raises(AttributeError):
            instantiate_metrics_and_loss(config)

    def test_missing_metrics_args_raises(self):
        """Missing metrics_args should raise AttributeError."""
        config = ModelConfig.model_construct(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
            # metrics_args intentionally omitted
        )
        with pytest.raises(AttributeError):
            instantiate_metrics_and_loss(config)

    def test_missing_loss_args_raises(self):
        """Missing loss_args should raise AttributeError."""
        config = ModelConfig.model_construct(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
            # loss_args intentionally omitted
        )
        with pytest.raises(AttributeError):
            instantiate_metrics_and_loss(config)

    def test_invalid_class_raises(self):
        """Invalid class path should raise an import error."""
        config = _make_model_config(metrics_cls="nonexistent.module.Class")
        with pytest.raises((ModuleNotFoundError, ImportError)):
            instantiate_metrics_and_loss(config)

    def test_return_types(self):
        """Should return nn.Module instances."""
        config = _make_model_config()
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, nn.Module)
        assert isinstance(criterion, nn.Module)
