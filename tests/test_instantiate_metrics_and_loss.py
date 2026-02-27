import pytest
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError
from cerberus.module import instantiate_metrics_and_loss


def _make_model_config(**overrides):
    """Create a minimal model config for testing."""
    config = {
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
    }
    config.update(overrides)
    return config


class TestInstantiateMetricsAndLoss:
    def test_basic_instantiation(self):
        """Should return (metrics, criterion) tuple."""
        config = _make_model_config()
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MeanSquaredError)
        assert isinstance(criterion, nn.MSELoss)

    def test_with_args(self):
        """Should pass args to constructors."""
        config = _make_model_config(
            loss_cls="torch.nn.MSELoss",
            loss_args={"reduction": "sum"},
        )
        _, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(criterion, nn.MSELoss)
        assert criterion.reduction == "sum"

    def test_bpnet_metrics(self):
        """Should work with BPNet metric collection."""
        config = _make_model_config(
            metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
            metrics_args={"num_channels": 1, "implicit_log_targets": False},
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
        )
        metrics, criterion = instantiate_metrics_and_loss(config)
        assert isinstance(metrics, MetricCollection)

    def test_pomeranian_metrics(self):
        """Should work with Pomeranian metric collection."""
        config = _make_model_config(
            metrics_cls="cerberus.models.pomeranian.PomeranianMetricCollection",
            metrics_args={"num_channels": 1, "implicit_log_targets": False},
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
        """Missing metrics_cls should raise KeyError."""
        config = _make_model_config()
        del config["metrics_cls"]
        with pytest.raises(KeyError, match="metrics_cls"):
            instantiate_metrics_and_loss(config)

    def test_missing_loss_cls_raises(self):
        """Missing loss_cls should raise KeyError."""
        config = _make_model_config()
        del config["loss_cls"]
        with pytest.raises(KeyError, match="loss_cls"):
            instantiate_metrics_and_loss(config)

    def test_missing_metrics_args_raises(self):
        """Missing metrics_args should raise KeyError (no implicit defaults)."""
        config = _make_model_config()
        del config["metrics_args"]
        with pytest.raises(KeyError, match="metrics_args"):
            instantiate_metrics_and_loss(config)

    def test_missing_loss_args_raises(self):
        """Missing loss_args should raise KeyError (no implicit defaults)."""
        config = _make_model_config()
        del config["loss_args"]
        with pytest.raises(KeyError, match="loss_args"):
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
