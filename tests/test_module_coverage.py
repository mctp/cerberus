"""Coverage tests for cerberus.module -- untested code paths."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from cerberus.config import ModelConfig, TrainConfig
from cerberus.module import (
    CerberusModule,
    configure_callbacks,
    instantiate_metrics_and_loss,
)

# ---------------------------------------------------------------------------
# configure_callbacks
# ---------------------------------------------------------------------------


class TestConfigureCallbacks:
    def _train_config(self) -> TrainConfig:
        return TrainConfig.model_construct(
            patience=5,
            batch_size=32,
            max_epochs=100,
            learning_rate=1e-3,
            weight_decay=0.0,
            optimizer="adam",
            filter_bias_and_bn=True,
            scheduler_type="default",
            scheduler_args={},
            reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8,
            gradient_clip_val=None,
        )

    def test_default_with_checkpointing_and_logger(self):
        tc = self._train_config()
        cbs = configure_callbacks(tc, enable_checkpointing=True, use_logger=True)
        types = {type(c) for c in cbs}
        assert ModelCheckpoint in types
        assert EarlyStopping in types
        assert LearningRateMonitor in types

    def test_no_checkpointing(self):
        tc = self._train_config()
        cbs = configure_callbacks(tc, enable_checkpointing=False, use_logger=True)
        types = {type(c) for c in cbs}
        assert ModelCheckpoint not in types
        assert EarlyStopping in types
        assert LearningRateMonitor in types

    def test_no_logger(self):
        tc = self._train_config()
        cbs = configure_callbacks(tc, enable_checkpointing=True, use_logger=False)
        types = {type(c) for c in cbs}
        assert LearningRateMonitor not in types
        assert ModelCheckpoint in types
        assert EarlyStopping in types

    def test_no_checkpointing_no_logger(self):
        tc = self._train_config()
        cbs = configure_callbacks(tc, enable_checkpointing=False, use_logger=False)
        types = {type(c) for c in cbs}
        assert ModelCheckpoint not in types
        assert LearningRateMonitor not in types
        assert EarlyStopping in types

    def test_existing_callbacks_not_duplicated(self):
        tc = self._train_config()
        existing_es = EarlyStopping(monitor="val_loss", patience=10)
        cbs = configure_callbacks(tc, existing_callbacks=[existing_es])
        es_count = sum(1 for c in cbs if isinstance(c, EarlyStopping))
        assert es_count == 1
        # The existing one should be preserved
        assert existing_es in cbs


# ---------------------------------------------------------------------------
# instantiate_metrics_and_loss
# ---------------------------------------------------------------------------


class TestInstantiateMetricsAndLoss:
    def _model_config(self) -> ModelConfig:
        return ModelConfig.model_construct(
            name="test",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=0.0,
        )

    def test_without_device(self):
        mc = self._model_config()
        metrics, criterion = instantiate_metrics_and_loss(mc)
        assert metrics is not None
        assert criterion is not None

    def test_with_cpu_device(self):
        mc = self._model_config()
        metrics, criterion = instantiate_metrics_and_loss(
            mc, device=torch.device("cpu")
        )
        assert metrics is not None
        assert criterion is not None


# ---------------------------------------------------------------------------
# CerberusModule.lr_scheduler_step
# ---------------------------------------------------------------------------


class TestLrSchedulerStep:
    def _make_module(self):
        """Create a minimal CerberusModule for testing."""
        from cerberus.loss import MSEMultinomialLoss
        from cerberus.metrics import DefaultMetricCollection

        model = nn.Linear(10, 10)
        criterion = MSEMultinomialLoss()
        metrics = DefaultMetricCollection()
        module = CerberusModule(model=model, criterion=criterion, metrics=metrics)
        return module

    def test_with_step_update(self):
        """Scheduler that has step_update should call both step() and step_update()."""
        module = self._make_module()

        scheduler = MagicMock()
        scheduler.step_update = MagicMock()
        module.lr_scheduler_step(scheduler, optimizer_idx=0, metric=0.5)
        scheduler.step.assert_called_once()
        scheduler.step_update.assert_called_once()

    def test_without_step_update_with_metric(self):
        """Scheduler without step_update, metric provided."""
        module = self._make_module()

        scheduler = MagicMock(spec=["step"])  # no step_update
        module.lr_scheduler_step(scheduler, optimizer_idx=0, metric=0.5)
        scheduler.step.assert_called_once_with(0.5)

    def test_without_step_update_no_metric(self):
        """Scheduler without step_update, metric is None."""
        module = self._make_module()

        scheduler = MagicMock(spec=["step"])
        module.lr_scheduler_step(scheduler, optimizer_idx=0, metric=None)
        scheduler.step.assert_called_once_with()
