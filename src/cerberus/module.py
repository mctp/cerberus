import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from torchmetrics import MetricCollection

from cerberus.config import (
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.loss import CerberusLoss
from cerberus.plots import save_count_scatter
from cerberus.utils import import_class

logger = logging.getLogger(__name__)


class CerberusModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Sequence-to-Function models.

    Assumes CerberusDataset structure:
    - inputs: (Batch, Channels, Length). First 4 channels are one-hot sequence.
    - targets: (Batch, Target_Channels, Length).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: CerberusLoss,
        metrics: MetricCollection,
        train_config: TrainConfig | None = None,
        # Optional logging configuration
        genome_config: GenomeConfig | None = None,
        data_config: DataConfig | None = None,
        sampler_config: SamplerConfig | None = None,
        model_config: ModelConfig | None = None,
    ):
        """
        Args:
            model: The main model to train.
            train_config: Training configuration (TrainConfig).
            criterion: Loss function. Required.
            metrics: MetricCollection for evaluation. Required.
            genome_config: Genome configuration for logging.
            data_config: Data configuration for logging.
            sampler_config: Sampler configuration for logging.
            model_config: Model configuration for logging.
        """
        super().__init__()

        # Save serialized configurations for checkpoint reproducibility
        self.save_hyperparameters(
            {
                "train_config": train_config.model_dump(mode="json")
                if train_config is not None
                else None,
                "genome_config": genome_config.model_dump(mode="json")
                if genome_config is not None
                else None,
                "data_config": data_config.model_dump(mode="json")
                if data_config is not None
                else None,
                "sampler_config": sampler_config.model_dump(mode="json")
                if sampler_config is not None
                else None,
                "model_config": model_config.model_dump(mode="json")
                if model_config is not None
                else None,
            }
        )

        self.model = model
        self.train_config = train_config

        self.criterion = criterion

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        logger.info(
            f"Initialized CerberusModule with model: {model.__class__.__name__}"
        )

    def configure_optimizers(self):  # type: ignore[override]
        if self.train_config is None:
            raise RuntimeError("Cannot configure optimizers: train_config is missing.")

        # Create optimizer using timm
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.train_config.optimizer,
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            filter_bias_and_bn=self.train_config.filter_bias_and_bn,
            eps=self.train_config.adam_eps,
        )

        optim_conf = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        scheduler_type = self.train_config.scheduler_type
        scheduler_args = self.train_config.scheduler_args

        if scheduler_type != "default":
            # Create scheduler using timm
            scheduler, _ = create_scheduler_v2(
                optimizer, sched=scheduler_type, **scheduler_args
            )

            optim_conf["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return optim_conf

    def lr_scheduler_step(
        self, scheduler: Any, optimizer_idx: int, metric: float | None = None
    ) -> None:  # type: ignore[override]
        if hasattr(scheduler, "step_update"):
            scheduler.step(epoch=self.current_epoch, metric=metric)
            scheduler.step_update(num_updates=self.global_step)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]

        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss and log named components
        batch_context = {
            k: v for k, v in batch.items() if k not in ("inputs", "targets")
        }
        batch_size = inputs.shape[0]
        sync_dist = prefix == "val_"

        loss = self.criterion(outputs, targets, **batch_context)
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        components = self.criterion.loss_components(outputs, targets, **batch_context)
        for name, value in components.items():
            self.log(
                f"{prefix}{name}", value, batch_size=batch_size, sync_dist=sync_dist
            )

        # Metrics
        metric_collection = (
            self.train_metrics if prefix == "train_" else self.val_metrics
        )

        if isinstance(outputs, tuple | list):
            outputs_detached = tuple(o.detach() for o in outputs)
        else:
            outputs_detached = outputs.detach()

        metric_collection.update(outputs_detached, targets.detach())

        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train_")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val_")

    def on_train_epoch_end(self):
        # Log aggregated metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # Extract scatter plot data from the LogCountsPearsonCorrCoef metric
        # *before* compute()/reset() clears the accumulated state.
        scatter_preds = None
        scatter_targets = None
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            pearson_key = "pearson_log_counts"
            if pearson_key in self.val_metrics:
                pearson_metric = self.val_metrics[pearson_key]
                preds_list = pearson_metric.preds_list  # type: ignore[union-attr]
                targets_list = pearson_metric.targets_list  # type: ignore[union-attr]
                if isinstance(preds_list, torch.Tensor):
                    scatter_preds = preds_list.cpu().float().numpy()
                    scatter_targets = targets_list.cpu().float().numpy()  # type: ignore[union-attr]
                elif preds_list:
                    scatter_preds = torch.cat(preds_list).cpu().float().numpy()  # type: ignore[arg-type]
                    scatter_targets = torch.cat(targets_list).cpu().float().numpy()  # type: ignore[arg-type]

        # Log aggregated metrics, then reset
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

        # Write scatter plot (after metric bookkeeping)
        if scatter_preds is not None and scatter_targets is not None:
            trainer_log_dir = getattr(self.trainer.logger, "log_dir", None)
            save_dir = trainer_log_dir or self.trainer.default_root_dir or "."
            save_count_scatter(
                scatter_preds, scatter_targets, save_dir, self.current_epoch
            )


def configure_callbacks(
    train_config: TrainConfig,
    existing_callbacks: list[pl.Callback] | None = None,
    enable_checkpointing: bool = True,
    use_logger: bool = True,
) -> list[pl.Callback]:
    """
    Helper to configure default callbacks (LearningRateMonitor, ModelCheckpoint, EarlyStopping).

    Args:
        train_config: Training configuration.
        existing_callbacks: List of user-provided callbacks.
        enable_checkpointing: Whether to enable checkpointing.
        use_logger: Whether to enable logging.

    Returns:
        List of configured callbacks.
    """
    current_callbacks = list(existing_callbacks) if existing_callbacks else []
    existing_types = {type(c) for c in current_callbacks}

    def add_if_missing(callback_cls, **kwargs):
        if callback_cls not in existing_types:
            current_callbacks.append(callback_cls(**kwargs))

    # 1. LearningRateMonitor
    if use_logger:
        add_if_missing(LearningRateMonitor, logging_interval="step")

    # 2. ModelCheckpoint
    if enable_checkpointing:
        add_if_missing(
            ModelCheckpoint,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        )

    # 3. EarlyStopping
    add_if_missing(
        EarlyStopping,
        monitor="val_loss",
        patience=train_config.patience,
        mode="min",
    )

    return current_callbacks


def instantiate_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    compile: bool = False,
) -> torch.nn.Module:
    """
    Instantiates just the user model (backbone) from configurations.
    """
    # derived arguments
    input_len = data_config.input_len
    output_len = data_config.output_len
    output_bin_size = data_config.output_bin_size

    # Instantiate user model
    model_cls_name = model_config.model_cls
    model_cls = import_class(model_cls_name)
    model_args = model_config.model_args

    logger.debug(f"Instantiating model {model_cls_name} with args: {model_args}")

    model = model_cls(
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        **model_args,
    )

    if compile:
        model = torch.compile(model)

    return model  # type: ignore


def instantiate(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig | None = None,
    compile: bool = False,
    genome_config: GenomeConfig | None = None,
    sampler_config: SamplerConfig | None = None,
) -> "CerberusModule":
    """
    Factory function to instantiate a CerberusModule from configurations.

    This function bridges the gap between static configurations and the runtime model.
    It extracts standard model arguments (input_len, output_len, output_bin_size) from
    the data configuration and instantiates the user's model class.

    Args:
        model_config: Model architecture configuration. Must contain model_cls, loss_cls, metrics_cls.
        data_config: Data inputs/outputs configuration.
        train_config: Training hyperparameters.
        compile: Whether to compile the model using torch.compile (default: False).
        genome_config: Genome configuration for logging.
        sampler_config: Sampler configuration for logging.

    Returns:
        Initialized CerberusModule ready for training.
    """
    model = instantiate_model(model_config, data_config, compile)

    # Instantiate criterion and metrics (injects count_pseudocount at construction time)
    metrics, criterion = instantiate_metrics_and_loss(model_config)

    logger.info(
        f"Instantiated CerberusModule (Model: {model_config.name}, Loss: {model_config.loss_cls})"
    )

    return CerberusModule(
        model=model,
        train_config=train_config,
        criterion=criterion,
        metrics=metrics,
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
    )


def instantiate_metrics_and_loss(
    model_config: ModelConfig, device: torch.device | None = None
) -> tuple[MetricCollection, CerberusLoss]:
    """
    Instantiates metrics and loss functions from model configuration.

    Injects ``count_pseudocount`` from ``model_config`` into both the loss and
    metrics constructor arguments, and sets ``log_counts_include_pseudocount``
    on the metrics based on whether the loss class uses the pseudocount.

    Args:
        model_config: Model configuration (Pydantic ModelConfig).
        device: Optional device to move metrics and loss to.

    Returns:
        tuple: (metrics, criterion)
    """
    loss_cls = import_class(model_config.loss_cls)
    loss_args = {
        **model_config.loss_args,
        "count_pseudocount": model_config.count_pseudocount,
    }
    criterion = loss_cls(**loss_args)

    metrics_cls = import_class(model_config.metrics_cls)
    metrics_args = {
        **model_config.metrics_args,
        "count_pseudocount": model_config.count_pseudocount,
        "log_counts_include_pseudocount": loss_cls.uses_count_pseudocount,
    }
    metrics = metrics_cls(**metrics_args)

    if device is not None:
        metrics = metrics.to(device)
        criterion = criterion.to(device)

    return metrics, criterion
