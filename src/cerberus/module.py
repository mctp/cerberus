import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics import MetricCollection
import logging
from typing import Callable, Any
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2

from cerberus.output import compute_total_log_counts
from cerberus.plots import save_count_scatter
from cerberus.config import (
    TrainConfig,
    validate_train_config,
    GenomeConfig,
    DataConfig,
    SamplerConfig,
    ModelConfig,
    _sanitize_config,
    validate_model_config,
    validate_data_config,
    validate_genome_config,
    validate_sampler_config,
    import_class,
)

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
        criterion: nn.Module | Callable,
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
            train_config: Configuration dictionary (TrainConfig).
            criterion: Loss function. Required.
            metrics: MetricCollection for evaluation. Required.
            genome_config: Genome configuration for logging.
            data_config: Data configuration for logging.
            sampler_config: Sampler configuration for logging.
            model_config: Model configuration for logging.
        """
        super().__init__()
        
        # Save sanitized configurations (converting Path -> str)
        self.save_hyperparameters({
            "train_config": _sanitize_config(train_config),
            "genome_config": _sanitize_config(genome_config),
            "data_config": _sanitize_config(data_config),
            "sampler_config": _sanitize_config(sampler_config),
            "model_config": _sanitize_config(model_config),
        })
        
        self.model = model
        self.train_config = train_config
        
        self.criterion = criterion
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # Accumulators for the per-epoch validation scatter plot.
        # Populated in _shared_step (val only, rank 0) and cleared in on_validation_epoch_end.
        self._val_log_count_preds: list[torch.Tensor] = []
        self._val_log_count_targets: list[torch.Tensor] = []

        logger.info(f"Initialized CerberusModule with model: {model.__class__.__name__}")

    def configure_optimizers(self): # type: ignore[override]
        if self.train_config is None:
            raise RuntimeError("Cannot configure optimizers: train_config is missing.")

        # Create optimizer using timm
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.train_config["optimizer"],
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"],
            filter_bias_and_bn=self.train_config["filter_bias_and_bn"],
            eps=self.train_config["adam_eps"],
        )
        
        optim_conf = {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

        scheduler_type = self.train_config["scheduler_type"]
        scheduler_args = self.train_config["scheduler_args"]

        if scheduler_type != "default":
             # Create scheduler using timm
             scheduler, _ = create_scheduler_v2(
                 optimizer,
                 sched=scheduler_type,
                 **scheduler_args
             )
             
             optim_conf["lr_scheduler"] = {
                 "scheduler": scheduler,
                 "interval": "step",
                 "frequency": 1,
             }
        
        return optim_conf

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None): # type: ignore[override]
        if hasattr(scheduler, "step_update"):
            scheduler.step(epoch=self.current_epoch, metric=metric)
            scheduler.step_update(num_updates=self.global_step)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def _shared_step(self, batch, batch_idx, prefix):
        inputs = batch["inputs"]
        targets = batch["targets"]
        
        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Logging
        batch_size = inputs.shape[0]
        # Sync dist for validation to avoid warning and ensure correct epoch accumulation
        sync_dist = (prefix == "val_")
        self.log(f"{prefix}loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=sync_dist)
             
        # Metrics
        metric_collection = self.train_metrics if prefix == "train_" else self.val_metrics
        
        if isinstance(outputs, (tuple, list)):
            outputs_detached = tuple(o.detach() for o in outputs)
        else:
            outputs_detached = outputs.detach()
            
        metric_collection.update(outputs_detached, targets.detach())

        # Accumulate log counts for the validation scatter plot (rank 0 only)
        if prefix == "val_" and self.trainer.is_global_zero:
            self._accumulate_log_counts(outputs_detached, targets.detach())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train_")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val_")
    
    def on_train_epoch_end(self):
        # Log aggregated metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.train_metrics.reset()
        
    def _accumulate_log_counts(self, outputs: Any, targets: torch.Tensor) -> None:
        """
        Accumulate predicted and target log counts for the epoch-end scatter plot.

        Silently skips output types not supported by compute_total_log_counts
        (e.g. raw-tensor or tuple outputs from non-standard models).

        Args:
            outputs: Detached model output (ModelOutput subclass or other).
            targets: Detached target tensor, shape (B, C, L).
        """
        try:
            pred_lc = compute_total_log_counts(outputs)       # (B,)
            target_lc = torch.log1p(targets.sum(dim=(1, 2)))  # (B,)
            self._val_log_count_preds.append(pred_lc.cpu())
            self._val_log_count_targets.append(target_lc.cpu())
        except (ValueError, AttributeError, TypeError):
            pass

    def on_validation_epoch_end(self):
        # Log aggregated metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

        # Generate count scatter plot (rank 0 only; skip Lightning's 2-batch sanity check)
        if (self._val_log_count_preds
                and self.trainer.is_global_zero
                and not self.trainer.sanity_checking):
            all_preds = torch.cat(self._val_log_count_preds).numpy()
            all_targets = torch.cat(self._val_log_count_targets).numpy()
            trainer_log_dir = getattr(self.trainer.logger, "log_dir", None)
            save_dir = trainer_log_dir or self.trainer.default_root_dir or "."
            save_count_scatter(all_preds, all_targets, save_dir, self.current_epoch)
        self._val_log_count_preds = []
        self._val_log_count_targets = []


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
            filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        )

    # 3. EarlyStopping
    add_if_missing(
        EarlyStopping,
        monitor="val_loss",
        patience=train_config["patience"],
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
    model_config = validate_model_config(model_config)
    data_config = validate_data_config(data_config)

    # derived arguments
    input_len = data_config["input_len"]
    output_len = data_config["output_len"]
    output_bin_size = data_config["output_bin_size"]

    # Instantiate user model
    model_cls_name = model_config["model_cls"]
    model_cls = import_class(model_cls_name)
    model_args = model_config["model_args"]
    
    logger.debug(f"Instantiating model {model_cls_name} with args: {model_args}")
    
    model = model_cls(
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        **model_args
    )

    if compile:
        model = torch.compile(model)
        
    return model # type: ignore


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
        model_config: Model architecture configuration. Must contain 'model_cls', 'loss_cls', 'metrics_cls'.
        data_config: Data inputs/outputs configuration.
        train_config: Training hyperparameters.
        compile: Whether to compile the model using torch.compile (default: False).

    Returns:
        Initialized CerberusModule ready for training.
    """
    # Validate optional configs used here or passed down
    if train_config is not None:
        train_config = validate_train_config(train_config)
    if genome_config is not None:
        genome_config = validate_genome_config(genome_config)
    if sampler_config is not None:
        sampler_config = validate_sampler_config(sampler_config)
    
    # model_config and data_config validated in instantiate_model
    model = instantiate_model(model_config, data_config, compile)
    
    # Ensure model_config is validated version for subsequent use
    model_config = validate_model_config(model_config)

    # Instantiate criterion and metrics
    metrics, criterion = instantiate_metrics_and_loss(model_config)

    logger.info(f"Instantiated CerberusModule (Model: {model_config['name']}, Loss: {model_config['loss_cls']})")

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
    model_config: ModelConfig, 
    device: torch.device | None = None
) -> tuple[Any, Any]:
    """
    Instantiates metrics and loss functions from model configuration.
    
    Args:
        model_config: Model configuration dictionary.
        device: Optional device to move metrics and loss to.
        
    Returns:
        tuple: (metrics, criterion)
    """
    metrics_cls = import_class(model_config["metrics_cls"])
    metrics_args = model_config["metrics_args"]
    metrics = metrics_cls(**metrics_args)

    loss_cls = import_class(model_config["loss_cls"])
    loss_args = model_config["loss_args"]
    criterion = loss_cls(**loss_args)
    
    if device is not None:
        metrics = metrics.to(device)
        criterion = criterion.to(device)
        
    return metrics, criterion
