import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from typing import Callable, Any, cast
from pathlib import Path
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2

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

    def configure_optimizers(self):
        if self.train_config is None:
            raise RuntimeError("Cannot configure optimizers: train_config is missing.")

        # Create optimizer using timm
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.train_config["optimizer"],
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"],
            filter_bias_and_bn=self.train_config["filter_bias_and_bn"]
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
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
        
    def on_validation_epoch_end(self):
        # Log aggregated metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()


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
    
    model = model_cls(
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        **model_args
    )

    if compile:
        model = cast(torch.nn.Module, torch.compile(model))
        
    return model


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
    loss_cls_name = model_config["loss_cls"]
    loss_cls = import_class(loss_cls_name)
    loss_args = model_config["loss_args"]
    criterion = loss_cls(**loss_args)

    metrics_cls_name = model_config["metrics_cls"]
    metrics_cls = import_class(metrics_cls_name)
    metrics_args = model_config["metrics_args"]
    metrics = metrics_cls(**metrics_args)

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
