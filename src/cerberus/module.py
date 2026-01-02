import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from typing import Callable
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2

from cerberus.config import (
    TrainConfig,
    validate_train_config,
    GenomeConfig,
    DataConfig,
    SamplerConfig,
    ModelConfig,
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
        train_config: TrainConfig,
        criterion: nn.Module | Callable,
        metrics: MetricCollection,
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
        self.save_hyperparameters(ignore=["model", "criterion", "metrics"])
        self.model = model
        self.train_config = validate_train_config(train_config)
        
        self.criterion = criterion
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(self):

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
