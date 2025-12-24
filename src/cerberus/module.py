import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from typing import Optional, Union, Callable
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2

from cerberus.config import TrainConfig


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
        criterion: Union[nn.Module, Callable],
        metrics: MetricCollection,
    ):
        """
        Args:
            model: The main model to train.
            train_config: Configuration dictionary (TrainConfig).
            criterion: Loss function. Required.
            metrics: MetricCollection for evaluation. Required.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion", "metrics"])
        self.model = model
        self.train_config = train_config
        
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

    def _shared_step(self, batch, batch_idx, prefix):
        inputs = batch["inputs"]
        targets = batch["targets"]
        
        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Logging
        self.log(f"{prefix}loss", loss, prog_bar=True)
             
        # Metrics
        metric_collection = self.train_metrics if prefix == "train_" else self.val_metrics
        metric_collection.update(outputs.detach(), targets.detach())
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train_")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val_")
    
    def on_train_epoch_end(self):
        # Log aggregated metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        # Log aggregated metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
