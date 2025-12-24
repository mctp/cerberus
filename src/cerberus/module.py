import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, MeanSquaredError, MetricCollection
from typing import Any, Optional, Union, Callable

class CerberusModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Sequence-to-Function models.
    
    Compatible with ASAP models that output predicted counts/rates directly.
    
    Assumes CerberusDataset structure:
    - inputs: (Batch, Channels, Length). First 4 channels are one-hot sequence.
    - targets: (Batch, Target_Channels, Length).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_config: dict,
        bias_model: Optional[nn.Module] = None,
        criterion: Optional[Union[nn.Module, Callable]] = None,
    ):
        """
        Args:
            model: The main model to train.
            train_config: Configuration dictionary (TrainConfig).
            bias_model: Optional bias model. Output is subtracted from main model logits.
            criterion: Loss function. Defaults to PoissonNLLLoss(log_input=False).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "bias_model", "criterion"])
        self.model = model
        self.bias_model = bias_model
        
        self.train_config = train_config
        
        # Loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            # Default to PoissonNLLLoss (log_input=False) as used in ASAP
            self.criterion = nn.PoissonNLLLoss(log_input=False, full=True)

        # Metrics
        # We assume single channel for now or mean over channels
        metrics = MetricCollection({
            "pearson": PearsonCorrCoef(),
            "mse": MeanSquaredError(),
        })
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(self):
        opt_name = self.train_config["optimizer"].lower()
        lr = self.train_config["learning_rate"]
        weight_decay = self.train_config["weight_decay"]
        
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
            
        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }

    def _shared_step(self, batch, batch_idx, prefix):
        inputs = batch["inputs"]
        targets = batch["targets"]
        
        # Forward pass
        outputs = self.model(inputs)

        # Bias correction
        if self.bias_model:
            # See docs/internal/bias_factorized_models.md
            pass

        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Logging
        self.log(f"{prefix}loss", loss, prog_bar=True)
             
        # Metrics
        metric_collection = self.train_metrics if prefix == "train_" else self.val_metrics
        
        # Flatten for correlation metrics
        outputs_flat = outputs.detach().flatten()
        targets_flat = targets.detach().flatten()
        
        metric_collection.update(outputs_flat, targets_flat)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train_")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val_")
        
    def on_validation_epoch_end(self):
        # Log aggregated metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
