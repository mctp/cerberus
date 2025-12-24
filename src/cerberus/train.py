import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Union, Optional

from .datamodule import CerberusDataModule
from .config import TrainConfig

def train(
    model: pl.LightningModule,
    datamodule: CerberusDataModule,
    train_config: dict | TrainConfig,
    callbacks: Optional[list[pl.Callback]] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Train a PyTorch Lightning model using Cerberus infrastructure.
    
    Args:
        model: A PyTorch LightningModule (e.g. CerberusModule) to train.
        datamodule: A pre-initialized CerberusDataModule.
        train_config: Training configuration dictionary containing keys like:
            - 'max_epochs': Maximum number of epochs to train.
            - 'patience': Patience for early stopping (based on val_loss).
        callbacks: Optional list of additional PyTorch Lightning callbacks.
            These will be added to the default set (LearningRateMonitor, ModelCheckpoint, EarlyStopping).
        **trainer_kwargs: Additional arguments passed directly to pl.Trainer.
            Common examples include:
            - accelerator: "auto", "gpu", "cpu"
            - devices: "auto", int, or list
            - default_root_dir: path for logs/checkpoints
            - precision: "16-mixed", "32", etc.
            - logger: Custom logger instance
        
    Returns:
        The fitted PyTorch Lightning Trainer object.
    """
    
    # Default Callbacks
    default_callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=train_config["patience"],
            mode="min",
        )
    ]
    
    if callbacks:
        default_callbacks.extend(callbacks)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=default_callbacks,
        **trainer_kwargs
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    return trainer
