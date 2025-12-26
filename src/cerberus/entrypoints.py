import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Optional

from .datamodule import CerberusDataModule
from .config import TrainConfig, ModelConfig, DataConfig
from .module import CerberusModule

def instantiate(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> "CerberusModule":
    """
    Factory function to instantiate a CerberusModule from configurations.
    
    This function bridges the gap between static configurations and the runtime model.
    It calculates standard model arguments (channels, length) from the configs and
    instantiates the user's model class.
    
    Args:
        model_config: Model architecture configuration. Must contain 'model_cls', 'loss_cls', 'metrics_cls'.
        data_config: Data inputs/outputs configuration.
        train_config: Training hyperparameters.
        
    Returns:
        Initialized CerberusModule ready for training.
    """
    # derived arguments
    in_channels = len(model_config["input_channels"])
    out_channels = len(model_config["output_channels"])
    input_len = data_config["input_len"]
    
    # Instantiate user model
    model_cls = model_config["model_cls"]
    model = model_cls(
        input_channels=in_channels,
        output_channels=out_channels,
        input_len=input_len
    )
    
    # Instantiate criterion and metrics
    loss_cls = model_config["loss_cls"]
    loss_args = model_config["loss_args"]
    criterion = loss_cls(**loss_args)
    
    metrics_cls = model_config["metrics_cls"]
    metrics_args = model_config["metrics_args"]
    metrics = metrics_cls(**metrics_args)
        
    return CerberusModule(
        model=model,
        train_config=train_config,
        criterion=criterion,
        metrics=metrics
    )

def train(
    module: pl.LightningModule,
    datamodule: CerberusDataModule,
    train_config: TrainConfig,
    callbacks: Optional[list[pl.Callback]] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Train a PyTorch Lightning model using Cerberus infrastructure.
    
    Args:
        module: A PyTorch LightningModule (e.g. CerberusModule) to train.
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
    default_callbacks = []

    # Only add LearningRateMonitor if logger is not disabled
    if trainer_kwargs.get("logger", True):
        default_callbacks.append(LearningRateMonitor(logging_interval="step"))

    default_callbacks.extend([
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
    ])
    
    if callbacks:
        default_callbacks.extend(callbacks)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=default_callbacks,
        **trainer_kwargs
    )
    
    # Train
    # Setup runtime parameters (batch_size, num_workers) from config
    datamodule.setup(
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        in_memory=train_config["in_memory"]
    )
    
    trainer.fit(module, datamodule=datamodule)
    
    return trainer
