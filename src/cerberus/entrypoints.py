import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import cast, Any
from pathlib import Path

from .datamodule import CerberusDataModule
from .config import (
    TrainConfig,
    ModelConfig,
    DataConfig,
    GenomeConfig,
    SamplerConfig,
)
from .module import CerberusModule


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
    # derived arguments
    input_len = data_config["input_len"]
    output_len = data_config["output_len"]
    output_bin_size = data_config["output_bin_size"]

    # Instantiate user model
    model_cls = model_config["model_cls"]
    model_args = model_config["model_args"]
    model = model_cls(
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        **model_args
    )

    if compile:
        model = cast(torch.nn.Module, torch.compile(model))

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
        metrics=metrics,
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
    )


def _configure_callbacks(
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


def train(
    module: pl.LightningModule,
    datamodule: CerberusDataModule,
    train_config: TrainConfig,
    callbacks: list[pl.Callback] | None = None,
    num_workers: int = 0,
    in_memory: bool = False,
    matmul_precision: str = "highest",
    precision: Any = "32-true",
    root_dir: str | Path | None = None,
    val_batch_size: int | None = None,
    **trainer_kwargs,
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
            These will be added to the default set (LearningRateMonitor, ModelCheckpoint, EarlyStopping)
            unless a callback of the same type is already provided.
        num_workers: Number of DataLoader workers (default: 0).
        in_memory: Whether to load data into memory (default: False).
        matmul_precision: Precision for float32 matrix multiplication.
            Options: 'highest', 'high', 'medium'. (default: 'highest').
            Set to 'medium' or 'high' on newer NVIDIA GPUs (Ampere+) to improve performance.
        precision: Mixed precision setting passed to pl.Trainer. 
            Options: '32-true', '16-mixed', 'bf16-mixed', etc. (default: '32-true').
        root_dir: Root directory for saving logs and checkpoints.
            If provided, it overrides 'default_root_dir' in trainer_kwargs.
        **trainer_kwargs: Additional arguments passed directly to pl.Trainer.
            Common examples include:
            - accelerator: "auto", "gpu", "cpu"
            - devices: "auto", int, or list
            - default_root_dir: path for logs/checkpoints
            - logger: Custom logger instance

    Returns:
        The fitted PyTorch Lightning Trainer object.
    """

    # Set float32 matmul precision
    torch.set_float32_matmul_precision(matmul_precision)

    # Determine logger status
    # Logic matches what we do below for creating default logger
    use_logger = trainer_kwargs.get("logger") is not False

    # Determine checkpointing status
    enable_checkpointing = trainer_kwargs.get("enable_checkpointing", True)

    # Prepare callbacks
    current_callbacks = _configure_callbacks(
        train_config,
        callbacks,
        enable_checkpointing=enable_checkpointing,
        use_logger=use_logger,
    )

    # Handle root_dir convenience argument
    if root_dir is not None:
        trainer_kwargs["default_root_dir"] = str(root_dir)

    # Explicitly use CSVLogger if none provided.
    if trainer_kwargs.get("logger") is True or trainer_kwargs.get("logger") is None:
        # Import inside function to allow patching in tests (e.g. test_entrypoint_logger.py)
        # where the logger class is mocked after module import.
        from pytorch_lightning.loggers import CSVLogger

        save_dir = trainer_kwargs.get("default_root_dir", ".")
        if save_dir is None:
            save_dir = "."
        trainer_kwargs["logger"] = CSVLogger(save_dir=str(save_dir))

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=current_callbacks,
        precision=precision,
        **trainer_kwargs,
    )

    # Setup DataModule
    # Passing runtime parameters to setup allowing for final adjustments
    datamodule.setup(
        batch_size=train_config["batch_size"],
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        in_memory=in_memory,
    )

    # Start Training
    trainer.fit(module, datamodule=datamodule)

    return trainer


def train_single(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    test_fold: int | None = None,
    val_fold: int | None = None,
    compile: bool = False,
    num_workers: int = 0,
    in_memory: bool = False,
    matmul_precision: str = "highest",
    precision: str = "32-true",
    root_dir: str | Path = ".",
    val_batch_size: int | None = None,
    **trainer_kwargs,
) -> pl.Trainer:
    """
    Train a single model for a specific fold from configurations.

    Acts as a high-level entrypoint that handles instantiation of the
    CerberusModule and CerberusDataModule before calling the low-level train().

    Args:
        genome_config: Genome configuration containing fold definitions.
        data_config: Data inputs/outputs configuration.
        sampler_config: Sampler configuration.
        model_config: Model architecture configuration.
        train_config: Training hyperparameters.
        test_fold: Fold index to use for testing (default: 0).
        val_fold: Fold index to use for validation (default: 1).
        compile: Whether to compile the model (default: False).
        num_workers: Number of DataLoader workers (default: 0).
        in_memory: Whether to load data into memory (default: False).
        matmul_precision: Precision for float32 matrix multiplication.
        precision: Mixed precision setting passed to pl.Trainer.
        root_dir: Root directory for saving logs and checkpoints.
        **trainer_kwargs: Additional arguments passed to pl.Trainer.

    Returns:
        The fitted PyTorch Lightning Trainer object.
    """
    # 1. Instantiate DataModule for this fold
    datamodule = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        test_fold=test_fold,
        val_fold=val_fold,
    )

    # 2. Instantiate new Model
    module = instantiate(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        compile=compile,
        genome_config=genome_config,
        sampler_config=sampler_config,
    )

    # 3. Train
    return train(
        module=module,
        datamodule=datamodule,
        train_config=train_config,
        num_workers=num_workers,
        in_memory=in_memory,
        matmul_precision=matmul_precision,
        precision=precision,
        root_dir=root_dir,
        val_batch_size=val_batch_size,
        **trainer_kwargs,
    )


def train_multi(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    compile: bool = False,
    num_workers: int = 0,
    in_memory: bool = False,
    matmul_precision: str = "highest",
    precision: str = "32-true",
    root_dir: str | Path = ".",
    val_batch_size: int | None = None,
    **trainer_kwargs,
) -> list[pl.Trainer]:
    """
    Train multiple models using k-fold cross-validation.

    This function iterates through all folds defined in the genome configuration,
    training a separate model for each fold where that fold is used as the test set.

    Args:
        genome_config: Genome configuration containing fold definitions.
        data_config: Data inputs/outputs configuration.
        sampler_config: Sampler configuration.
        model_config: Model architecture configuration.
        train_config: Training hyperparameters.
        compile: Whether to compile the model (default: False).
        num_workers: Number of DataLoader workers (default: 0).
        in_memory: Whether to load data into memory (default: False).
        matmul_precision: Precision for float32 matrix multiplication.
        precision: Mixed precision setting passed to pl.Trainer.
        root_dir: Root directory for saving logs and checkpoints.
                  Each fold will be saved in a subdirectory 'fold_{i}'.
        **trainer_kwargs: Additional arguments passed to pl.Trainer.

    Returns:
        List of fitted PyTorch Lightning Trainer objects (one per fold).
    """
    k = genome_config["fold_args"]["k"]
    trainers = []
    root_path = Path(root_dir)

    for i in range(k):
        test_fold = i
        val_fold = (i + 1) % k

        # Setup directory for this fold
        fold_dir = root_path / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for Fold {test_fold} (Val: {val_fold})...")

        trainer = train_single(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            model_config=model_config,
            train_config=train_config,
            test_fold=test_fold,
            val_fold=val_fold,
            compile=compile,
            num_workers=num_workers,
            in_memory=in_memory,
            matmul_precision=matmul_precision,
            precision=precision,
            root_dir=fold_dir,
            val_batch_size=val_batch_size,
            **trainer_kwargs,
        )

        trainers.append(trainer)

    return trainers
