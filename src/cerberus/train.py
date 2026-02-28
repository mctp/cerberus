import os
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from typing import Any
from pathlib import Path

from .datamodule import CerberusDataModule
from .config import (
    TrainConfig,
    ModelConfig,
    DataConfig,
    GenomeConfig,
    SamplerConfig,
)
from .module import instantiate, configure_callbacks
from .model_ensemble import update_ensemble_metadata

logger = logging.getLogger(__name__)


def compute_counts_loss_weight(median_counts: float, scale: float = 10.0) -> float:
    """
    Compute the count loss weight (alpha) from training data statistics.

    Implements the formula from chrombpnet-pytorch:
        alpha = median_total_counts / scale    (default scale=10)

    The profile loss (full multinomial NLL) scales linearly with peak depth N,
    while the count loss (MSE of log-counts) scales as (log N)^2. Without a
    data-derived weight, the profile term increasingly dominates as depth grows.
    Linear scaling of alpha with N counteracts this. See
    docs/internal/adaptive_counts_loss_weight.md for the full derivation.

    Args:
        median_counts: Median total signal counts per peak from the training fold,
            already scaled by data_config["target_scale"]. Obtain from
            CerberusDataModule.compute_median_counts().
        scale: Divisor. Default 10 matches chrombpnet-pytorch. Use a smaller value
            for stronger count supervision, larger for stronger profile supervision.

    Returns:
        Count loss weight to pass as loss_args["alpha"] or loss_args["count_weight"].

    Raises:
        ValueError: If median_counts is not positive.
    """
    if median_counts <= 0:
        raise ValueError(f"median_counts must be positive, got {median_counts}")
    return median_counts / scale


def resolve_adaptive_loss_args(
    model_config: "ModelConfig",
    datamodule: "CerberusDataModule",
    n_samples: int = 2000,
) -> "ModelConfig":
    """
    Resolve "adaptive" sentinel values in loss_args to data-derived floats.

    Any loss_args entry whose value is the string "adaptive" is replaced with
    compute_counts_loss_weight(datamodule.compute_median_counts()). This allows
    users to write model configs like:

        loss_args = {"alpha": "adaptive"}           # BPNetLoss
        loss_args = {"count_weight": "adaptive"}    # MSEMultinomialLoss / PoissonMultinomialLoss

    and have the weight computed from the actual training data rather than a
    hard-coded constant. The datamodule must already be setup (setup() called).

    The returned ModelConfig is a new dict; the input is not modified.

    Args:
        model_config: ModelConfig dict, possibly containing "adaptive" in loss_args.
        datamodule: A setup CerberusDataModule for the current fold.
        n_samples: Number of training intervals to sample when computing the median.

    Returns:
        A new ModelConfig with all "adaptive" values in loss_args replaced by floats.
    """
    loss_args = model_config["loss_args"]
    adaptive_keys = [k for k, v in loss_args.items() if v == "adaptive"]
    if not adaptive_keys:
        return model_config

    median_counts = datamodule.compute_median_counts(n_samples=n_samples)
    weight = compute_counts_loss_weight(median_counts)
    logger.info(
        f"Resolved adaptive loss_args {adaptive_keys} → {weight:.4f} "
        f"(median_counts={median_counts:.1f} / 10)"
    )
    resolved_loss_args = {
        k: (weight if v == "adaptive" else v)
        for k, v in loss_args.items()
    }
    return {**model_config, "loss_args": resolved_loss_args}


def _train(
    model_config: ModelConfig,
    data_config: DataConfig,
    datamodule: CerberusDataModule,
    train_config: TrainConfig,
    compile: bool = False,
    genome_config: GenomeConfig | None = None,
    sampler_config: SamplerConfig | None = None,
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
    Train a Cerberus model from configuration.

    Handles the full training lifecycle in order:
      1. Datamodule setup (loads datasets, sets batch_size / num_workers).
      2. Adaptive loss weight resolution — any "adaptive" sentinel in loss_args
         is replaced with compute_counts_loss_weight(datamodule.compute_median_counts())
         before the module is instantiated.
      3. Module instantiation with the resolved model_config.
      4. trainer.fit().

    Keeping setup before instantiation ensures that loss weights derived from
    training data statistics are always concrete floats by the time the loss
    object is constructed.

    Args:
        model_config: Model architecture configuration. loss_args values may be
            the string "adaptive"; see resolve_adaptive_loss_args().
        data_config: Data inputs/outputs configuration.
        datamodule: A CerberusDataModule (not yet setup).
        train_config: Training hyperparameters.
        compile: Whether to compile the model with torch.compile (default: False).
        genome_config: Genome configuration; passed through to the module for
            hparam logging.
        sampler_config: Sampler configuration; passed through to the module for
            hparam logging.
        callbacks: Optional list of additional PyTorch Lightning callbacks.
            Added to the default set (LearningRateMonitor, ModelCheckpoint,
            EarlyStopping) unless a callback of the same type is already present.
        num_workers: Number of DataLoader workers (default: 0).
        in_memory: Whether to load data into memory (default: False).
        matmul_precision: Precision for float32 matrix multiplication.
            Options: 'highest', 'high', 'medium'. (default: 'highest').
        precision: Mixed precision setting passed to pl.Trainer.
            Options: '32-true', '16-mixed', 'bf16-mixed', etc. (default: '32-true').
        root_dir: Root directory for saving logs and checkpoints.
            If provided, it overrides 'default_root_dir' in trainer_kwargs.
        **trainer_kwargs: Additional arguments passed directly to pl.Trainer.

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
    current_callbacks = configure_callbacks(
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
        save_dir = trainer_kwargs.get("default_root_dir", ".")
        if save_dir is None:
            save_dir = "."
        trainer_kwargs["logger"] = pl_loggers.CSVLogger(save_dir=str(save_dir))

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=current_callbacks,
        precision=precision,
        reload_dataloaders_every_n_epochs=train_config["reload_dataloaders_every_n_epochs"],
        gradient_clip_val=train_config["gradient_clip_val"],
        **trainer_kwargs,
    )

    # 1. Setup DataModule — must happen before adaptive resolution and instantiation.
    datamodule.setup(
        batch_size=train_config["batch_size"],
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        in_memory=in_memory,
    )

    # 2. Resolve any "adaptive" sentinels in loss_args to data-derived floats.
    # Returns a new ModelConfig; the input is not modified so train_multi can
    # safely reuse the same model_config dict across folds.
    model_config = resolve_adaptive_loss_args(model_config, datamodule)

    # 3. Instantiate module with the resolved model_config.
    module = instantiate(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        compile=compile,
        genome_config=genome_config,
        sampler_config=sampler_config,
    )

    # 4. Train
    trainer.fit(module, datamodule=datamodule)

    return trainer


def train_single(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    test_fold: int = 0,
    val_fold: int = 1,
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

    Acts as a high-level entrypoint that constructs the CerberusDataModule and
    delegates to _train(), which handles datamodule setup, adaptive loss weight
    resolution, module instantiation, and trainer.fit() in that order.

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
    # 0. Update Metadata and Prepare Directory
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        update_ensemble_metadata(root_dir, test_fold)

    root_path = Path(root_dir)
    fold_dir = root_path / f"fold_{test_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate DataModule for this fold
    datamodule = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        test_fold=test_fold,
        val_fold=val_fold,
    )

    # Update genome_config to reflect the actual folds used
    # This ensures hparams.yaml logged by Lightning matches the directory structure
    if "fold_args" in genome_config:
        genome_config = genome_config.copy()
        genome_config["fold_args"] = genome_config["fold_args"].copy()
        genome_config["fold_args"]["test_fold"] = test_fold
        genome_config["fold_args"]["val_fold"] = val_fold

    # 2. Train (setup → adaptive resolution → instantiate → fit happen inside _train)
    return _train(
        model_config=model_config,
        data_config=data_config,
        datamodule=datamodule,
        train_config=train_config,
        compile=compile,
        genome_config=genome_config,
        sampler_config=sampler_config,
        num_workers=num_workers,
        in_memory=in_memory,
        matmul_precision=matmul_precision,
        precision=precision,
        root_dir=fold_dir,
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

    for i in range(k):
        test_fold = i
        val_fold = (i + 1) % k

        logger.info(f"Starting training for Fold {test_fold} (Val: {val_fold})...")

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
            root_dir=root_dir,
            val_batch_size=val_batch_size,
            **trainer_kwargs,
        )

        trainers.append(trainer)

    return trainers
