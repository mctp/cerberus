#!/usr/bin/env python
"""
Pomeranian Training Tool.

This script implements a generic training tool for Pomeranian models, allowing
users to provide any BigWig and BED file for training.

Models:
- Pomeranian (Default): Large Kernel (9), Factorized Stem. Input 2112bp.
- PomeranianK5 (--k5): Medium Kernel (5), Factorized Stem. Input 2112bp.

Usage:
    python tools/train_pomeranian.py --bigwig path/to/signal.bw --peaks path/to/peaks.narrowPeak --output-dir models/my_model
"""

import argparse
import logging
import os
from pathlib import Path
from pprint import pformat

import torch

# Cerberus imports
import cerberus
from cerberus.config import (
    DataConfig,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.download import download_human_reference
from cerberus.genome import create_genome_config
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs


def _parse_alpha(value: str) -> "float | str":
    """Accept a float or the literal string 'adaptive' for --alpha."""
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--alpha must be a float or 'adaptive', got: {value!r}"
        ) from None


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Pomeranian models with Cerberus using any BigWig and BED file"
    )

    # Input files
    parser.add_argument(
        "--bigwig", type=str, required=True, help="Path to the BigWig file (signal)"
    )
    parser.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Path to the BED/narrowPeak file (training regions)",
    )

    # Genome arguments
    parser.add_argument(
        "--genome", type=str, default="hg38", help="Genome name (default: hg38)"
    )
    parser.add_argument(
        "--species", type=str, default="human", help="Species (default: human)"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        help="Path to genome FASTA file (if not provided, will try to download for hg38)",
    )
    parser.add_argument("--blacklist", type=str, help="Path to blacklist file")
    parser.add_argument("--gaps", type=str, help="Path to gaps file")

    # Script arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="tests/data",
        help="Directory to store/load data (e.g., genome reference)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for logs and checkpoints",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data sampling (default: 42)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Disable tqdm progress bar during training",
    )

    # Mode arguments
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run multi-fold cross-validation instead of single fold",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=1,
        help="Validation fold (for single-fold training)",
    )
    parser.add_argument("--test-fold", type=int, default=0, help="Test fold")

    # Model variants
    parser.add_argument(
        "--k5", action="store_true", help="Use PomeranianK5 (Medium Kernel Variant)"
    )

    # Hyperparameters
    parser.add_argument(
        "--input-len", type=int, default=2112, help="Input sequence length"
    )
    parser.add_argument(
        "--output-len", type=int, default=1024, help="Output signal length"
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=256,
        help="Maximum jitter for data augmentation (half-width)",
    )
    parser.add_argument(
        "--alpha",
        type=_parse_alpha,
        default="adaptive",
        help="Weight for count loss. Use 'adaptive' (default) to compute from training data "
        "(alpha = median_total_counts / 10), or a float to set explicitly.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bpnet",
        choices=["bpnet", "poisson", "nb"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--total-count",
        type=float,
        default=10.0,
        help="Total count (dispersion) parameter for NB loss",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.0,
        help="Ratio of background (negative) intervals to peaks",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling factor for targets (e.g., 1000 for fractional BigWig values)",
    )
    parser.add_argument(
        "--count-pseudocount",
        type=float,
        default=150.0,
        help="Additive offset before log-transforming count targets (in raw coverage units)",
    )

    # Pretrained weights
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model.pt for warm-start / fine-tuning",
    )

    # Training parameters
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--min-lr", type=float, default=5e-6, help="Minimum learning rate"
    )

    # Hardware arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
        help="Accelerator type",
    )
    parser.add_argument(
        "--devices", type=str, default="auto", help="Number of devices or 'auto'"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "mps", "full"],
        help="Precision strategy: 'bf16' for NVIDIA bf16-mixed (default), "
        "'mps' for Apple Silicon fp16-mixed, "
        "'full' for safest float32 (32-true, matmul=highest, no compile)",
    )

    return parser.parse_args()


def main():
    # Setup logging
    cerberus.setup_logging()
    logging.info("Starting Generic Pomeranian training tool...")

    args = get_args()

    # Setup directories
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).resolve()
    if args.multi:
        output_dir = output_dir / "multi-fold"
    else:
        output_dir = output_dir / "single-fold"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Data Directory: {data_dir}")
    logging.info(f"Output Directory: {output_dir}")

    # 1. Genome Reference
    if not args.fasta:
        if args.genome == "hg38":
            logging.info("Downloading/Checking Human Reference (hg38)...")
            genome_files = download_human_reference(data_dir / "genome", name="hg38")
            fasta_path = genome_files["fasta"]
            blacklist_path = args.blacklist or genome_files["blacklist"]
            gaps_path = args.gaps or genome_files["gaps"]
        else:
            raise ValueError(
                f"Fasta path must be provided for genome {args.genome} if not hg38"
            )
    else:
        fasta_path = Path(args.fasta)
        blacklist_path = Path(args.blacklist) if args.blacklist else None
        gaps_path = Path(args.gaps) if args.gaps else None

    # 2. Configuration

    # Build exclude_intervals dict, filtering out None values
    exclude_intervals = {}
    if blacklist_path:
        exclude_intervals["blacklist"] = blacklist_path
    if gaps_path:
        exclude_intervals["gaps"] = gaps_path

    # Genome Config
    genome_config = create_genome_config(
        name=args.genome,
        fasta_path=fasta_path,
        species=args.species,
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": args.val_fold, "test_fold": args.test_fold},
        exclude_intervals=exclude_intervals,
    )

    # Data Config
    input_len = args.input_len
    output_len = args.output_len
    output_bin_size = 1
    max_jitter = args.jitter
    target_scale = args.target_scale

    data_config = DataConfig(
        inputs={},
        targets={"signal": args.bigwig},
        input_len=input_len,
        output_len=output_len,
        max_jitter=max_jitter,
        output_bin_size=output_bin_size,
        encoding="ACGT",
        log_transform=False,  # Uses raw counts for multinomial loss
        reverse_complement=True,  # Augmentation
        use_sequence=True,
        target_scale=target_scale,
    )

    # Sampler Config - Peak Intervals
    padded_size = input_len + 2 * max_jitter
    logging.info(
        f"Using Peak Sampler (Positives + Negatives) with padded_size={padded_size}..."
    )

    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=padded_size,
        sampler_args={
            "intervals_path": args.peaks,
            "background_ratio": args.background_ratio,
        },
    )

    # Train Config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        optimizer=args.optimizer,
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        scheduler_type=args.scheduler_type,
        scheduler_args={
            "num_epochs": args.max_epochs,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
        },
        adam_eps=1e-8,
        gradient_clip_val=None,
    )

    # Model Config
    if args.k5:
        logging.info("Using PomeranianK5 (Medium Kernel) Model...")
        model_args = {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "filters": 64,
            "n_dilated_layers": 8,
            "conv_kernel_size": [11, 11],
            "dil_kernel_size": 5,
            "profile_kernel_size": 49,
            "expansion": 1,
            "dropout": 0.1,
            "predict_total_count": True,
            "stem_expansion": 2,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
        }
        model_cls_name = "cerberus.models.pomeranian.PomeranianK5"
        model_name = "PomeranianK5"
    else:
        # Default to Pomeranian (K9 Config)
        logging.info("Using Pomeranian (Default) Model...")
        model_args = {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "filters": 64,
            "n_dilated_layers": 8,
            "conv_kernel_size": [11, 11],
            "dil_kernel_size": 9,
            "profile_kernel_size": 45,
            "expansion": 1,
            "dropout": 0.1,
            "predict_total_count": True,
            "stem_expansion": 2,
            "dilations": [1, 1, 2, 4, 8, 16, 32, 64],
        }
        model_cls_name = "cerberus.models.pomeranian.Pomeranian"
        model_name = "Pomeranian"

    if args.loss == "poisson":
        loss_cls = "cerberus.loss.PoissonMultinomialLoss"
        loss_args = {"count_weight": args.alpha}
        logging.info("Using PoissonMultinomialLoss (count_weight=%s)...", args.alpha)
    elif args.loss == "nb":
        loss_cls = "cerberus.loss.NegativeBinomialMultinomialLoss"
        loss_args = {"count_weight": args.alpha, "total_count": args.total_count}
        logging.info(
            "Using NegativeBinomialMultinomialLoss (count_weight=%s, total_count=%s)...",
            args.alpha,
            args.total_count,
        )
    else:
        loss_cls = "cerberus.models.bpnet.BPNetLoss"
        loss_args = {"alpha": args.alpha}
        logging.info("Using BPNetLoss (alpha=%s)...", args.alpha)

    pretrained: list[PretrainedConfig] = []
    if args.pretrained:
        pretrained.append(
            PretrainedConfig(
                weights_path=args.pretrained,
                source=None,
                target=None,
                freeze=False,
            )
        )

    model_config = ModelConfig(
        name=model_name,
        model_cls=model_cls_name,
        loss_cls=loss_cls,
        loss_args=loss_args,
        metrics_cls="cerberus.models.pomeranian.PomeranianMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=pretrained,
        count_pseudocount=args.count_pseudocount * target_scale,
    )

    # 3. Training
    # Handle devices argument
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass

    # Hardware-specific optimization
    accelerator = args.accelerator
    num_workers = args.num_workers

    # Auto-detect Apple Silicon (MPS)
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    if accelerator == "mps":
        logging.info("Using Apple Silicon (MPS) acceleration.")
        if num_workers > 0:
            logging.warning(
                "num_workers=%d may cause instability on MPS. Recommend setting --num-workers 0.",
                num_workers,
            )

    precision_args = get_precision_kwargs(args.precision, accelerator, devices)

    logging.info("Configurations:")
    logging.info("Genome Config:\n%s", pformat(genome_config))
    logging.info("Data Config:\n%s", pformat(data_config))
    logging.info("Sampler Config:\n%s", pformat(sampler_config))
    logging.info("Train Config:\n%s", pformat(train_config))
    logging.info("Model Config:\n%s", pformat(model_config))
    logging.info("Precision and Hardware Args:\n%s", pformat(precision_args))

    if args.multi:
        logging.info("Starting Multi-Fold Training (train_multi)...")
        train_multi(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            model_config=model_config,
            train_config=train_config,
            num_workers=args.num_workers,
            in_memory=False,
            root_dir=str(output_dir),
            enable_checkpointing=True,
            log_every_n_steps=10,
            val_batch_size=args.batch_size * 4,
            enable_progress_bar=not args.silent,
            seed=args.seed,
            **precision_args,
        )
    else:
        logging.info("Starting Single Fold Training (train_single)...")
        train_single(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            model_config=model_config,
            train_config=train_config,
            num_workers=args.num_workers,
            in_memory=False,
            root_dir=str(output_dir),
            enable_checkpointing=True,
            log_every_n_steps=10,
            val_batch_size=args.batch_size * 4,
            enable_progress_bar=not args.silent,
            seed=args.seed,
            **precision_args,
        )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info(
            f"Training finished. Logs and checkpoints are in subdirectories of {output_dir}"
        )


if __name__ == "__main__":
    main()
