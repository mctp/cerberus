#!/usr/bin/env python
"""
BiasNet Training Tool.

This script trains a standalone BiasNet model (lightweight Conv1d + ReLU stack)
for Tn5 enzymatic bias modeling. By default it trains on negative (non-peak)
regions only, matching the ChromBPNet bias-model training paradigm.

BiasNet is fully DeepLIFT/DeepSHAP compatible (Conv1d + ReLU + residual add only).

Default configuration (matching exp19f):
  - Filters: 12, Stem: [11, 11], Body: 5 x (k=9, d=1), Head: k=45 linear
  - RF: 105bp, ~9.3K params
  - Loss: MSEMultinomialLoss (count_weight=adaptive)
  - Sampler: negative_peak (background regions only)
  - Optimizer: AdamW, lr=1e-3, wd=1e-4, no scheduler

Usage:
    # Train on ATAC-seq negative peaks (default)
    python tools/train_biasnet.py \\
        --bigwig path/to/signal.bw \\
        --peaks path/to/peaks.narrowPeak \\
        --output-dir models/my_bias_model

    # Train on peaks + background (like Pomeranian)
    python tools/train_biasnet.py \\
        --bigwig path/to/signal.bw \\
        --peaks path/to/peaks.narrowPeak \\
        --output-dir models/my_bias_model \\
        --sampler-type peak
"""

import argparse
import logging
import os
from pathlib import Path
from pprint import pformat

import torch

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


def _parse_count_weight(value: str) -> "float | str":
    """Accept a float or the literal string 'adaptive' for --count-weight."""
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--count-weight must be a float or 'adaptive', got: {value!r}"
        ) from None


def get_args():
    parser = argparse.ArgumentParser(
        description="Train BiasNet (lightweight Tn5 bias model) with Cerberus"
    )

    # Input files
    parser.add_argument("--bigwig", type=str, required=True, help="Path to the BigWig file (signal)")
    parser.add_argument("--peaks", type=str, required=True, help="Path to the BED/narrowPeak file (peak regions)")

    # Genome arguments
    parser.add_argument("--genome", type=str, default="hg38", help="Genome name (default: hg38)")
    parser.add_argument("--species", type=str, default="human", help="Species (default: human)")
    parser.add_argument("--fasta", type=str, help="Path to genome FASTA file (if not provided, will try to download for hg38)")
    parser.add_argument("--blacklist", type=str, help="Path to blacklist file")
    parser.add_argument("--gaps", type=str, help="Path to gaps file")

    # Script arguments
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory to store/load data (e.g., genome reference)")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling (default: 42)")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--silent", action="store_true", help="Disable tqdm progress bar during training")

    # Mode arguments
    parser.add_argument("--multi", action="store_true", help="Run multi-fold cross-validation instead of single fold")
    parser.add_argument("--val-fold", type=int, default=1, help="Validation fold (for single-fold training)")
    parser.add_argument("--test-fold", type=int, default=0, help="Test fold")

    # Data hyperparameters
    parser.add_argument("--output-len", type=int, default=1024, help="Output signal length")
    parser.add_argument("--jitter", type=int, default=256, help="Maximum jitter for data augmentation (half-width)")
    parser.add_argument("--sampler-type", type=str, default="negative_peak",
                        choices=["peak", "negative_peak"],
                        help="Sampler type (default: negative_peak for bias-only training)")
    parser.add_argument("--background-ratio", type=float, default=1.0, help="Ratio of background intervals to peaks (for peak sampler)")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Multiplicative scaling factor for targets")
    parser.add_argument("--count-pseudocount", type=float, default=1.0, help="Additive offset before log-transforming count targets")

    # Architecture
    parser.add_argument("--filters", type=int, default=12, help="Number of conv filters (model dimension)")
    parser.add_argument("--n-layers", type=int, default=5, help="Number of residual tower layers")
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 1, 1, 1, 1],
                        help="Dilation schedule for tower layers")
    parser.add_argument("--dil-kernel-size", type=int, default=9, help="Tower conv kernel size")
    parser.add_argument("--conv-kernel-size", type=int, nargs="+", default=[11, 11],
                        help="Stem kernel size(s)")
    parser.add_argument("--profile-kernel-size", type=int, default=45, help="Profile head spatial kernel size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections in tower blocks")
    parser.add_argument("--linear-head", action="store_true", default=True,
                        help="Use single linear spatial conv for profile head (default: True)")
    parser.add_argument("--no-linear-head", action="store_true",
                        help="Use pointwise+ReLU+spatial conv for profile head")

    # Pretrained weights
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained BiasNet model.pt for warm-start / fine-tuning")

    # Loss
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "bpnet", "poisson"],
                        help="Loss function (default: mse)")
    parser.add_argument("--count-weight", type=_parse_count_weight, default="adaptive",
                        help="Count loss weight. Use 'adaptive' (default) for data-derived, or a float.")

    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--scheduler-type", type=str, default="default", help="Learning rate scheduler type")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs (for cosine scheduler)")
    parser.add_argument("--min-lr", type=float, default=5e-6, help="Minimum learning rate (for cosine scheduler)")

    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or 'auto'")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "mps", "full"],
                        help="Precision strategy: 'bf16' for NVIDIA bf16-mixed (default), "
                             "'mps' for Apple Silicon fp16-mixed, "
                             "'full' for safest float32 (32-true, matmul=highest, no compile)")

    return parser.parse_args()


def main():
    cerberus.setup_logging()
    logging.info("Starting BiasNet training tool...")

    args = get_args()

    # Resolve --no-linear-head override
    if args.no_linear_head:
        args.linear_head = False

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
            raise ValueError(f"Fasta path must be provided for genome {args.genome} if not hg38")
    else:
        fasta_path = Path(args.fasta)
        blacklist_path = Path(args.blacklist) if args.blacklist else None
        gaps_path = Path(args.gaps) if args.gaps else None

    # 2. Configuration

    exclude_intervals = {}
    if blacklist_path:
        exclude_intervals["blacklist"] = blacklist_path
    if gaps_path:
        exclude_intervals["gaps"] = gaps_path

    genome_config = create_genome_config(
        name=args.genome,
        fasta_path=fasta_path,
        species=args.species,
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": args.val_fold, "test_fold": args.test_fold},
        exclude_intervals=exclude_intervals,
    )

    # Compute input_len from architecture shrinkage
    output_len = args.output_len
    stem_shrinkage = sum(k - 1 for k in args.conv_kernel_size)
    tower_shrinkage = sum(d * (args.dil_kernel_size - 1) for d in args.dilations)
    head_shrinkage = args.profile_kernel_size - 1
    total_shrinkage = stem_shrinkage + tower_shrinkage + head_shrinkage
    input_len = output_len + total_shrinkage

    logging.info(
        f"Architecture: RF={total_shrinkage + 1}bp, shrinkage={total_shrinkage} "
        f"(stem={stem_shrinkage}, tower={tower_shrinkage}, head={head_shrinkage}), "
        f"input_len={input_len}"
    )

    max_jitter = args.jitter
    padded_size = input_len + 2 * max_jitter
    target_scale = args.target_scale

    data_config = DataConfig(
        inputs={},
        targets={"signal": args.bigwig},
        input_len=input_len,
        output_len=output_len,
        max_jitter=max_jitter,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=True,
        use_sequence=True,
        target_scale=target_scale,
    )

    # Build the appropriate sampler args based on sampler type
    if args.sampler_type == "negative_peak":
        sampler_args_obj = {
            "intervals_path": args.peaks,
            "background_ratio": args.background_ratio,
        }
    else:
        sampler_args_obj = {
            "intervals_path": args.peaks,
            "background_ratio": args.background_ratio,
        }

    sampler_config = SamplerConfig(
        sampler_type=args.sampler_type,
        padded_size=padded_size,
        sampler_args=sampler_args_obj,
    )

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
    model_args = {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "filters": args.filters,
        "n_layers": args.n_layers,
        "dilations": args.dilations,
        "dil_kernel_size": args.dil_kernel_size,
        "conv_kernel_size": args.conv_kernel_size,
        "profile_kernel_size": args.profile_kernel_size,
        "dropout": args.dropout,
        "predict_total_count": True,
        "residual": not args.no_residual,
        "linear_head": args.linear_head,
    }

    if args.loss == "poisson":
        loss_cls = "cerberus.loss.PoissonMultinomialLoss"
        loss_args: dict = {"count_weight": args.count_weight}
        logging.info("Using PoissonMultinomialLoss (count_weight=%s)...", args.count_weight)
    elif args.loss == "bpnet":
        loss_cls = "cerberus.models.bpnet.BPNetLoss"
        loss_args = {"alpha": args.count_weight}
        logging.info("Using BPNetLoss (alpha=%s)...", args.count_weight)
    else:
        loss_cls = "cerberus.loss.MSEMultinomialLoss"
        loss_args = {
            "count_per_channel": True,
            "count_weight": args.count_weight,
        }
        logging.info("Using MSEMultinomialLoss (count_weight=%s)...", args.count_weight)

    # Build pretrained weight configs
    pretrained: list[PretrainedConfig] = []
    if args.pretrained:
        pretrained.append(PretrainedConfig(
            weights_path=args.pretrained,
            source=None,
            target=None,
            freeze=False,
        ))

    model_config = ModelConfig(
        name="BiasNet",
        model_cls="cerberus.models.biasnet.BiasNet",
        loss_cls=loss_cls,
        loss_args=loss_args,
        metrics_cls="cerberus.models.pomeranian.PomeranianMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=pretrained,
        count_pseudocount=args.count_pseudocount * target_scale,
    )

    # 3. Training
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass

    accelerator = args.accelerator
    num_workers = args.num_workers

    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    if accelerator == "mps":
        logging.info("Using Apple Silicon (MPS) acceleration.")
        if num_workers > 0:
            logging.warning(
                "num_workers=%d may cause instability on MPS. Recommend setting --num-workers 0.",
                num_workers,
            )

    if args.precision == "full":
        precision_args = {
            "precision": "32-true",
            "matmul_precision": "highest",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }
    elif args.precision == "mps":
        precision_args = {
            "precision": "16-mixed",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }
    else:
        precision_args = {
            "precision": "bf16-mixed",
            "matmul_precision": "medium",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "benchmark": True,
            "compile": True,
        }

    logging.info("Configurations:")
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
        logging.info(f"Training finished. Logs and checkpoints are in subdirectories of {output_dir}")


if __name__ == "__main__":
    main()
