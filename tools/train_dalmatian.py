#!/usr/bin/env python
"""
Dalmatian Training Tool.

This script implements a generic training tool for Dalmatian models.
Dalmatian is an end-to-end bias-factorized sequence-to-function model
that decomposes signal into BiasNet (local Tn5 bias, RF~105bp, Conv1d+ReLU)
and SignalNet (regulatory grammar, RF~1089bp, Pomeranian) sub-networks.
Loss: L_recon (combined, all examples) + bias_weight * L_bias (bias-only, background).

Designed for ATAC-seq pseudobulk data: sequence+BED input, BigWig output.

Usage:
    python tools/train_dalmatian.py --bigwig path/to/signal.bw --peaks path/to/peaks.bed --output-dir models/my_model
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
from cerberus.pseudocount import resolve_reads_equivalent_pseudocount
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Dalmatian models with Cerberus using any BigWig and BED file"
    )

    # Input files
    parser.add_argument(
        "--bigwig",
        type=str,
        required=True,
        help="Path to the BigWig file (target signal)",
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
        "--background-ratio",
        type=float,
        default=1.0,
        help="Ratio of background (negative) intervals to peaks",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling factor for targets (1.0 for raw-count pseudobulk BigWig values)",
    )
    parser.add_argument(
        "--count-pseudocount",
        type=float,
        default=1.0,
        help="Additive offset before log-transforming count targets. "
        "Ignored when --pseudocount-reads is set.",
    )
    parser.add_argument(
        "--pseudocount-reads",
        type=float,
        default=None,
        help="Scale-aware pseudocount in reads-equivalent units (overrides --count-pseudocount).",
    )
    parser.add_argument(
        "--read-length",
        type=int,
        default=150,
        help="Read or fragment length in bp (used only with --pseudocount-reads).",
    )
    parser.add_argument(
        "--input-scale",
        type=str,
        default="raw",
        choices=["raw", "cpm"],
        help="Input bigWig scale (used only with --pseudocount-reads).",
    )
    parser.add_argument(
        "--total-reads",
        type=float,
        default=None,
        help="Library total reads (required when --input-scale=cpm and --pseudocount-reads is set).",
    )

    # Loss arguments
    parser.add_argument(
        "--base-loss",
        type=str,
        default="mse",
        choices=["mse", "poisson"],
        help="Base loss function for DalmatianLoss (default: mse)",
    )
    parser.add_argument(
        "--bias-weight",
        type=float,
        default=1.0,
        help="Weight for bias-only reconstruction term",
    )

    # BiasNet overrides
    parser.add_argument(
        "--bias-filters",
        type=int,
        default=12,
        help="BiasNet filter count (default: 12)",
    )
    parser.add_argument(
        "--bias-dropout", type=float, default=0.1, help="BiasNet dropout rate"
    )

    # SignalNet overrides
    parser.add_argument(
        "--signal-preset",
        type=str,
        default="standard",
        choices=["large", "standard"],
        help="SignalNet preset: 'standard' (f=64, ~150K, Pomeranian K9) or 'large' (f=256, ~3.9M)",
    )
    parser.add_argument(
        "--signal-filters",
        type=int,
        default=None,
        help="SignalNet model dimension (overrides preset)",
    )
    parser.add_argument(
        "--signal-dropout", type=float, default=0.1, help="SignalNet dropout rate"
    )

    # Pretrained weights
    parser.add_argument(
        "--pretrained-bias",
        type=str,
        default=None,
        help="Path to pretrained BiasNet model.pt for weight initialization",
    )
    parser.add_argument(
        "--freeze-bias",
        action="store_true",
        help="Freeze BiasNet weights after loading (requires --pretrained-bias)",
    )

    # Training parameters
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (higher than typical to survive Phase 1 plateau)",
    )
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="default",
        help="Learning rate scheduler type (default = constant)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="Number of warmup epochs"
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
    logging.info("Starting Dalmatian training tool...")

    args = get_args()

    # Validate pretrained args
    if args.freeze_bias and not args.pretrained_bias:
        raise SystemExit("--freeze-bias requires --pretrained-bias")

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
        log_transform=False,
        reverse_complement=True,
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
    logging.info("Using Dalmatian (bias-factorized) Model...")

    # input_len, output_len, output_bin_size are injected by instantiate_model
    bias_args: dict[str, object] = {
        "filters": args.bias_filters,
        "dropout": args.bias_dropout,
    }
    signal_args: dict[str, object] = {
        "dropout": args.signal_dropout,
    }
    if args.signal_filters is not None:
        signal_args["filters"] = args.signal_filters

    model_args: dict[str, object] = {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "signal_preset": args.signal_preset,
        "bias_args": bias_args,
        "signal_args": signal_args,
    }

    # DalmatianLoss configuration
    if args.base_loss == "poisson":
        base_loss_cls = "cerberus.loss.PoissonMultinomialLoss"
        base_loss_args: dict[str, object] = {"count_per_channel": True}
        logging.info("Using DalmatianLoss with PoissonMultinomialLoss base...")
    else:
        base_loss_cls = "cerberus.loss.MSEMultinomialLoss"
        base_loss_args = {"count_per_channel": True}
        logging.info("Using DalmatianLoss with MSEMultinomialLoss base...")

    if args.pseudocount_reads is not None:
        count_pseudocount_scaled = resolve_reads_equivalent_pseudocount(
            reads_equiv=args.pseudocount_reads,
            read_length=args.read_length,
            bin_size=output_bin_size,
            target_scale=target_scale,
            input_scale=args.input_scale,
            total_reads=args.total_reads,
        )
    else:
        count_pseudocount_scaled = args.count_pseudocount * target_scale

    loss_args: dict[str, object] = {
        "base_loss_cls": base_loss_cls,
        "base_loss_args": base_loss_args,
        "bias_weight": args.bias_weight,
        # The value below is overridden by ModelConfig.count_pseudocount via
        # instantiate_metrics_and_loss; kept here only for hparam visibility.
        "count_pseudocount": count_pseudocount_scaled,
    }

    # Build pretrained weight configs
    pretrained: list[PretrainedConfig] = []
    if args.pretrained_bias:
        pretrained.append(
            PretrainedConfig(
                weights_path=args.pretrained_bias,
                source=None,
                target="bias_model",
                freeze=args.freeze_bias,
            )
        )

    model_config = ModelConfig(
        name="Dalmatian",
        model_cls="cerberus.models.dalmatian.Dalmatian",
        loss_cls="cerberus.loss.DalmatianLoss",
        loss_args=loss_args,
        metrics_cls="cerberus.models.pomeranian.PomeranianMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=pretrained,
        count_pseudocount=count_pseudocount_scaled,
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
