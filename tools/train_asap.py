#!/usr/bin/env python
"""
ASAP Training Tool.

This script implements a generic training tool for ASAP (ConvNeXtDCNN) models,
allowing users to provide any BigWig and BED file for training.

Models:
- ConvNeXtDCNN (Default): ConvNeXtV2 stem + Basenji-style dilated residual tower.
  Input 2048bp → Output 512 bins at 4bp resolution (2048bp coverage).

Usage:
    python tools/train_asap.py --bigwig path/to/signal.bw --peaks path/to/peaks.narrowPeak --output-dir models/my_model
"""

import argparse
import os
import logging
import torch
from pathlib import Path
from pprint import pformat

# Cerberus imports
import cerberus
from cerberus.download import download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.train import train_single, train_multi


def get_args():
    parser = argparse.ArgumentParser(description="Train ASAP (ConvNeXtDCNN) models with Cerberus using any BigWig and BED file")

    # Input files
    parser.add_argument("--bigwig", type=str, required=True, help="Path to the BigWig file (signal)")
    parser.add_argument("--peaks", type=str, required=True, help="Path to the BED/narrowPeak file (training regions)")

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
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling (default: 42)")
    parser.add_argument("--silent", action="store_true", help="Disable tqdm progress bar during training")

    # Mode arguments
    parser.add_argument("--multi", action="store_true", help="Run multi-fold cross-validation instead of single fold")
    parser.add_argument("--val-fold", type=int, default=1, help="Validation fold (for single-fold training)")
    parser.add_argument("--test-fold", type=int, default=0, help="Test fold")

    # Hyperparameters
    parser.add_argument("--input-len", type=int, default=2048, help="Input sequence length (must be divisible by output_bin_size)")
    parser.add_argument("--output-len", type=int, default=2048, help="Output profile length in base pairs")
    parser.add_argument("--output-bin-size", type=int, default=4, help="Output bin size in base pairs (output_len must be divisible by this)")
    parser.add_argument("--jitter", type=int, default=256, help="Maximum jitter for data augmentation (half-width)")
    parser.add_argument("--background-ratio", type=float, default=1.0, help="Ratio of background (negative) intervals to peaks")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Multiplicative scaling factor for targets")
    parser.add_argument("--count-pseudocount", type=float, default=1.0, help="Additive offset before log-transforming count targets (in raw coverage units)")

    # Architecture arguments
    parser.add_argument("--residual-blocks", type=int, default=11, help="Number of dilated residual blocks in the Basenji core (default: 11)")
    parser.add_argument("--filters0", type=int, default=256, help="Number of filters after the ConvNeXtV2 stem and throughout the residual tower (default: 256)")
    parser.add_argument("--filters1", type=int, default=128, help="Number of filters in the dilated convolution inside each residual block (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate within residual blocks (default: 0.3)")

    # Pretrained weights
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model.pt for warm-start / fine-tuning")

    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--scheduler-type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")

    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or 'auto'")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "mps", "full"],
                        help="Precision strategy: 'bf16' for NVIDIA bf16-mixed (default), "
                             "'mps' for Apple Silicon fp16-mixed, "
                             "'full' for safest float32 (32-true, matmul=highest, no compile)")

    return parser.parse_args()


def main():
    # Setup logging
    cerberus.setup_logging()
    logging.info("Starting Generic ASAP training tool...")

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
            raise ValueError(f"Fasta path must be provided for genome {args.genome} if not hg38")
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
    genome_config: GenomeConfig = create_genome_config(
        name=args.genome,
        fasta_path=fasta_path,
        species=args.species,
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": args.val_fold, "test_fold": args.test_fold},
        exclude_intervals=exclude_intervals
    )

    # Data Config
    input_len = args.input_len
    output_len = args.output_len
    output_bin_size = args.output_bin_size
    max_jitter = args.jitter

    data_config: DataConfig = {
        "inputs": {},
        "targets": {"signal": args.bigwig},
        "input_len": input_len,
        "output_len": output_len,
        "max_jitter": max_jitter,
        "output_bin_size": output_bin_size,
        "encoding": "ACGT",
        "log_transform": True,  # ASAP trains on log(x+1) data
        "reverse_complement": True,  # Augmentation
        "use_sequence": True,
        "target_scale": args.target_scale,
        "count_pseudocount": args.count_pseudocount,
    }

    # Sampler Config - Peak Intervals
    padded_size = input_len + 2 * max_jitter
    logging.info(f"Using Peak Sampler (Positives + Negatives) with padded_size={padded_size}...")

    sampler_config: SamplerConfig = {
        "sampler_type": "peak",
        "padded_size": padded_size,
        "sampler_args": {
            "intervals_path": args.peaks,
            "background_ratio": args.background_ratio,
        }
    }

    # Train Config
    train_config: TrainConfig = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "scheduler_type": args.scheduler_type,
        "scheduler_args": {
            "num_epochs": args.max_epochs,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
        },
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }

    # Model Config
    logging.info("Using ConvNeXtDCNN (ASAP) Model...")
    pretrained: list[dict[str, object]] = []
    if args.pretrained:
        pretrained.append({"weights_path": args.pretrained, "source": None, "target": None, "freeze": False})

    model_config: ModelConfig = {
        "name": "ConvNeXtDCNN",
        "model_cls": "cerberus.models.asap.ConvNeXtDCNN",
        "loss_cls": "cerberus.loss.ProfilePoissonNLLLoss",
        "loss_args": {
            "log_input": True,
            "log1p_targets": True,
        },
        "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "residual_blocks": args.residual_blocks,
            "filters0": args.filters0,
            "filters1": args.filters1,
            "dropout": args.dropout,
        },
        "pretrained": pretrained,
    }

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
            logging.warning("num_workers=%d may cause instability on MPS. Recommend setting --num-workers 0.", num_workers)

    # Precision settings
    if args.precision == "full":
        # Safest full float32 — no reduced precision, no compile, no cuDNN benchmark.
        # Use when numerical reproducibility or debugging is a priority.
        precision_args = {
            "precision": "32-true",
            "matmul_precision": "highest",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }
    elif args.precision == "mps":
        # Apple Silicon (MPS) — fp16 mixed precision.
        precision_args = {
            "precision": "16-mixed",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "compile": False,
        }
    else:
        # bf16 (default) — NVIDIA Ampere+ bf16 mixed precision.
        precision_args = {
            "precision": "bf16-mixed",
            "matmul_precision": "medium",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "ddp_find_unused_parameters_false" if accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto",
            "benchmark": True,
            "compile": True,
        }

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
            **precision_args
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
            **precision_args
        )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info(f"Training finished. Logs and checkpoints are in subdirectories of {output_dir}")


if __name__ == "__main__":
    main()
