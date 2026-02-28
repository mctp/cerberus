#!/usr/bin/env python
"""
Pomeranian Training Example.

This script implements training for Pomeranian models on the MDA-PCA-2b AR dataset.

Models:
- Pomeranian (Default): Large Kernel (9), Factorized Stem. Input 2112bp.
- PomeranianK5 (--k5): Medium Kernel (5), Factorized Stem. Input 2112bp.

Usage:
    python examples/chip_ar_mdapca2b_pomeranian.py --batch-size 32         # Uses Default (K9)
    python examples/chip_ar_mdapca2b_pomeranian.py --k5 --batch-size 32    # Uses K5
"""

import argparse
import os
import logging
import torch
from pathlib import Path
from pprint import pformat

# Cerberus imports
import cerberus
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.train import train_single, train_multi

def get_args():
    parser = argparse.ArgumentParser(description="Train Pomeranian models with Cerberus")
    
    # Script arguments
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory to store/load data")
    parser.add_argument("--output-dir", type=str, default="tests/data/models/chip_ar_mdapca2b_pomeranian", help="Root directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs")
    
    # Mode arguments
    parser.add_argument("--multi", action="store_true", help="Run multi-fold cross-validation instead of single fold")
    
    # Model variants
    parser.add_argument("--k5", action="store_true", help="Use PomeranianK5 (Medium Kernel Variant)")

    # Hyperparameters
    parser.add_argument("--jitter", type=int, default=256, help="Maximum jitter for data augmentation (half-width)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for count loss (lambda)")
    parser.add_argument("--loss", type=str, default="bpnet", choices=["bpnet", "poisson", "nb"], help="Loss function to use")
    parser.add_argument("--total-count", type=float, default=10.0, help="Total count (dispersion) parameter for NB loss")
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or 'auto'")
    
    return parser.parse_args()

def main():
    # Setup logging
    cerberus.setup_logging()
    logging.info("Starting Pomeranian training script...")

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

    # 1. Download/Check Data
    logging.info("Downloading/Checking Human Reference (hg38)...")
    genome_files = download_human_reference(data_dir / "genome", name="hg38")

    logging.info("Downloading/Checking Dataset (MDA-PCA-2b AR)...")
    dataset_files = download_dataset(data_dir / "dataset", name="mdapca2b_ar")

    # 2. Configuration
    
    # Genome Config
    genome_config: GenomeConfig = create_genome_config(
        name="hg38",
        fasta_path=genome_files["fasta"],
        species="human",
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": 1, "test_fold": 0},
        exclude_intervals={
            "blacklist": genome_files["blacklist"],
            "gaps": genome_files["gaps"],
        }
    )

    # Data Config
    # Both use 2112bp input
    input_len = 2112
    output_len = 1024
    output_bin_size = 1
    max_jitter = args.jitter

    data_config: DataConfig = {
        "inputs": {}, 
        "targets": {"signal": dataset_files["bigwig"]},
        "input_len": input_len,
        "output_len": output_len, 
        "max_jitter": max_jitter,
        "output_bin_size": output_bin_size,
        "encoding": "ACGT",
        "log_transform": False, # Uses raw counts for multinomial loss
        "reverse_complement": True, # Augmentation
        "target_scale": 1.0,
        "use_sequence": True,
    }

    # Sampler Config - Peak Intervals
    padded_size = input_len + 2 * max_jitter
    logging.info(f"Using Peak Sampler (Positives + Negatives) with padded_size={padded_size}...")
    
    sampler_config: SamplerConfig = {
        "sampler_type": "peak",
        "padded_size": padded_size,
        "sampler_args": {
            "intervals_path": dataset_files["narrowPeak"],
            "background_ratio": 1.0,
        }
    }

    # Train Config
    train_config: TrainConfig = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "patience": 10,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "scheduler_type": "cosine",
        "scheduler_args": {
            "num_epochs": args.max_epochs,
            "warmup_epochs": 10,
            "min_lr": 5e-6
        },
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }

    # Model Config
    if args.k5:
        logging.info(f"Using PomeranianK5 (Medium Kernel) Model...")
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
        logging.info(f"Using Pomeranian (Default) Model...")
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
        logging.info(f"Using PoissonMultinomialLoss (count_weight={args.alpha})...")
    elif args.loss == "nb":
        loss_cls = "cerberus.loss.NegativeBinomialMultinomialLoss"
        loss_args = {"count_weight": args.alpha, "total_count": args.total_count}
        logging.info(f"Using NegativeBinomialMultinomialLoss (count_weight={args.alpha}, total_count={args.total_count})...")
    else:
        loss_cls = "cerberus.models.bpnet.BPNetLoss"
        loss_args = {"alpha": args.alpha}
        logging.info(f"Using BPNetLoss (alpha={args.alpha})...")

    model_config: ModelConfig = {
        "name": model_name,
        "model_cls": model_cls_name,
        "loss_cls": loss_cls,
        "loss_args": loss_args,
        "metrics_cls": "cerberus.models.pomeranian.PomeranianMetricCollection",
        "metrics_args": {},
        "model_args": model_args
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
        logging.info(f"[INFO] Using Apple Silicon (MPS) acceleration.")
        if num_workers > 0:
            logging.warning(f"[WARN] num_workers={num_workers} may cause instability on MPS. Recommend setting --num-workers 0.")
    
    # Precision settings
    if accelerator == "mps":
        # MPS-optimized settings
        precision_args = {
            "precision": "16-mixed",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto", 
            "compile": False
        }
    else:
        # NVIDIA A100 / CUDA optimized settings
        precision_args = {
            "precision": "bf16-mixed",
            "matmul_precision": "medium",
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "ddp_find_unused_parameters_false" if accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto",
            "benchmark": True,
            "compile": True
        }

    logging.info("\nConfigurations:")
    logging.info("-" * 20)
    logging.info("Genome Config:
%s", pformat(genome_config))
    logging.info("Data Config:
%s", pformat(data_config))
    logging.info("Sampler Config:
%s", pformat(sampler_config))
    logging.info("Train Config:
%s", pformat(train_config))
    logging.info("Model Config:
%s", pformat(model_config))
    logging.info("Precision and Hardware Args:
%s", pformat(precision_args))
    logging.info("-" * 20 + "\n")


    if args.multi:
        logging.info("Starting Multi-Fold Training (train_multi)...")
        logging.info("Calling train_multi...")
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
            **precision_args
        )
    else:
        logging.info("Starting Single Fold Training (train_single)...")
        logging.info("Calling train_single...")
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
            **precision_args
        )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info(f"Training finished. Logs and checkpoints are in subdirectories of {output_dir}")

if __name__ == "__main__":
    main()
