#!/usr/bin/env python
"""
GemiNet training example for ChIP-seq data using Cerberus.

This script implements a GemiNet training run using the MDA-PCA-2b AR dataset.
GemiNet is a drop-in replacement for BPNet using Projected Gated Convolutions (PGC).

It follows the standard configuration:
- Input: 2114bp DNA sequence
- Output: 1000bp base-resolution profile
- Model: GemiNet (8 dilated PGC layers)
- Loss: BPNetLoss (Multinomial NLL + MSE log counts)
- Training: Based on peak intervals (narrowPeak)

Usage:
    python examples/chip_ar_mdapca2b_geminet.py --batch-size 32 --max-epochs 50
    python examples/chip_ar_mdapca2b_geminet.py --multi --batch-size 32
"""

import argparse
import torch
torch.set_flush_denormal(True)
from pathlib import Path
from pprint import pprint

# Cerberus imports
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
# Import GemiNet model
from cerberus.models.geminet import GemiNet
# Import BPNet Loss/Metrics (since output heads are compatible)
from cerberus.models.bpnet import BPNetMetricCollection, BPNetLoss
from cerberus.entrypoints import train_single, train_multi

def get_args():
    parser = argparse.ArgumentParser(description="Train a GemiNet model with Cerberus")
    
    # Script arguments
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory to store/load data")
    parser.add_argument("--output-dir", type=str, default="tests/data/models/chip_ar_mdapca2b_geminet", help="Root directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs")
    
    # Mode arguments
    parser.add_argument("--multi", action="store_true", help="Run multi-fold cross-validation instead of single fold")

    # Hyperparameters
    parser.add_argument("--jitter", type=int, default=256, help="Maximum jitter for data augmentation (half-width)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for count loss (lambda)")
    parser.add_argument("--expansion", type=int, default=1, help="GemiNet expansion factor (default: 1)")

    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "cpu", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or 'auto'")
    
    return parser.parse_args()

def main():
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
    
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")

    # 1. Download/Check Data
    print("Downloading/Checking Human Reference (hg38)...")
    genome_files = download_human_reference(data_dir / "genome", name="hg38")

    print("Downloading/Checking Dataset (MDA-PCA-2b AR)...")
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

    # Data Config for GemiNet
    # Input: 2048bp DNA (Cleaner power-of-2 length, possible due to GemiNet's flexibility)
    # Output: 1024bp at base resolution
    input_len = 2048
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
        "use_sequence": True,
    }

    # Sampler Config - Peak Intervals
    padded_size = input_len + 2 * max_jitter
    print(f"Using Peak Sampler (Intervals) with padded_size={padded_size}...")
    
    sampler_config: SamplerConfig = {
        "sampler_type": "interval",
        "padded_size": padded_size,
        "sampler_args": {
            "intervals_path": dataset_files["narrowPeak"]
        }
    }

    # Train Config
    train_config: TrainConfig = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 15,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {
            "num_epochs": args.max_epochs,
            "warmup_epochs": 10,
            "min_lr": 1e-5
        }
    }

    # Model Config for GemiNet
    print("Using GemiNet Model...")
    model_config: ModelConfig = {
        "name": "GemiNet",
        "model_cls": GemiNet,
        "loss_cls": BPNetLoss,
        "loss_args": {
            "alpha": args.alpha,
        },
        "metrics_cls": BPNetMetricCollection,
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "filters": 64,
            "n_dilated_layers": 8,
            "expansion": args.expansion # GemiNet specific argument
        }
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
        print(f"[INFO] Using Apple Silicon (MPS) acceleration.")
        if num_workers > 0:
            print(f"[WARN] num_workers={num_workers} may cause instability on MPS. Recommend setting --num-workers 0.")
    
    # Precision settings
    precision_args = {
        "precision": "16-mixed",
        "matmul_precision": "high",
        "accelerator": accelerator,
        "devices": devices,
        "strategy": "ddp" if accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto"
    }

    print("\nConfigurations:")
    print("-" * 20)
    print("Genome Config:")
    pprint(genome_config)
    print("\nData Config:")
    pprint(data_config)
    print("\nSampler Config:")
    pprint(sampler_config)
    print("\nTrain Config:")
    pprint(train_config)
    print("\nModel Config:")
    pprint(model_config)
    print("\nPrecision and Hardware Args:")
    pprint(precision_args)
    print("-" * 20 + "\n")


    if args.multi:
        print("Starting Multi-Fold Training (train_multi)...")
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
        print("Starting Single Fold Training (train_single)...")
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

    print(f"Training finished. Logs and checkpoints are in {output_dir}")

if __name__ == "__main__":
    main()
