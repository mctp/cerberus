#!/usr/bin/env python
"""
GemiNet training example for ChIP-seq data using Cerberus.

This script implements a GemiNet training run using the MDA-PCA-2b AR dataset.
GemiNet is a drop-in replacement for BPNet using Projected Gated Convolutions (PGC).

It follows the standard configuration:
- Input: 2048bp DNA sequence
- Output: 1024bp base-resolution profile
- Model: GemiNet (8 dilated PGC layers)
- Loss: BPNetLoss (Multinomial NLL + MSE log counts)
- Training: Based on peak intervals (narrowPeak)

Usage:
    python examples/chip_ar_mdapca2b_geminet.py --batch-size 32 --max-epochs 50
    python examples/chip_ar_mdapca2b_geminet.py --multi --batch-size 32
"""

import argparse
import os
import torch
from pathlib import Path
from pprint import pprint

# Cerberus imports
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.train import train_single, train_multi

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
    parser.add_argument("--loss", type=str, default="bpnet", choices=["bpnet", "poisson"], help="Loss function to use")
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "large", "xl"], help="GemiNet model size (small, large, xl)")

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
    
    # Update output directory with model size
    output_dir = output_dir / args.model_size
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
    # Use optimized parameters for Extra Large model to prevent overfitting
    if args.model_size == "xl":
        learning_rate = 5e-4
        weight_decay = 0.1
    else:
        learning_rate = 1e-3
        weight_decay = 0.01

    train_config: TrainConfig = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": 10,
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
    print(f"Using GemiNet Model ({args.model_size})...")

    if args.loss == "poisson":
        loss_cls = "cerberus.loss.PoissonMultinomialLoss"
        loss_args = {"count_weight": args.alpha}
        print(f"Using PoissonMultinomialLoss (count_weight={args.alpha})...")
    else:
        loss_cls = "cerberus.models.bpnet.BPNetLoss"
        loss_args = {"alpha": args.alpha}
        print(f"Using BPNetLoss (alpha={args.alpha})...")

    # Select Model Class based on size
    if args.model_size == "small":
        model_cls = "cerberus.models.geminet.GemiNet"
        model_args = {}
    elif args.model_size == "medium":
        model_cls = "cerberus.models.geminet.GemiNetMedium"
        model_args = {} 
    elif args.model_size == "large":
        model_cls = "cerberus.models.geminet.GemiNetLarge"
        model_args = {} 
    elif args.model_size == "xl":
        model_cls = "cerberus.models.geminet.GemiNetExtraLarge"
        model_args = {}
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")
    
    # Common args
    model_args["input_channels"] = ["A", "C", "G", "T"]
    model_args["output_channels"] = ["signal"]

    model_config: ModelConfig = {
        "name": f"GemiNet-{args.model_size}",
        "model_cls": model_cls,
        "loss_cls": loss_cls,
        "loss_args": loss_args,
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
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

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Training finished. Logs and checkpoints are in subdirectories of {output_dir}")

if __name__ == "__main__":
    main()
