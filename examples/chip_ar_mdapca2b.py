#!/usr/bin/env python
"""
Full-scale training example for a CNN model using Cerberus.

This script demonstrates how to train a model using the high-level configuration-driven API.
It supports both single-fold training (default) and multi-fold cross-validation.

Usage:
    python examples/chip_ar_mdapca2b.py --help
    python examples/chip_ar_mdapca2b.py --batch-size 64 --max-epochs 50
    python examples/chip_ar_mdapca2b.py --multi --batch-size 64

    # Apple Silicon (M1/M2/M3) Recommendation:
    python examples/chip_ar_mdapca2b.py --accelerator mps --batch-size 64 --num-workers 0
"""

import argparse
import torch
from pathlib import Path
from pprint import pprint
import torch.nn as nn
from typing import cast
from torchmetrics import MetricCollection

# Cerberus imports
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.models.gopher import GlobalProfileCNN
from cerberus.models.bpnet import BPNet, BPNetMetricCollection, BPNetLoss
from cerberus.metrics import DefaultMetricCollection
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.entrypoints import train_fold, train_multi

def get_args():
    parser = argparse.ArgumentParser(description="Train a CNN model with Cerberus")
    
    # Script arguments
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory to store/load data")
    parser.add_argument("--output-dir", type=str, default="tests/data/models/chip_ar_mdapca2b", help="Root directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum number of epochs")
    
    # Mode arguments
    parser.add_argument("--multi", action="store_true", help="Run multi-fold cross-validation instead of single fold")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "bpnet"], help="Model architecture to use")

    # Sampler arguments
    parser.add_argument("--genome", action="store_true", help="Use genome-wide sliding window sampler instead of peak intervals")

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
    if args.multi and args.genome:
        output_dir = output_dir / "genome-multi"
    elif args.multi:
        output_dir = output_dir / "peaks-multi"
    elif args.genome:
        output_dir = output_dir / "genome-single"
    else:
        output_dir = output_dir / "peaks-single"
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

    # Data Config
    # Input: 2048bp DNA
    # Output: 256 bins @ 4bp resolution (covering 1024bp)
    
    if args.model == "bpnet":
        # BPNet standard params
        input_len = 2114
        output_len = 1000
        output_bin_size = 1 # BPNet is usually base-resolution
        log_transform = False # BPNet typically trains on raw counts with PoissonMultinomial/MSEMultinomialLoss
    else:
        # Baseline Gopher params
        input_len = 2048
        output_len = 1024
        output_bin_size = 4
        log_transform = True

    data_config: DataConfig = {
        "inputs": {}, # No additional input tracks, just DNA
        "targets": {"signal": dataset_files["bigwig"]},
        "input_len": input_len,
        "output_len": output_len, 
        "max_jitter": 128,  # Augmentation jitter
        "output_bin_size": output_bin_size, # Binning resolution
        "encoding": "ACGT", # Standard One-Hot
        "log_transform": log_transform, # Log(x+1) transform targets
        "reverse_complement": True, # Augmentation
        "use_sequence": True,
    }

    # Sampler Config
    # padded_size >= input_len + 2 * max_jitter
    padded_size = input_len + 2 * data_config["max_jitter"]
    
    if args.genome:
        print("Using Genome Sampler (Sliding Window)...")
        sampler_config: SamplerConfig = {
            "sampler_type": "sliding_window",
            "padded_size": padded_size,
            "sampler_args": {
                "stride": 1024
            }
        }
    else:
        print("Using Peak Sampler (Intervals)...")
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
        "patience": 10,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {
            "num_epochs": args.max_epochs,
            "warmup_epochs": 5, # Warmup for 5 epochs
            "min_lr": 1e-5
        }
    }

    # Model Config
    if args.model == "bpnet":
        print("Using BPNet Model...")
        model_config: ModelConfig = {
            "name": "BPNet",
            "model_cls": BPNet,
            "loss_cls": BPNetLoss,
            "loss_args": {
                "alpha": 1.0,
                "implicit_log_targets": log_transform # Should be False if data not transformed
            },
            "metrics_cls": BPNetMetricCollection,
            "metrics_args": {"num_channels": 1},
            "model_args": {
                "input_channels": ["A", "C", "G", "T"],
                "output_channels": ["signal"],
                "filters": 64,
                "n_dilated_layers": 8
            }
        }
    else:
        print("Using Baseline (GlobalProfileCNN) Model...")
        model_config: ModelConfig = {
            "name": "GlobalProfileCNN",
            "model_cls": GlobalProfileCNN,
            "loss_cls": ProfilePoissonNLLLoss,
            "loss_args": {"log_input": True, "full": False},
            # Cast function to expected type for static analysis
            "metrics_cls": DefaultMetricCollection,
            "metrics_args": {"num_channels": 1},
            "model_args": {
                "input_channels": ["A", "C", "G", "T"],
                "output_channels": ["signal"],
            }
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
    print("-" * 20 + "\n")

    # 3. Training
    # Handle devices argument
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            # If it's a list like "0,1", keep as string or parse list if needed by PL
            pass

    # Hardware-specific optimization
    accelerator = args.accelerator
    num_workers = args.num_workers

    # Auto-detect Apple Silicon (MPS)
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    if accelerator == "mps":
        print(f"[INFO] Using Apple Silicon (MPS) acceleration.")
        # MPS is often unstable with multiprocessing in DataLoaders
        if num_workers > 0:
            print(f"[WARN] num_workers={num_workers} may cause instability on MPS. Recommend setting --num-workers 0.")
    
    # Precision settings
    precision_args = {
        "precision": "16-mixed", # Mixed precision works well on M2
        "matmul_precision": "high", # Mainly for NVIDIA Tensor Cores
        "accelerator": accelerator,
        "devices": devices,
        "strategy": "ddp" if accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto"
    }

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
            **precision_args
        )
    else:
        print("Starting Single Fold Training (train_fold)...")
        train_fold(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            model_config=model_config,
            train_config=train_config,
            test_fold=0, # Default fold
            val_fold=1, # Default fold
            num_workers=args.num_workers,
            in_memory=False,
            root_dir=str(output_dir),
            enable_checkpointing=True,
            log_every_n_steps=10,
            **precision_args
        )

    print(f"Training finished. Logs and checkpoints are in {output_dir}")

if __name__ == "__main__":
    main()
