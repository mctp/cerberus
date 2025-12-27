#!/usr/bin/env python
"""
Full-scale training example for a CNN model using Cerberus.

This script demonstrates how to train a model with production-ready parameters,
including CLI argument parsing, logging, and checkpointing.

Usage:
    python examples/train_full_scale.py --help
    python examples/train_full_scale.py --batch-size 64 --max-epochs 50
"""

import argparse
import os
from pathlib import Path
from pprint import pprint
import torch

# Cerberus imports
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig
from cerberus.genome import create_genome_config
from cerberus.datamodule import CerberusDataModule
from cerberus.models.baseline_gopher import GlobalProfileCNN
from cerberus.loss import get_default_loss, get_default_metrics
from cerberus.module import CerberusModule
from cerberus.entrypoints import train

def get_args():
    parser = argparse.ArgumentParser(description="Train a CNN model with Cerberus")
    
    # Script arguments
    parser.add_argument("--data-dir", type=str, default="tests/data", help="Directory to store/load data")
    parser.add_argument("--output-dir", type=str, default="training_logs", help="Directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
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
    input_len = 2048
    output_len = 1024
    
    data_config: DataConfig = {
        "inputs": {}, # No additional input tracks, just DNA
        "targets": {"signal": dataset_files["bigwig"]},
        "input_len": input_len,
        "output_len": output_len, 
        "max_jitter": 128,  # Augmentation jitter
        "output_bin_size": 4, # Binning resolution
        "encoding": "ACGT", # Standard One-Hot
        "log_transform": True, # Log(x+1) transform targets
        "reverse_complement": True, # Augmentation
        "use_sequence": True,
    }

    # Sampler Config
    # padded_size >= input_len + 2 * max_jitter
    padded_size = input_len + 2 * data_config["max_jitter"]
    sampler_config: SamplerConfig = {
        "sampler_type": "interval",
        "padded_size": padded_size,
        "sampler_args": {
            "intervals_path": dataset_files["narrowPeak"]
        }
    }

    # Train Config
    max_epochs = 100
    
    train_config: TrainConfig = {
        "batch_size": 128,
        "max_epochs": max_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 10,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {
            "num_epochs": max_epochs,
            "warmup_epochs": 5, # Warmup for 5 epochs
            "min_lr": 1e-5
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

    # 3. Initialize DataModule
    datamodule = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
    )

    # 4. Model Setup
    model = GlobalProfileCNN(
        input_len=input_len, 
        output_len=output_len, 
        output_bin_size=data_config["output_bin_size"]
    )

    criterion = get_default_loss()
    metrics = get_default_metrics(num_channels=1)

    module = CerberusModule(
        model=model,
        train_config=train_config,
        criterion=criterion,
        metrics=metrics
    )

    # 5. Training
    # Handle devices argument
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            # If it's a list like "0,1", keep as string or parse list if needed by PL
            pass

    trainer = train(
        module=module,
        datamodule=datamodule,
        train_config=train_config,
        num_workers=args.num_workers,
        in_memory=False, # Full scale usually implies too big for memory
        # Trainer kwargs
        accelerator=args.accelerator,
        devices=devices,
        default_root_dir=str(output_dir),
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=10,
        strategy="ddp" if args.accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto"
    )

    print(f"Training finished. Logs and checkpoints are in {output_dir}")

if __name__ == "__main__":
    main()
