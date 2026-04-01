#!/usr/bin/env python
"""
Multi-task Dalmatian Training Tool.

Trains a multi-task Dalmatian model on multiple BigWig targets (e.g. cell-type
pseudobulk tracks from scATAC-seq).  Target channels are specified via a JSON
file mapping channel names to ``{"bigwig": "path.bw", "peaks": "path.bed"}``
pairs.

With ``--shared-bias``, BiasNet has a single output channel (capturing
sequence-dependent Tn5 insertion bias shared across all cell types) while
SignalNet has one channel per target.

Usage:
    python tools/train_dalmatian_multitask.py \
        --targets-json targets.json \
        --peaks merged_peaks.bed \
        --output-dir models/multitask \
        --shared-bias --pretrained-bias bias_model.pt --freeze-bias
"""

import argparse
import json
import logging
import os
import tempfile
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


def get_args():
    parser = argparse.ArgumentParser(
        description="Train multi-task Dalmatian models from a JSON target specification"
    )

    # --- Multi-task target specification ---
    parser.add_argument(
        "--targets-json",
        type=str,
        required=True,
        help='JSON file mapping channel names to {"bigwig": ..., "peaks": ...} dicts',
    )
    parser.add_argument(
        "--peaks",
        type=str,
        default=None,
        help="Pre-merged peaks BED for the sampler. If omitted, per-channel "
        "peaks from the JSON are concatenated automatically.",
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
        help="Multiplicative scaling factor for targets",
    )
    parser.add_argument(
        "--count-pseudocount",
        type=float,
        default=1.0,
        help="Additive offset before log-transforming count targets",
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
        help="SignalNet preset: 'standard' (f=64, ~150K) or 'large' (f=256, ~3.9M)",
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

    # shared_bias
    parser.add_argument(
        "--shared-bias",
        action="store_true",
        help="Use single-channel BiasNet shared across all output channels",
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
        help="Patience for early stopping",
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


def _merge_peaks(peak_paths: list[str], tmpdir: str) -> str:
    """Concatenate per-channel peak BED files into a merged file."""
    merged_path = os.path.join(tmpdir, "merged_peaks.bed")
    with open(merged_path, "w") as out:
        for path in peak_paths:
            with open(path) as f:
                for line in f:
                    out.write(line)
    return merged_path


def main():
    cerberus.setup_logging()
    logging.info("Starting multi-task Dalmatian training tool...")

    args = get_args()

    if args.freeze_bias and not args.pretrained_bias:
        raise SystemExit("--freeze-bias requires --pretrained-bias")

    # --- Parse targets JSON ---
    with open(args.targets_json) as f:
        targets_spec: dict[str, dict[str, str]] = json.load(f)

    # Sort channel names alphabetically (matches UniversalExtractor ordering)
    channel_names = sorted(targets_spec.keys())
    # Sanitize: replace spaces with underscores
    sanitized_names = [name.replace(" ", "_") for name in channel_names]

    # Build DataConfig targets dict and collect peak paths
    targets_dict: dict[str, str] = {}
    peak_paths: list[str] = []
    for orig_name, sanitized_name in zip(channel_names, sanitized_names, strict=False):
        entry = targets_spec[orig_name]
        targets_dict[sanitized_name] = entry["bigwig"]
        if "peaks" in entry:
            peak_paths.append(entry["peaks"])

    output_channels = sanitized_names
    logging.info(
        f"Multi-task training with {len(output_channels)} channels: {output_channels}"
    )

    # --- Resolve peaks for sampler ---
    tmpdir_obj = None
    if args.peaks:
        peaks_path = args.peaks
    elif peak_paths:
        tmpdir_obj = tempfile.TemporaryDirectory()
        peaks_path = _merge_peaks(peak_paths, tmpdir_obj.name)
        logging.info(f"Merged {len(peak_paths)} peak files into {peaks_path}")
    else:
        raise SystemExit(
            "No peaks provided: use --peaks or include 'peaks' entries in targets JSON"
        )

    try:
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
                genome_files = download_human_reference(
                    data_dir / "genome", name="hg38"
                )
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
            fold_args={
                "k": 5,
                "val_fold": args.val_fold,
                "test_fold": args.test_fold,
            },
            exclude_intervals=exclude_intervals,
        )

        input_len = args.input_len
        output_len = args.output_len
        output_bin_size = 1
        max_jitter = args.jitter
        target_scale = args.target_scale

        data_config = DataConfig(
            inputs={},
            targets=targets_dict,
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

        padded_size = input_len + 2 * max_jitter
        sampler_config = SamplerConfig(
            sampler_type="peak",
            padded_size=padded_size,
            sampler_args={
                "intervals_path": peaks_path,
                "background_ratio": args.background_ratio,
            },
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
            "output_channels": output_channels,
            "signal_preset": args.signal_preset,
            "bias_args": bias_args,
            "signal_args": signal_args,
            "shared_bias": args.shared_bias,
        }

        if args.base_loss == "poisson":
            base_loss_cls = "cerberus.loss.PoissonMultinomialLoss"
            base_loss_args: dict[str, object] = {"count_per_channel": True}
        else:
            base_loss_cls = "cerberus.loss.MSEMultinomialLoss"
            base_loss_args = {"count_per_channel": True}

        loss_args: dict[str, object] = {
            "base_loss_cls": base_loss_cls,
            "base_loss_args": base_loss_args,
            "bias_weight": args.bias_weight,
            "count_pseudocount": args.count_pseudocount,
        }

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
                    "num_workers=%d may cause instability on MPS. "
                    "Recommend setting --num-workers 0.",
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
                "strategy": "ddp_find_unused_parameters_false"
                if accelerator == "gpu" and isinstance(devices, int) and devices > 1
                else "auto",
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
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()


if __name__ == "__main__":
    main()
