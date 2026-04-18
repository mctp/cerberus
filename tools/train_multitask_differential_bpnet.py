#!/usr/bin/env python
"""
Multitask-Differential BPNet Training Tool.

Two-phase training for comparing chromatin accessibility between two conditions:

  Phase 1 — Multi-task absolute model (MultitaskBPNet + MultitaskBPNetLoss)
    Jointly trains a shared BPNet trunk on both conditions so the shared
    latent representation encodes cross-condition sequence grammar.

  Phase 2 — Differential fine-tuning (DifferentialCountLoss)
    Loads the Phase 1 checkpoint via ``ModelConfig.pretrained`` and fine-tunes
    the count heads on the per-peak log2FC derived inline from the shared
    two-channel targets tensor (no offline precompute, no wrapper dataset).

  Interpretation (optional --interpret)
    Runs DeepLIFTSHAP through AttributionTarget(reduction="delta_log_counts")
    on the test-fold peaks and pipes the result into TF-MoDISco.

Usage:
    python tools/train_multitask_differential_bpnet.py \\
        --bigwig-a LNCAP.rpm.bw      --peaks-a LNCAP-macs2.bed.gz  \\
        --bigwig-b 22Rv1.rpm.bw      --peaks-b 22Rv1-macs2.bed.gz  \\
        --name-a LNCAP --name-b 22Rv1 \\
        --output-dir models/foxa1_differential \\
        --stable --interpret
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import subprocess
import sys
from pathlib import Path
from pprint import pformat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import cerberus
from cerberus import (
    CerberusDataModule,
    setup_logging,
)
from cerberus.config import (
    DataConfig,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.download import download_human_reference
from cerberus.config import GenomeConfig
from cerberus.genome import create_genome_config
from cerberus.interval import Interval, merge_intervals
from cerberus.models.bpnet import MultitaskBPNet
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_alpha(value: str) -> "float | str":
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--alpha must be a float or 'adaptive', got: {value!r}"
        ) from None


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase multitask-differential BPNet training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input data ---
    grp = p.add_argument_group("Input data")
    grp.add_argument("--bigwig-a", required=True, help="Depth-normalised bigwig for condition A")
    grp.add_argument("--peaks-a", required=True, help="Peaks BED/narrowPeak for condition A")
    grp.add_argument("--name-a", default="cond_a", help="Label for condition A")
    grp.add_argument("--bigwig-b", required=True, help="Depth-normalised bigwig for condition B")
    grp.add_argument("--peaks-b", required=True, help="Peaks BED/narrowPeak for condition B")
    grp.add_argument("--name-b", default="cond_b", help="Label for condition B")

    # --- Genome ---
    grp = p.add_argument_group("Genome")
    grp.add_argument("--genome", default="hg38")
    grp.add_argument("--species", default="human")
    grp.add_argument("--fasta", default=None)
    grp.add_argument("--blacklist", default=None)
    grp.add_argument("--gaps", default=None)
    grp.add_argument("--data-dir", default="data", help="Directory for genome downloads")

    # --- Output ---
    grp = p.add_argument_group("Output")
    grp.add_argument("--output-dir", required=True)

    # --- Hardware ---
    grp = p.add_argument_group("Hardware")
    grp.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu", "mps"])
    grp.add_argument("--devices", default="auto")
    grp.add_argument(
        "--precision", default="bf16", choices=["bf16", "mps", "full"],
        help="'bf16': NVIDIA bf16-mixed (default); 'mps': Apple fp16-mixed; 'full': float32"
    )
    grp.add_argument("--num-workers", type=int, default=8)
    grp.add_argument("--seed", type=int, default=42)
    grp.add_argument("--silent", action="store_true")

    # --- Fold / cross-validation ---
    grp = p.add_argument_group("Cross-validation")
    grp.add_argument("--multi", action="store_true", help="Multi-fold CV")
    grp.add_argument("--val-fold", type=int, default=1)
    grp.add_argument("--test-fold", type=int, default=0)

    # --- Architecture ---
    grp = p.add_argument_group("Model architecture")
    grp.add_argument("--input-len", type=int, default=2114)
    grp.add_argument("--output-len", type=int, default=1000)
    grp.add_argument("--filters", type=int, default=64)
    grp.add_argument("--n-layers", type=int, default=8)
    grp.add_argument(
        "--stable", action="store_true",
        help="weight_norm + GELU + AdamW/cosine — required for reliable DeepLIFTSHAP"
    )
    grp.add_argument(
        "--residual-architecture", default="residual_pre-activation_conv",
        choices=[
            "residual_post-activation_conv",
            "residual_pre-activation_conv",
            "activated_residual_pre-activation_conv",
        ],
    )

    # --- Phase 1 hyperparameters ---
    grp = p.add_argument_group("Phase 1 (MultitaskBPNet)")
    grp.add_argument("--jitter", type=int, default=256)
    grp.add_argument("--alpha", type=_parse_alpha, default="adaptive",
                     help="Count loss weight: float or 'adaptive'")
    grp.add_argument("--count-pseudocount", type=float, default=150.0,
                     help="Pseudocount for count head log-transform (raw signal units)")
    grp.add_argument("--target-scale", type=float, default=1.0)
    grp.add_argument("--batch-size", type=int, default=64)
    grp.add_argument("--phase1-epochs", type=int, default=50)
    grp.add_argument("--phase1-lr", type=float, default=1e-3)
    grp.add_argument("--weight-decay", type=float, default=0.0)
    grp.add_argument("--patience", type=int, default=10)
    grp.add_argument("--optimizer", default="adam")
    grp.add_argument("--scheduler-type", default="default")
    grp.add_argument("--warmup-epochs", type=int, default=0)
    grp.add_argument("--min-lr", type=float, default=1e-6)

    # --- Phase 2 hyperparameters ---
    #
    # Phase 2 reuses Phase 1's pseudocount (``--count-pseudocount *
    # --target-scale``) so both phases derive log-counts in the same linear-to-
    # log space. The pseudocount is on *linear scale* (added to length-summed
    # signal before taking ``log2``), not a log-scale offset.
    grp = p.add_argument_group("Phase 2 (DifferentialCountLoss)")
    grp.add_argument("--phase2-epochs", type=int, default=20)
    grp.add_argument("--phase2-lr", type=float, default=1e-4)
    grp.add_argument("--phase2-batch-size", type=int, default=64)
    grp.add_argument("--phase2-patience", type=int, default=7)

    # --- Workflow control ---
    grp = p.add_argument_group("Workflow control")
    grp.add_argument("--skip-phase1", action="store_true",
                     help="Skip Phase 1; load existing model from --phase1-dir")
    grp.add_argument("--phase1-dir", default=None,
                     help="Path to existing Phase 1 fold directory (used with --skip-phase1)")
    grp.add_argument("--skip-phase2", action="store_true")

    # --- Interpretation ---
    grp = p.add_argument_group("Interpretation (DeepLIFTSHAP + TF-MoDISco)")
    grp.add_argument("--interpret", action="store_true")
    grp.add_argument("--n-interp", type=int, default=2000,
                     help="Number of test peaks for attribution")
    grp.add_argument("--interp-batch-size", type=int, default=32)
    grp.add_argument("--dls-n-baselines", type=int, default=20,
                     help="Number of dinucleotide-shuffled baselines per sequence")
    grp.add_argument("--meme-db", default=None,
                     help="MEME database for TF-MoDISco motif matching")
    grp.add_argument("--modisco-window", type=int, default=400)
    grp.add_argument("--max-seqlets", type=int, default=2000)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# This script accepts two peak files (one per condition), but the existing
# Cerberus peak sampler API expects a single ``intervals_path``.  Single-task
# training tools can pass their one peak file straight through to the sampler;
# the differential workflow cannot.  We therefore read the raw intervals from
# both inputs, merge them into a union peak set, and write a temporary BED that
# both Phase 1 and Phase 2 can sample from consistently.


def _read_bed_intervals(path: "str | Path") -> list[Interval]:
    """Read chrom/start/end from a BED or narrowPeak file (plain or .gz)."""
    path = Path(path)
    open_fn = gzip.open if path.name.endswith(".gz") else open
    intervals: list[Interval] = []
    with open_fn(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            cols = line.split("\t")
            if len(cols) >= 3:
                try:
                    intervals.append(Interval(cols[0], int(cols[1]), int(cols[2])))
                except ValueError:
                    continue
    return intervals


def _merge_and_write_peaks(peaks_a: Path, peaks_b: Path, out_bed: Path) -> Path:
    """Merge peaks from both conditions and write a 3-column BED file."""
    ivs_a = _read_bed_intervals(peaks_a)
    ivs_b = _read_bed_intervals(peaks_b)
    merged = merge_intervals(ivs_a + ivs_b)
    with open(out_bed, "w") as fh:
        for iv in merged:
            fh.write(f"{iv.chrom}\t{iv.start}\t{iv.end}\n")
    logger.info(
        "Merged %d + %d peaks → %d non-overlapping regions → %s",
        len(ivs_a), len(ivs_b), len(merged), out_bed,
    )
    return out_bed


def _find_phase1_checkpoint(phase1_dir: Path) -> Path:
    """Find the ``model.pt`` written by ``train_single`` / ``train_multi``."""
    candidates = sorted(phase1_dir.glob("**/model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {phase1_dir}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def run_phase1(args: argparse.Namespace, merged_peaks: Path, output_dir: Path,
               genome_config: GenomeConfig, data_dir: Path) -> Path:
    """Train Phase 1 MultitaskBPNet.  Returns the fold directory."""
    phase1_dir = output_dir / "phase1"
    if args.multi:
        phase1_dir = phase1_dir / "multi-fold"
    else:
        phase1_dir = phase1_dir / "single-fold"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    # Stable-mode overrides (mirroring train_bpnet.py convention)
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    scheduler_type = args.scheduler_type
    warmup_epochs = args.warmup_epochs
    min_lr = args.min_lr
    if args.stable:
        if optimizer == "adam":
            optimizer = "adamw"
        if weight_decay == 0.0:
            weight_decay = 0.01
        if scheduler_type == "default":
            scheduler_type = "cosine"
        if warmup_epochs == 0:
            warmup_epochs = 5

    input_len = args.input_len
    output_len = args.output_len
    max_jitter = args.jitter
    padded_size = input_len + 2 * max_jitter

    data_config = DataConfig(
        inputs={},
        targets={
            args.name_a: args.bigwig_a,
            args.name_b: args.bigwig_b,
        },
        input_len=input_len,
        output_len=output_len,
        max_jitter=max_jitter,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=True,
        use_sequence=True,
        target_scale=args.target_scale,
    )

    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=padded_size,
        sampler_args={
            "intervals_path": str(merged_peaks),
            "background_ratio": 0.0,
        },
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.phase1_epochs,
        learning_rate=args.phase1_lr,
        weight_decay=weight_decay,
        patience=args.patience,
        optimizer=optimizer,
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        scheduler_type=scheduler_type,
        scheduler_args={
            "num_epochs": args.phase1_epochs,
            "warmup_epochs": warmup_epochs,
            "min_lr": min_lr,
        },
        adam_eps=1e-7,
        gradient_clip_val=None,
    )

    activation = "gelu" if args.stable else "relu"
    model_args: dict = {
        "output_channels": [args.name_a, args.name_b],
        "filters": args.filters,
        "n_dilated_layers": args.n_layers,
        "conv_kernel_size": 21,
        "dil_kernel_size": 3,
        "profile_kernel_size": 75,
        "activation": activation,
        "weight_norm": args.stable,
        "residual_architecture": args.residual_architecture,
    }

    model_config = ModelConfig(
        name="MultitaskBPNet",
        model_cls="cerberus.models.bpnet.MultitaskBPNet",
        loss_cls="cerberus.models.bpnet.MultitaskBPNetLoss",
        loss_args={"alpha": args.alpha, "beta": 1.0},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=[],
        count_pseudocount=args.count_pseudocount * args.target_scale,
    )

    # Parse devices
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass

    accelerator = args.accelerator
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    precision_kwargs = get_precision_kwargs(args.precision, accelerator, devices)

    logger.info("Phase 1 config:\n%s", pformat(model_config))

    train_fn = train_multi if args.multi else train_single
    train_fn(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
        train_config=train_config,
        num_workers=args.num_workers,
        in_memory=False,
        root_dir=str(phase1_dir),
        enable_checkpointing=True,
        log_every_n_steps=10,
        val_batch_size=args.batch_size * 4,
        enable_progress_bar=not args.silent,
        seed=args.seed,
        **precision_kwargs,
    )

    return phase1_dir


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _plot_phase2_losses(phase2_dir: Path) -> None:
    """Plot Phase 2 train/val delta loss from the CSVLogger metrics file.

    The standard ``train_single`` / ``train_multi`` path logs metrics via a
    CSVLogger under ``<phase2_dir>/<name>/version_<n>/metrics.csv``. We pick
    the newest ``metrics.csv`` under ``phase2_dir`` to stay agnostic to the
    exact nested layout (single-fold vs. multi-fold).
    """
    metrics_paths = sorted(
        phase2_dir.glob("**/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not metrics_paths:
        logger.warning("Phase 2 metrics not found under %s — skipping plot.", phase2_dir)
        return
    csv_path = metrics_paths[0]
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, label in [
        (axes[0], "train_delta_loss", "Train delta loss"),
        (axes[1], "val_delta_loss",   "Val delta loss"),
    ]:
        if col in df.columns:
            vals = df[col].dropna()
            steps = df.loc[vals.index, "step"] if "step" in df.columns else vals.index
            ax.plot(steps, vals, lw=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("MSE delta loss")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    fig.suptitle("Phase 2: Differential fine-tuning losses", fontsize=12)
    fig.tight_layout()

    plots_dir = phase2_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "phase2_losses.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Phase 2 loss plot saved to %s", out_path)


def _select_phase2_strategy(precision_kwargs: dict) -> dict:
    """Phase 2 supervises only the count-head delta — the profile heads
    receive no gradient. DDP therefore needs ``find_unused_parameters=True``;
    override the single-GPU default from :func:`get_precision_kwargs`."""
    strategy = precision_kwargs.get("strategy")
    if strategy == "ddp_find_unused_parameters_false":
        return {**precision_kwargs, "strategy": "ddp_find_unused_parameters_true"}
    return precision_kwargs


def run_phase2(
    args: argparse.Namespace,
    phase1_model_path: Path,
    merged_peaks: Path,
    output_dir: Path,
    genome_config: GenomeConfig,
) -> Path:
    """Fine-tune with DifferentialCountLoss via the standard cerberus pipeline.

    Phase 2 is a regular ``train_single`` / ``train_multi`` call with a
    :class:`DifferentialCountLoss` ``ModelConfig`` that lists the Phase 1
    checkpoint under ``pretrained=[PretrainedConfig(...)]``. The per-peak
    log2FC target is derived inline inside the loss from the two-channel
    ``(B, 2, L)`` targets tensor Phase 1 already supervised against — no
    offline precompute, no dataset wrapper, no bespoke PL module.
    """
    phase2_dir = output_dir / "phase2"
    if args.multi:
        phase2_dir = phase2_dir / "multi-fold"
    else:
        phase2_dir = phase2_dir / "single-fold"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 pseudocount in scaled units (raw × target_scale) — keeps Phase 1
    # and Phase 2 in the same log-space.
    count_pseudocount_scaled = args.count_pseudocount * args.target_scale

    # 1. Phase 2 data config: no jitter, peaks only, same bigwigs as Phase 1.
    data_config = DataConfig(
        inputs={},
        targets={
            args.name_a: args.bigwig_a,
            args.name_b: args.bigwig_b,
        },
        input_len=args.input_len,
        output_len=args.output_len,
        max_jitter=0,          # no jitter — stable log2FC targets
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=True,
        use_sequence=True,
        target_scale=args.target_scale,
    )
    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=args.input_len,   # no padding needed (jitter=0)
        sampler_args={
            "intervals_path": str(merged_peaks),
            "background_ratio": 0.0,  # peaks only
        },
    )

    # 2. Phase 2 model: same architecture as Phase 1, loaded from the Phase 1
    #    checkpoint via ``pretrained=[PretrainedConfig(...)]``.
    activation = "gelu" if args.stable else "relu"
    model_args: dict = {
        "output_channels": [args.name_a, args.name_b],
        "filters": args.filters,
        "n_dilated_layers": args.n_layers,
        "conv_kernel_size": 21,
        "dil_kernel_size": 3,
        "profile_kernel_size": 75,
        "activation": activation,
        "weight_norm": args.stable,
        "residual_architecture": args.residual_architecture,
    }

    model_config = ModelConfig(
        name="MultitaskBPNet_Phase2",
        model_cls="cerberus.models.bpnet.MultitaskBPNet",
        loss_cls="cerberus.loss.DifferentialCountLoss",
        loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args=model_args,
        pretrained=[
            PretrainedConfig(
                weights_path=str(phase1_model_path),
                source=None,
                target=None,
                freeze=False,
            )
        ],
        count_pseudocount=count_pseudocount_scaled,
    )

    # 3. Phase 2 training hyperparameters.
    optimizer = "adamw" if args.stable else args.optimizer
    weight_decay = 0.01 if args.stable else args.weight_decay
    scheduler_type = "cosine" if args.stable else args.scheduler_type
    warmup_epochs = args.warmup_epochs if args.warmup_epochs > 0 else (
        5 if args.stable else 0
    )

    train_config = TrainConfig(
        batch_size=args.phase2_batch_size,
        max_epochs=args.phase2_epochs,
        learning_rate=args.phase2_lr,
        weight_decay=weight_decay,
        patience=args.phase2_patience,
        optimizer=optimizer,
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        scheduler_type=scheduler_type,
        scheduler_args={
            "num_epochs": args.phase2_epochs,
            "warmup_epochs": warmup_epochs,
            "min_lr": args.min_lr,
        },
        adam_eps=1e-7,
        gradient_clip_val=None,
    )

    # 4. Parse devices / accelerator / precision.
    devices = args.devices
    if devices != "auto":
        try:
            devices = int(devices)
        except ValueError:
            pass
    accelerator = args.accelerator
    if accelerator == "auto" and torch.backends.mps.is_available():
        accelerator = "mps"

    precision_kwargs = get_precision_kwargs(args.precision, accelerator, devices)
    precision_kwargs = _select_phase2_strategy(precision_kwargs)

    logger.info("Phase 2 config:\n%s", pformat(model_config))

    train_fn = train_multi if args.multi else train_single
    train_fn(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        model_config=model_config,
        train_config=train_config,
        num_workers=args.num_workers,
        in_memory=False,
        root_dir=str(phase2_dir),
        enable_checkpointing=True,
        log_every_n_steps=10,
        val_batch_size=args.phase2_batch_size * 4,
        enable_progress_bar=not args.silent,
        seed=args.seed,
        **precision_kwargs,
    )

    # 5. Plot losses (rank 0 only — avoid NFS read races).
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        _plot_phase2_losses(phase2_dir)

    return _find_phase1_checkpoint(phase2_dir)


# ---------------------------------------------------------------------------
# Interpretation: DeepLIFTSHAP + TF-MoDISco
# ---------------------------------------------------------------------------


def _dinuc_shuffle(seq_1hot: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Dinucleotide-shuffle a (4, L) one-hot array *n* times.

    Returns (n, 4, L) shuffled sequences.
    """
    L = seq_1hot.shape[1]
    # Decode to string
    bases = "ACGT"
    seq_str = "".join(bases[seq_1hot[:, i].argmax()] for i in range(L))

    shuffled = np.zeros((n, 4, L), dtype=np.float32)
    for k in range(n):
        # Collect dinucleotide pairs and shuffle their order
        pairs = [seq_str[i : i + 2] for i in range(0, L - 1, 2)]
        rng.shuffle(pairs)
        shuffled_str = "".join(pairs)
        if len(shuffled_str) < L:
            shuffled_str += seq_str[-1]  # append last base if odd length
        for i, b in enumerate(shuffled_str[:L]):
            shuffled[k, bases.index(b), i] = 1.0
    return shuffled


def run_interpretation(
    args: argparse.Namespace,
    phase2_model_path: Path,
    merged_peaks: Path,
    output_dir: Path,
    genome_config: GenomeConfig,
) -> None:
    """Compute DeepLIFTSHAP attributions and run TF-MoDISco."""
    try:
        from captum.attr import DeepLiftShap
    except ImportError:
        logger.error(
            "captum is not installed. Run: pip install captum\n"
            "Skipping interpretation."
        )
        return

    from cerberus.attribution import AttributionTarget

    interp_dir = output_dir / "interpretation"
    interp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Interpretation device: %s", device)

    # 1. Load Phase 2 model
    activation = "gelu" if args.stable else "relu"
    model = MultitaskBPNet(
        output_channels=[args.name_a, args.name_b],
        input_len=args.input_len,
        output_len=args.output_len,
        filters=args.filters,
        n_dilated_layers=args.n_layers,
        conv_kernel_size=21,
        dil_kernel_size=3,
        profile_kernel_size=75,
        activation=activation,
        weight_norm=args.stable,
        residual_architecture=args.residual_architecture,
    )
    state_dict = torch.load(phase2_model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 2. AttributionTarget(delta_log_counts): outputs log_counts_B − log_counts_A
    diff_target = AttributionTarget(
        model=model,
        reduction="delta_log_counts",
        channels=(0, 1),
    )
    diff_target.eval().to(device)

    # 3. DeepLiftShap
    dls = DeepLiftShap(diff_target)

    # 4. Load test-set sequences via a CerberusDataModule (test fold)
    data_config = DataConfig(
        inputs={},
        targets={
            args.name_a: args.bigwig_a,
            args.name_b: args.bigwig_b,
        },
        input_len=args.input_len,
        output_len=args.output_len,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,   # no RC augmentation for attribution
        use_sequence=True,
        target_scale=args.target_scale,
    )
    sampler_config = SamplerConfig(
        sampler_type="peak",
        padded_size=args.input_len,
        sampler_args={
            "intervals_path": str(merged_peaks),
            "background_ratio": 0.0,
        },
    )
    datamodule = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        seed=args.seed,
    )
    datamodule.batch_size = args.interp_batch_size
    datamodule.num_workers = args.num_workers
    datamodule.prepare_data()
    datamodule.setup("test")

    # 5. Collect attributions
    rng = np.random.default_rng(args.seed)
    all_ohe: list[np.ndarray] = []
    all_attr: list[np.ndarray] = []
    n_done = 0

    logger.info("Computing DeepLIFTSHAP for up to %d test peaks…", args.n_interp)
    for batch in datamodule.test_dataloader():
        if n_done >= args.n_interp:
            break

        inputs = batch["inputs"].to(device)  # (B, 4, L)
        ohe = inputs[:, :4, :]               # one-hot part only

        batch_attrs: list[torch.Tensor] = []
        for i in range(ohe.shape[0]):
            if n_done >= args.n_interp:
                break
            seq_np = ohe[i].cpu().numpy()           # (4, L)
            baselines_np = _dinuc_shuffle(seq_np, args.dls_n_baselines, rng)  # (N, 4, L)

            seq_t = ohe[i : i + 1].expand(args.dls_n_baselines, -1, -1).to(device)  # (N, 4, L)
            baselines_t = torch.from_numpy(baselines_np).to(device)

            with torch.enable_grad():
                attr = dls.attribute(seq_t, baselines=baselines_t)  # (N, 4, L)

            attr_mean = attr.mean(dim=0, keepdim=True)  # (1, 4, L)
            batch_attrs.append(attr_mean)
            all_ohe.append(seq_np[np.newaxis])          # (1, 4, L)
            n_done += 1

        if batch_attrs:
            all_attr.append(torch.cat(batch_attrs, dim=0).detach().cpu().numpy())

    if not all_ohe:
        logger.warning("No sequences collected for interpretation — skipping.")
        return

    ohe_arr = np.concatenate(all_ohe, axis=0)   # (N, 4, L)
    attr_arr = np.concatenate(all_attr, axis=0)  # (N, 4, L)

    ohe_path = interp_dir / "ohe.npz"
    attr_path = interp_dir / "attr.npz"
    np.savez(ohe_path, arr_0=ohe_arr)
    np.savez(attr_path, arr_0=attr_arr)
    logger.info("Saved OHE (%s) and attributions (%s)", ohe_path, attr_path)

    # 6. Run TF-MoDISco
    modisco_out = interp_dir / "modisco_results.h5"
    report_dir = interp_dir / "report"
    tfmodisco_script = Path(__file__).parent / "run_tfmodisco.py"

    cmd = [
        sys.executable, str(tfmodisco_script),
        "--ohe-path", str(ohe_path),
        "--attr-path", str(attr_path),
        "--modisco-output", str(modisco_out),
        "--modisco-window", str(args.modisco_window),
        "--max-seqlets", str(args.max_seqlets),
        "--run-report",
        "--report-dir", str(report_dir),
    ]
    if args.meme_db:
        cmd += ["--meme-db", str(args.meme_db)]

    logger.info("Running TF-MoDISco: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("TF-MoDISco exited with code %d", result.returncode)
    else:
        logger.info("TF-MoDISco complete. Report at %s", report_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    args = get_args()

    logger.info("=== Multitask-Differential BPNet ===")
    logger.info("Condition A: %s  bw=%s  peaks=%s", args.name_a, args.bigwig_a, args.peaks_a)
    logger.info("Condition B: %s  bw=%s  peaks=%s", args.name_b, args.bigwig_b, args.peaks_b)

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Genome reference ---
    if not args.fasta:
        if args.genome == "hg38":
            genome_files = download_human_reference(data_dir / "genome", name="hg38")
            fasta_path = genome_files["fasta"]
            blacklist_path = args.blacklist or genome_files.get("blacklist")
            gaps_path = args.gaps or genome_files.get("gaps")
        else:
            raise ValueError(f"--fasta required for genome {args.genome!r}")
    else:
        fasta_path = Path(args.fasta)
        blacklist_path = Path(args.blacklist) if args.blacklist else None
        gaps_path = Path(args.gaps) if args.gaps else None

    exclude_intervals: dict = {}
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

    # --- Merge peaks ---
    merged_bed = output_dir / "merged_peaks.bed"
    if not merged_bed.exists():
        _merge_and_write_peaks(Path(args.peaks_a), Path(args.peaks_b), merged_bed)
    else:
        logger.info("Using existing merged peaks: %s", merged_bed)

    # --- Phase 1 ---
    if args.skip_phase1:
        if not args.phase1_dir:
            raise ValueError("--phase1-dir required when --skip-phase1 is set")
        phase1_dir = Path(args.phase1_dir)
    else:
        phase1_dir = run_phase1(args, merged_bed, output_dir, genome_config, data_dir)

    phase1_model_path = _find_phase1_checkpoint(phase1_dir)
    logger.info("Phase 1 model: %s", phase1_model_path)

    # --- Phase 2 ---
    phase2_root = output_dir / "phase2" / ("multi-fold" if args.multi else "single-fold")
    if not args.skip_phase2:
        phase2_model_path = run_phase2(
            args, phase1_model_path, merged_bed, output_dir, genome_config
        )
    else:
        if not phase2_root.exists():
            raise FileNotFoundError(
                f"--skip-phase2 set but {phase2_root} not found"
            )
        phase2_model_path = _find_phase1_checkpoint(phase2_root)
        logger.info("Skipping Phase 2; using existing %s", phase2_model_path)

    # --- Interpretation ---
    # DDP re-runs main() on every rank after trainer.fit().  Guard single-rank
    # work (attribution + TF-MoDISco) to LOCAL_RANK 0 to avoid NFS lock races.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.interpret and local_rank == 0:
        run_interpretation(
            args, phase2_model_path, merged_bed, output_dir, genome_config
        )

    logger.info("=== Pipeline complete. Output: %s ===", output_dir)


if __name__ == "__main__":
    main()
