#!/usr/bin/env python
"""Train a multi-task ChromBPNet with a parallel differential objective.

This is the from-scratch counterpart to ``train_chrombpnet_multitask.py``:
it keeps the usual absolute profile/count objective for every task and adds
a delta-log-count loss between two task channels.  The phase-2-only
fine-tuning workflow remains in ``train_chrombpnet_multitask_differential.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from pprint import pformat

import torch

import cerberus
from cerberus.config import (
    DataConfig,
    FreezeSpec,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.download import download_human_reference
from cerberus.genome import create_genome_config
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from _pseudocount_cli import (  # noqa: E402
    add_pseudocount_cli_args,
    resolve_count_pseudocount_from_args,
)
from train_chrombpnet_multitask import (  # noqa: E402
    _export_accessibility_checkpoints,
    _load_targets_json,
    _merge_peaks,
    _parse_alpha,
    _plot_training_curves,
)

logger = logging.getLogger(__name__)


def _parse_devices(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-task ChromBPNet with absolute profile/count loss "
            "plus a parallel differential log-count loss."
        ),
    )

    # --- Input data ---
    parser.add_argument(
        "--targets-json",
        type=str,
        required=True,
        help='JSON mapping task names to {"bigwig": ..., "peaks": ...} entries.',
    )
    parser.add_argument(
        "--peaks",
        type=str,
        default=None,
        help="Merged peak BED/narrowPeak for sampling. If omitted, per-task "
        "peaks from --targets-json are concatenated.",
    )

    # --- Genome / reference ---
    parser.add_argument("--genome", type=str, default="hg38")
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--fasta", type=str)
    parser.add_argument("--blacklist", type=str)
    parser.add_argument("--gaps", type=str)

    # --- I/O & runtime ---
    parser.add_argument("--data-dir", type=str, default="tests/data")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--silent", action="store_true")

    # --- Cross-validation ---
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--test-fold", type=int, default=0)

    # --- Data config ---
    parser.add_argument("--input-len", type=int, default=2114)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--jitter", type=int, default=256)
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=1.0,
        help="Non-peak to peak ratio during training",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling factor for already-normalised targets "
        "(typically 1.0 when targets came through scatac_normalize_pseudobulk).",
    )

    # --- Pseudocount CLI family (shared with the other train_* tools) ---
    add_pseudocount_cli_args(parser, default_count_pseudocount=1.0)

    # --- Pretrained bias / accessibility branches ---
    parser.add_argument(
        "--pretrained-bias",
        type=str,
        required=True,
        help="Path to the stage-1 single-channel ChromBPNet bias BPNet checkpoint "
        "(e.g. produced by tools/train_chrombpnet_bias.py).",
    )
    parser.add_argument(
        "--pretrained-bias-source",
        type=str,
        default=None,
        help="Optional prefix to extract from the bias checkpoint "
        "(e.g. 'bias_model' when loading from a full ChromBPNet checkpoint).",
    )
    parser.add_argument(
        "--pretrained-accessibility",
        type=str,
        default=None,
        help="Optional checkpoint to warm-start the multi-task accessibility branch.",
    )
    parser.add_argument(
        "--pretrained-accessibility-source",
        type=str,
        default=None,
        help="Optional prefix to extract from the accessibility checkpoint.",
    )
    parser.add_argument(
        "--no-freeze-bias",
        action="store_true",
        help="Allow the loaded shared bias branch to keep training. Default behaviour "
        "adds FreezeSpec(pattern='bias_model', eval_mode=True) to ModelConfig.freeze "
        "so the bias broadcast contributes a fixed Tn5-bias baseline.",
    )
    parser.add_argument(
        "--bias-logcount-offset",
        type=float,
        default=0.0,
        help="Initial scalar offset applied to the bias log-count head before "
        "logaddexp combination. Use estimate_bias_logcount_offset() to calibrate "
        "from data, or pass an explicit value.",
    )

    # --- Accessibility branch architecture (chrombpnet-pytorch defaults) ---
    parser.add_argument("--filters", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--conv-kernel-size", type=int, default=21)
    parser.add_argument("--dil-kernel-size", type=int, default=3)
    parser.add_argument("--profile-kernel-size", type=int, default=75)

    # --- Bias branch architecture (must match stage-1 trainer) ---
    parser.add_argument("--bias-filters", type=int, default=128)
    parser.add_argument("--bias-layers", type=int, default=4)
    parser.add_argument("--bias-conv-kernel-size", type=int, default=21)
    parser.add_argument("--bias-dil-kernel-size", type=int, default=3)
    parser.add_argument("--bias-profile-kernel-size", type=int, default=75)

    parser.add_argument(
        "--residual-architecture",
        type=str,
        default="residual_post-activation_conv",
        choices=[
            "residual_post-activation_conv",
            "residual_pre-activation_conv",
            "activated_residual_pre-activation_conv",
        ],
    )

    # --- Loss ---
    parser.add_argument("--alpha", type=_parse_alpha, default="adaptive")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--differential-cond-a",
        type=int,
        default=0,
        help="Condition A channel index for the delta-log-count loss.",
    )
    parser.add_argument(
        "--differential-cond-b",
        type=int,
        default=1,
        help="Condition B channel index for the delta-log-count loss.",
    )
    parser.add_argument(
        "--differential-pseudocount",
        type=float,
        default=None,
        help=(
            "Optional pseudocount for the delta-log-count target. Defaults to "
            "--count-pseudocount so absolute and delta targets share the same "
            "count offset unless explicitly separated."
        ),
    )

    # --- Optimisation ---
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler-type", type=str, default="default")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # --- Hardware ---
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
    )
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="full",
        choices=["bf16", "mps", "full"],
        help="Default 'full' (fp32) for ChromBPNet: the bias-count "
        "logaddexp branch is sensitive to bf16 underflow.",
    )

    return parser.parse_args()


def main() -> None:
    cerberus.setup_logging()
    logger.info("Starting parallel differential multi-task ChromBPNet training tool...")
    args = get_args()

    targets_dict, json_peak_paths = _load_targets_json(args.targets_json)
    output_channels = list(targets_dict.keys())
    logger.info(
        "Parallel differential ChromBPNet targets (%d): %s",
        len(output_channels),
        output_channels,
    )

    tmpdir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.peaks:
        peaks_path = args.peaks
    elif json_peak_paths:
        tmpdir_obj = tempfile.TemporaryDirectory()
        peaks_path = _merge_peaks(json_peak_paths, tmpdir_obj.name)
        logger.info(
            "Merged %d per-task peak files into %s",
            len(json_peak_paths),
            peaks_path,
        )
    else:
        raise SystemExit(
            "No peaks provided: use --peaks or include 'peaks' entries in --targets-json"
        )

    try:
        data_dir = Path(args.data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(args.output_dir).resolve()
        output_dir = output_dir / ("multi-fold" if args.multi else "single-fold")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.fasta:
            if args.genome != "hg38":
                raise ValueError(
                    f"Fasta path must be provided for genome {args.genome} if not hg38"
                )
            logger.info("Downloading/Checking Human Reference (hg38)...")
            genome_files = download_human_reference(data_dir / "genome", name="hg38")
            fasta_path = genome_files["fasta"]
            blacklist_path = args.blacklist or genome_files["blacklist"]
            gaps_path = args.gaps or genome_files["gaps"]
        else:
            fasta_path = Path(args.fasta)
            blacklist_path = Path(args.blacklist) if args.blacklist else None
            gaps_path = Path(args.gaps) if args.gaps else None

        exclude_intervals: dict[str, Path] = {}
        if blacklist_path:
            exclude_intervals["blacklist"] = Path(blacklist_path)
        if gaps_path:
            exclude_intervals["gaps"] = Path(gaps_path)

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
        padded_size = input_len + 2 * max_jitter

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
            adam_eps=1e-7,
            gradient_clip_val=None,
        )

        count_pseudocount_scaled = resolve_count_pseudocount_from_args(
            args,
            bin_size=output_bin_size,
            target_scale=target_scale,
        )

        model_args = {
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": output_channels,
            "bias_logcount_offset": args.bias_logcount_offset,
            "accessibility_args": {
                "filters": args.filters,
                "n_dilated_layers": args.n_layers,
                "conv_kernel_size": args.conv_kernel_size,
                "dil_kernel_size": args.dil_kernel_size,
                "profile_kernel_size": args.profile_kernel_size,
                "activation": "relu",
                "weight_norm": False,
                "residual_architecture": args.residual_architecture,
            },
            "bias_args": {
                "filters": args.bias_filters,
                "n_dilated_layers": args.bias_layers,
                "conv_kernel_size": args.bias_conv_kernel_size,
                "dil_kernel_size": args.bias_dil_kernel_size,
                "profile_kernel_size": args.bias_profile_kernel_size,
                "activation": "relu",
                "weight_norm": False,
                "residual_architecture": args.residual_architecture,
            },
        }

        pretrained: list[PretrainedConfig] = [
            PretrainedConfig(
                weights_path=args.pretrained_bias,
                source=args.pretrained_bias_source,
                target="bias_model",
            )
        ]
        if args.pretrained_accessibility:
            pretrained.append(
                PretrainedConfig(
                    weights_path=args.pretrained_accessibility,
                    source=args.pretrained_accessibility_source,
                    target="accessibility_model",
                )
            )

        freeze: list[FreezeSpec] = []
        if not args.no_freeze_bias:
            freeze.append(FreezeSpec(pattern="bias_model", eval_mode=True))

        loss_args = {
            "alpha": args.alpha,
            "beta": args.beta,
            "cond_a_idx": args.differential_cond_a,
            "cond_b_idx": args.differential_cond_b,
        }
        if args.differential_pseudocount is not None:
            loss_args["delta_count_pseudocount"] = args.differential_pseudocount

        model_config = ModelConfig(
            name="MultitaskChromBPNetDifferentialParallel",
            model_cls="cerberus.models.chrombpnet.MultitaskChromBPNet",
            loss_cls="cerberus.models.bpnet.MultitaskBPNetJointDifferentialLoss",
            loss_args=loss_args,
            metrics_cls="cerberus.models.bpnet.JointBPNetMetricCollection",
            metrics_args={
                "cond_a_idx": args.differential_cond_a,
                "cond_b_idx": args.differential_cond_b,
            },
            model_args=model_args,
            pretrained=pretrained,
            freeze=freeze,
            count_pseudocount=count_pseudocount_scaled,
        )

        devices = _parse_devices(args.devices)
        accelerator = args.accelerator
        if accelerator == "auto" and torch.backends.mps.is_available():
            accelerator = "mps"

        precision_args = get_precision_kwargs(args.precision, accelerator, devices)

        logger.info("Genome Config:\n%s", pformat(genome_config))
        logger.info("Data Config:\n%s", pformat(data_config))
        logger.info("Sampler Config:\n%s", pformat(sampler_config))
        logger.info("Train Config:\n%s", pformat(train_config))
        logger.info("Model Config:\n%s", pformat(model_config))
        logger.info("Precision and Hardware Args:\n%s", pformat(precision_args))

        train_fn = train_multi if args.multi else train_single
        train_fn(
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
            _export_accessibility_checkpoints(output_dir)
            _plot_training_curves(output_dir)
            logger.info(
                "Parallel differential training finished. Logs and checkpoints "
                "are in subdirectories of %s",
                output_dir,
            )
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()


if __name__ == "__main__":
    main()
