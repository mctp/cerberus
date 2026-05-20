#!/usr/bin/env python
"""Multi-task ChromBPNet stage-2 training tool.

Trains a :class:`cerberus.models.MultitaskChromBPNet`: one accessibility
branch with one output channel per task, sharing a single pretrained
ChromBPNet bias BPNet across all tasks via :attr:`ChromBPNet.shared_bias`
broadcasting.

The bias subtree is loaded from the stage-1 checkpoint via
``PretrainedConfig(target="bias_model")`` and frozen by default through
``ModelConfig.freeze=[FreezeSpec(pattern="bias_model", eval_mode=True)]``
(use ``--no-freeze-bias`` to keep it trainable).  The accessibility
branch trains normally.

After training, exports an accessibility-only ``chrombpnet_wo_bias.pt``
checkpoint next to each ``model.pt``, matching the single-task
ChromBPNet trainer's export convention.

Assumes the per-task target BigWigs are already on a comparable scale
(e.g. CPM-normalised + constitutive-rescaled via
``tools/scatac_normalize_pseudobulk.py``); the count head trains
absolute log-counts per task.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import TextIO

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
from cerberus.pretrained import extract_prefix
from cerberus.train import train_multi, train_single
from cerberus.utils import get_precision_kwargs

# Sibling-tool import: works as a script (sys.path[0] == tools/) and from
# pytest after the test shim adds tools/ to sys.path.
from _pseudocount_cli import (  # noqa: E402  -- intentional after std imports
    add_pseudocount_cli_args,
    resolve_count_pseudocount_from_args,
)

logger = logging.getLogger(__name__)


def _parse_alpha(value: str) -> float | str:
    if value == "adaptive":
        return "adaptive"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--alpha must be a float or 'adaptive', got {value!r}"
        ) from exc


@contextmanager
def _open_text(path: str | Path) -> Iterator[TextIO]:
    """Open ``.bed`` or ``.bed.gz`` transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            yield handle
    else:
        with path.open() as handle:
            yield handle


def _sanitize_channel_name(name: str) -> str:
    # DataConfig.targets keys end up as Lightning metric names and
    # state-dict component names; whitespace is rejected downstream.
    return name.replace(" ", "_")


def _load_targets_json(path: str | Path) -> tuple[dict[str, str], list[str]]:
    """Load per-task BigWig paths (and optional peak paths) from a JSON spec.

    Accepted formats::

        {"task name": {"bigwig": "task.bw", "peaks": "task.peaks.bed.gz"}}
        {"task name": "task.bw"}   # peaks must be supplied via --peaks

    Returns:
        ``(targets, peak_paths)`` where ``targets`` is an ordered
        ``{channel_name: bigwig_path}`` mapping (keys sorted for
        reproducible Lightning logging) and ``peak_paths`` is a list of
        every ``"peaks"`` entry that appeared, in spec order.
    """
    with Path(path).open() as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict) or not spec:
        raise ValueError("--targets-json must contain a non-empty object")

    targets: dict[str, str] = {}
    peak_paths: list[str] = []
    for raw_name in sorted(spec):
        entry = spec[raw_name]
        channel = _sanitize_channel_name(raw_name)
        if channel in targets:
            raise ValueError(
                f"Duplicate sanitised channel name {channel!r}; "
                "choose unique task names"
            )
        if isinstance(entry, str):
            targets[channel] = entry
            continue
        if not isinstance(entry, dict) or "bigwig" not in entry:
            raise ValueError(
                "Each targets-json entry must be a BigWig string or an object "
                f"with a 'bigwig' key; got {raw_name!r}: {entry!r}"
            )
        targets[channel] = str(entry["bigwig"])
        if "peaks" in entry:
            peak_paths.append(str(entry["peaks"]))

    if len(targets) < 2:
        raise ValueError(
            f"Multi-task ChromBPNet requires at least two targets, got {len(targets)}"
        )
    return targets, peak_paths


def _merge_peaks(peak_paths: list[str], tmpdir: str) -> str:
    """Concatenate per-task peak BED/BED.gz files into one sampler BED."""
    if not peak_paths:
        raise ValueError("No peak paths were provided to merge")
    merged_path = Path(tmpdir) / "merged_peaks.bed"
    with merged_path.open("w") as out:
        for path in peak_paths:
            with _open_text(path) as handle:
                for line in handle:
                    out.write(line)
    return str(merged_path)


def _export_accessibility_checkpoints(root_dir: Path) -> None:
    """Export accessibility-only state dicts next to each full ``model.pt``.

    Matches the single-task ``train_chrombpnet.py`` convention so downstream
    inference / interpretation tools can load just the accessibility branch
    without instantiating the bias submodule.
    """
    for model_pt in sorted(root_dir.glob("**/model.pt")):
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
        acc_sd = extract_prefix(state_dict, "accessibility_model")
        out_path = model_pt.with_name("chrombpnet_wo_bias.pt")
        torch.save(acc_sd, out_path)
        logger.info("Saved accessibility-only checkpoint to %s", out_path)


def _plot_training_curves(root_dir: Path) -> None:
    """Best-effort plot of every Lightning CSV under the training root."""
    try:
        from plot_training_results import plot_metrics
    except Exception as exc:  # pragma: no cover - optional plotting deps
        logger.warning("Skipping training plots: %s", exc)
        return

    metrics_files = sorted(root_dir.rglob("metrics.csv"))
    if not metrics_files:
        logger.warning("No metrics.csv files found under %s; skipping plots.", root_dir)
        return
    for metrics_path in metrics_files:
        logger.info("Plotting training metrics from %s", metrics_path)
        plot_metrics(metrics_path, metrics_path.parent)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-task ChromBPNet model sharing one bias BPNet",
    )

    # --- Input data ---
    parser.add_argument(
        "--targets-json", type=str, required=True,
        help='JSON mapping task names to {"bigwig": ..., "peaks": ...} entries.',
    )
    parser.add_argument(
        "--peaks", type=str, default=None,
        help="Merged peak BED/narrowPeak for stage-2 sampling. If omitted, "
        "per-task peaks from --targets-json are concatenated.",
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
        "--background-ratio", type=float, default=1.0,
        help="Non-peak to peak ratio during stage-2 training",
    )
    parser.add_argument(
        "--target-scale", type=float, default=1.0,
        help="Multiplicative scaling factor for already-normalised targets "
        "(typically 1.0 when targets came through scatac_normalize_pseudobulk).",
    )

    # --- Pseudocount CLI family (shared with the other train_* tools) ---
    add_pseudocount_cli_args(parser, default_count_pseudocount=1.0)

    # --- Pretrained bias / accessibility branches ---
    parser.add_argument(
        "--pretrained-bias", type=str, required=True,
        help="Path to the stage-1 single-channel ChromBPNet bias BPNet checkpoint "
        "(e.g. produced by tools/train_chrombpnet_bias.py).",
    )
    parser.add_argument(
        "--pretrained-bias-source", type=str, default=None,
        help="Optional prefix to extract from the bias checkpoint "
        "(e.g. 'bias_model' when loading from a full ChromBPNet checkpoint).",
    )
    parser.add_argument(
        "--pretrained-accessibility", type=str, default=None,
        help="Optional checkpoint to warm-start the multi-task accessibility branch.",
    )
    parser.add_argument(
        "--pretrained-accessibility-source", type=str, default=None,
        help="Optional prefix to extract from the accessibility checkpoint.",
    )
    parser.add_argument(
        "--no-freeze-bias", action="store_true",
        help="Allow the loaded shared bias branch to keep training. Default behaviour "
        "adds FreezeSpec(pattern='bias_model', eval_mode=True) to ModelConfig.freeze "
        "so the bias broadcast contributes a fixed Tn5-bias baseline.",
    )
    parser.add_argument(
        "--bias-logcount-offset", type=float, default=0.0,
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
        "--residual-architecture", type=str,
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
        "--accelerator", type=str, default="auto",
        choices=["auto", "gpu", "cpu", "mps"],
    )
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument(
        "--precision", type=str, default="full",
        choices=["bf16", "mps", "full"],
        help="Default 'full' (fp32) for ChromBPNet: the bias-count "
        "logaddexp branch is sensitive to bf16 underflow.",
    )

    return parser.parse_args()


def main() -> None:
    cerberus.setup_logging()
    logger.info("Starting multi-task ChromBPNet training tool...")
    args = get_args()

    targets_dict, json_peak_paths = _load_targets_json(args.targets_json)
    output_channels = list(targets_dict.keys())
    logger.info(
        "Multi-task ChromBPNet targets (%d): %s",
        len(output_channels), output_channels,
    )

    tmpdir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.peaks:
        peaks_path = args.peaks
    elif json_peak_paths:
        tmpdir_obj = tempfile.TemporaryDirectory()
        peaks_path = _merge_peaks(json_peak_paths, tmpdir_obj.name)
        logger.info(
            "Merged %d per-task peak files into %s", len(json_peak_paths), peaks_path,
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
                "k": 5, "val_fold": args.val_fold, "test_fold": args.test_fold,
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
            args, bin_size=output_bin_size, target_scale=target_scale,
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

        # Freeze the bias subtree by default (ChromBPNet convention: the
        # Tn5-bias model is dataset-wide and frozen during accessibility
        # training).  eval_mode=True stops any Dropout/BatchNorm inside the
        # bias branch from drifting.
        freeze: list[FreezeSpec] = []
        if not args.no_freeze_bias:
            freeze.append(FreezeSpec(pattern="bias_model", eval_mode=True))

        model_config = ModelConfig(
            name="MultitaskChromBPNet",
            model_cls="cerberus.models.chrombpnet.MultitaskChromBPNet",
            loss_cls="cerberus.models.bpnet.MultitaskBPNetLoss",
            loss_args={"alpha": args.alpha, "beta": args.beta},
            metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
            metrics_args={},
            model_args=model_args,
            pretrained=pretrained,
            freeze=freeze,
            count_pseudocount=count_pseudocount_scaled,
        )

        devices: str | int = args.devices
        if devices != "auto":
            try:
                devices = int(devices)
            except ValueError:
                pass

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
                "Training finished. Logs and checkpoints are in subdirectories of %s",
                output_dir,
            )
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()


# Ensure the sibling-tool import works when this file is loaded by tests via
# importlib without going through the script entrypoint.
if __name__ == "__main__":  # pragma: no cover - covered by integration runs
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
