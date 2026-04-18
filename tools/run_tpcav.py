#!/usr/bin/env python
"""Run minimal TPCAV analysis on a trained single-task Cerberus BPNet model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from cerberus import setup_logging
from cerberus.attribution import TARGET_REDUCTIONS
from cerberus.model_ensemble import ModelEnsemble
from cerberus.tpcav import (
    build_tpcav_target_model,
    list_tpcav_probe_layers,
    resolve_tpcav_layer_name,
)
from cerberus.utils import resolve_device

logger = logging.getLogger(__name__)

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run minimal TPCAV analysis on a trained single-task Cerberus BPNet "
            "checkpoint using the upstream tpcav package."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "Training output directory. Can be a fold directory (contains model.pt) "
            "or a parent directory with fold_* subdirectories."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to load if --checkpoint-dir is a parent directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="tower_out",
        help=(
            "TPCAV probe layer alias or raw module path. "
            "Use --list-layers to inspect the Cerberus-friendly aliases."
        ),
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available Cerberus-friendly probe-layer aliases and exit.",
    )
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_REDUCTIONS),
        default="log_counts",
        help="Scalar output target used for TPCAV.",
    )
    parser.add_argument(
        "--target-channel",
        type=int,
        default=0,
        help="Target channel index for the scalar wrapper.",
    )
    parser.add_argument(
        "--bin-index",
        type=int,
        default=None,
        help="Output bin index for *_bin target modes (default: center bin).",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=None,
        help="Start bin (inclusive) for *_window_sum target modes.",
    )
    parser.add_argument(
        "--window-end",
        type=int,
        default=None,
        help="End bin (exclusive) for *_window_sum target modes.",
    )
    parser.add_argument(
        "--motif-file",
        type=Path,
        default=None,
        help="Concept source motif file (.meme or consensus TSV). Required unless --list-layers.",
    )
    parser.add_argument(
        "--motif-file-fmt",
        choices=["meme", "consensus"],
        default="meme",
        help="Motif file format expected by the upstream tpcav package.",
    )
    parser.add_argument(
        "--num-motif-insertions",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Number of motif insertions used to construct motif concepts.",
    )
    parser.add_argument(
        "--num-samples-for-pca",
        type=int,
        default=10,
        help="Number of examples per concept used to fit the PCA basis.",
    )
    parser.add_argument(
        "--num-samples-for-cav",
        type=int,
        default=1000,
        help="Number of examples per concept used to train each CAV.",
    )
    parser.add_argument(
        "--num-pc",
        type=str,
        default="full",
        help="PCA dimensionality passed through to tpcav (e.g. 'full', '0', '32').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for TPCAV concept sampling.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker count for concept data iterators.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Parallel process count for CAV training.",
    )
    parser.add_argument(
        "--max-pending-jobs",
        type=int,
        default=4,
        help="Maximum queued CAV jobs before backpressure.",
    )
    parser.add_argument(
        "--backend",
        choices=["sklearn", "torch"],
        default="sklearn",
        help="Classifier backend used by upstream tpcav.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
        help="Random seed passed through to tpcav.",
    )
    parser.add_argument(
        "--save-cav-trainer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist upstream cav_trainer.pt artifacts.",
    )
    parser.add_argument(
        "--generate-html-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the upstream HTML report.",
    )
    parser.add_argument(
        "--html-report-fscore-thresh",
        type=float,
        default=0.9,
        help="F-score threshold used by the upstream HTML report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for TPCAV artifacts. Required unless --list-layers.",
    )
    return parser


def _parse_num_pc(value: str) -> str | int:
    if value == "full":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"--num-pc must be 'full' or an integer, got {value!r}"
        ) from exc


def _load_target_model(
    checkpoint_dir: Path,
    fold: int,
    device: torch.device,
    *,
    target_mode: str,
    target_channel: int,
    bin_index: int | None,
    window_start: int | None,
    window_end: int | None,
):
    ensemble = ModelEnsemble(checkpoint_dir.resolve(), device=device, fold=fold)
    cfg = ensemble.cerberus_config
    model = ensemble[str(fold)]

    target_model = build_tpcav_target_model(
        model,
        reduction=target_mode,
        channel=target_channel,
        bin_index=bin_index,
        window_start=window_start,
        window_end=window_end,
    )
    target_model.to(device)
    target_model.eval()

    return target_model, cfg


def _print_layers(target_model: torch.nn.Module) -> None:
    aliases = list_tpcav_probe_layers(target_model)
    print("Available TPCAV probe layers:")
    for alias, raw in aliases.items():
        print(f"  {alias:<18} {raw}")


def _write_run_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    layer_name: str,
    cfg: Any,
) -> None:
    meta = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "fold": args.fold,
        "device": args.device,
        "layer_alias": args.layer,
        "layer_name": layer_name,
        "target_mode": args.target_mode,
        "target_channel": args.target_channel,
        "bin_index": args.bin_index,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "motif_file": str(args.motif_file.resolve()) if args.motif_file else None,
        "motif_file_fmt": args.motif_file_fmt,
        "num_motif_insertions": args.num_motif_insertions,
        "num_samples_for_pca": args.num_samples_for_pca,
        "num_samples_for_cav": args.num_samples_for_cav,
        "num_pc": args.num_pc,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_processes": args.num_processes,
        "backend": args.backend,
        "seed": args.seed,
        "genome_fasta": str(cfg.genome_config.fasta_path),
        "input_window_length": cfg.data_config.input_len,
    }
    with open(output_dir / "run_config.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def main() -> None:
    setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.list_layers:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --list-layers is used.")
        if args.motif_file is None:
            parser.error("--motif-file is required unless --list-layers is used.")

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    target_model, cfg = _load_target_model(
        args.checkpoint_dir,
        args.fold,
        device,
        target_mode=args.target_mode,
        target_channel=args.target_channel,
        bin_index=args.bin_index,
        window_start=args.window_start,
        window_end=args.window_end,
    )

    if args.list_layers:
        _print_layers(target_model)
        return

    layer_name = resolve_tpcav_layer_name(target_model, args.layer)
    logger.info("Resolved probe layer %r -> %s", args.layer, layer_name)

    try:
        from tpcav import run_tpcav
    except ImportError as exc:
        raise RuntimeError(
            "tpcav is required for tools/run_tpcav.py. "
            "Install it with `pip install -e .[tpcav]` or install the upstream "
            "`tpcav` package into the active environment."
        ) from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cavs_fscores_df, _motif_cav_trainers, _bed_cav_trainer = run_tpcav(
        model=target_model,
        motif_file=str(args.motif_file.resolve()),
        genome_fasta=str(cfg.genome_config.fasta_path),
        motif_file_fmt=args.motif_file_fmt,
        num_motif_insertions=sorted(args.num_motif_insertions),
        layer_name=layer_name,
        output_dir=str(output_dir),
        num_samples_for_pca=args.num_samples_for_pca,
        num_samples_for_cav=args.num_samples_for_cav,
        input_window_length=cfg.data_config.input_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_pc=_parse_num_pc(args.num_pc),
        p=args.num_processes,
        max_pending_jobs=args.max_pending_jobs,
        save_cav_trainer=args.save_cav_trainer,
        generate_html_report=args.generate_html_report,
        html_report_fscore_thresh=args.html_report_fscore_thresh,
        seed=args.seed,
        backend=args.backend,
        device=str(device),
    )

    if cavs_fscores_df is not None:
        cavs_fscores_path = output_dir / "motif_auc_fscores.tsv"
        cavs_fscores_df.to_csv(cavs_fscores_path, sep="\t", index=False)
        logger.info("Saved motif AUC/F-score summary to %s", cavs_fscores_path)

    _write_run_metadata(
        output_dir,
        args=args,
        layer_name=layer_name,
        cfg=cfg,
    )
    logger.info("TPCAV run complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
