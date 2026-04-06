#!/usr/bin/env python
"""Run TF-MoDISco motifs/report on precomputed NPZ inputs.

This script expects precomputed NPZ files with default key ``arr_0`` and
shape ``(N, 4, L)`` for both sequences and attribution scores.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np

import cerberus

logger = logging.getLogger(__name__)


def _load_modisco_array(npz_path: Path, label: str) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"{label} NPZ not found: {npz_path}")

    with np.load(npz_path) as data:
        if "arr_0" not in data:
            raise ValueError(
                f"{label} NPZ '{npz_path}' is missing key 'arr_0' required by modisco motifs."
            )
        arr = data["arr_0"]

    if arr.ndim != 3:
        raise ValueError(f"{label} array must be 3D (N, 4, L); got shape {arr.shape}")
    if arr.shape[1] != 4:
        raise ValueError(
            f"{label} array must have 4 nucleotide channels in axis 1; got shape {arr.shape}"
        )
    return arr


def _run_modisco_motifs(
    ohe_path: Path,
    attr_path: Path,
    output_h5: Path,
    max_seqlets: int,
    window: int,
) -> None:
    cmd = [
        "modisco",
        "motifs",
        "-s",
        str(ohe_path),
        "-a",
        str(attr_path),
        "-n",
        str(max_seqlets),
        "-w",
        str(window),
        "-o",
        str(output_h5),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_modisco_report(modisco_h5: Path, report_dir: Path, meme_db: Path | None) -> None:
    cmd = [
        "modisco",
        "report",
        "-i",
        str(modisco_h5),
        "-o",
        str(report_dir),
        "-s",
        str(report_dir),
    ]
    if meme_db is not None:
        cmd.extend(["-m", str(meme_db)])

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TF-MoDISco aggregation from precomputed sequence and attribution NPZ files."
        )
    )
    parser.add_argument(
        "--ohe-path",
        type=Path,
        default=None,
        help="Path to one-hot sequence NPZ (arr_0 shape: N x 4 x L). Required with --run-motifs.",
    )
    parser.add_argument(
        "--attr-path",
        type=Path,
        default=None,
        help="Path to attribution NPZ (arr_0 shape: N x 4 x L). Required with --run-motifs.",
    )
    parser.add_argument(
        "--run-motifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `modisco motifs`.",
    )
    parser.add_argument(
        "--max-seqlets",
        type=int,
        default=2000,
        help="`modisco motifs -n` max seqlets per metacluster.",
    )
    parser.add_argument(
        "--modisco-window",
        type=int,
        default=400,
        help="`modisco motifs -w` window size around center.",
    )
    parser.add_argument(
        "--modisco-output",
        type=Path,
        default=Path("modisco_results.h5"),
        help="Output HDF5 file for `modisco motifs`.",
    )

    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Run `modisco report` after motifs (or on existing --modisco-output if --no-run-motifs).",
    )
    parser.add_argument(
        "--meme-db",
        type=Path,
        default=None,
        help=(
            "Optional motif DB (.meme) for `modisco report -m`. "
            "For human data, TF-MoDISco recommends MotifCompendium."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Output directory name/path for `modisco report`.",
    )

    return parser


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_h5 = args.modisco_output.resolve()

    if not args.run_motifs and not args.run_report:
        raise ValueError("Nothing to do: both --no-run-motifs and --run-report not set.")

    if args.run_motifs:
        if args.ohe_path is None or args.attr_path is None:
            raise ValueError("--ohe-path and --attr-path are required when --run-motifs is enabled.")

        ohe_path = args.ohe_path.resolve()
        attr_path = args.attr_path.resolve()

        ohe = _load_modisco_array(ohe_path, "Sequence")
        attrs = _load_modisco_array(attr_path, "Attribution")
        if ohe.shape != attrs.shape:
            raise ValueError(
                f"Shape mismatch between sequence and attribution arrays: {ohe.shape} vs {attrs.shape}"
            )

        output_h5.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_modisco_motifs(
                ohe_path=ohe_path,
                attr_path=attr_path,
                output_h5=output_h5,
                max_seqlets=args.max_seqlets,
                window=args.modisco_window,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`modisco` command not found. Install TF-MoDISco CLI first, "
                "e.g. `pip install modisco` or `pip install -e .[tfmodisco]`."
            ) from exc

    if args.run_report:
        if not output_h5.exists():
            raise FileNotFoundError(
                f"Cannot run report: MoDISco output not found at {output_h5}. "
                "Run with --run-motifs first or point --modisco-output to an existing .h5 file."
            )

        report_dir = args.report_dir
        if not report_dir.is_absolute():
            report_dir = output_h5.parent / report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        _run_modisco_report(output_h5, report_dir, args.meme_db)


if __name__ == "__main__":
    main()
