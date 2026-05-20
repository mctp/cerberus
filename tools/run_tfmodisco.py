#!/usr/bin/env python
"""Run TF-MoDISco motifs/report on precomputed NPZ inputs.

This script expects precomputed NPZ files with default key ``arr_0`` and
shape ``(N, 4, L)`` for both sequences and attribution scores.

Reporting modes
---------------
``--run-report`` runs the legacy ``modisco report`` HTML output.

``--run-descriptive-report`` invokes ``modiscolite.descriptive_report``,
which renders embedded base64 logos + per-pattern seqlet distributions
+ optional Python ``tomtom-lite`` MEME matching.  Requires the
``modisco`` PyPI package at version ``>=2.5.2``.

Note on packaging confusion: the ``modiscolite`` Python module is
provided by **two** PyPI packages, which install to the same import
name and therefore conflict.

- ``modisco`` (kundajelab/tfmodisco, post-lite-merge): current upstream;
  versions ``2.4.0+``; ships ``modiscolite.descriptive_report`` from
  ``2.5.2`` onward.  Required for ``--run-descriptive-report``.
- ``modisco-lite`` (jmschrei/tfmodisco-lite, the historical fork): max
  ``2.4.0`` on PyPI, no ``descriptive_report``.  Do not install
  alongside ``modisco`` -- they conflict.

If ``--run-descriptive-report`` raises ``RuntimeError: Descriptive
TF-MoDISco reports require the newer 'modisco' package...``, install
``modisco>=2.5.2`` via ``pip install -U modisco``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
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


def _run_modisco_report(
    modisco_h5: Path, report_dir: Path, meme_db: Path | None
) -> None:
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


def _run_descriptive_report(
    modisco_h5: Path,
    report_dir: Path,
    meme_db: Path | None,
    use_tomtom_lite: bool,
    top_n_matches: int,
    n_examples: int,
    trim_threshold: float,
) -> None:
    """Generate a modisco>=2.5.2 descriptive HTML report (logos + matches).

    No version-compat shims are applied: the PyPI 2.5.2 release of the
    ``modisco`` package already aligns ``descriptive_report._plot_weights``
    (which has ``clamp=True``) and ``report.tomtomlite_dataframe`` (which
    matches the call signature ``descriptive_report`` uses internally).
    Older modiscolite forks that diverge on those APIs are unsupported.
    """
    try:
        from modiscolite import descriptive_report
    except ImportError as exc:
        raise RuntimeError(
            "Descriptive TF-MoDISco reports require the newer 'modisco' "
            "package (>=2.5.2) that exposes modiscolite.descriptive_report. "
            "Install it via `pip install -U 'modisco>=2.5.2'`.  Note: the "
            "legacy 'modisco-lite' PyPI package (max 2.4.0) does NOT include "
            "descriptive_report and installs to the same modiscolite import "
            "name -- if you have it, uninstall first."
        ) from exc

    report_dir.mkdir(parents=True, exist_ok=True)
    descriptive_report.generate_descriptive_report(
        str(modisco_h5),
        str(report_dir),
        meme_motif_db=str(meme_db) if meme_db is not None else None,
        top_n_matches=top_n_matches,
        ttl=use_tomtom_lite,
        n_examples=n_examples,
        trim_threshold=trim_threshold,
    )


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
            "For human data, TF-MoDISco recommends MotifCompendium. "
            "Requires external MEME Suite (`tomtom`) on PATH."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Output directory name/path for `modisco report`.",
    )
    parser.add_argument(
        "--run-descriptive-report",
        action="store_true",
        help=(
            "Run the newer modiscolite descriptive report after motifs "
            "(or on an existing --modisco-output if --no-run-motifs). "
            "Requires `modisco>=2.5.2`."
        ),
    )
    parser.add_argument(
        "--descriptive-report-dir",
        type=Path,
        default=Path("report"),
        help="Output directory name/path for the descriptive report.",
    )
    parser.add_argument(
        "--descriptive-report-tomtom-lite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Python tomtom-lite for motif matching in descriptive reports.",
    )
    parser.add_argument(
        "--descriptive-report-top-n-matches",
        type=int,
        default=3,
        help="Top motif database matches to show per pattern.",
    )
    parser.add_argument(
        "--descriptive-report-n-examples",
        type=int,
        default=10,
        help="Number of seqlet examples to show per pattern.",
    )
    parser.add_argument(
        "--descriptive-report-trim-threshold",
        type=float,
        default=0.3,
        help="Contribution threshold for trimming descriptive report logos.",
    )

    return parser


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_h5 = args.modisco_output.resolve()

    if not args.run_motifs and not args.run_report and not args.run_descriptive_report:
        raise ValueError(
            "Nothing to do: --no-run-motifs set and neither report mode requested."
        )

    if args.run_motifs:
        if args.ohe_path is None or args.attr_path is None:
            raise ValueError(
                "--ohe-path and --attr-path are required when --run-motifs is enabled."
            )

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
                "e.g. `pip install 'modisco>=2.5.2'` or `pip install -e .[extras]`."
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

        meme_db = args.meme_db
        if meme_db is not None:
            meme_db = meme_db.resolve()
            if not meme_db.exists():
                raise FileNotFoundError(f"Motif database not found: {meme_db}")
            if shutil.which("tomtom") is None:
                raise RuntimeError(
                    "`--meme-db` was provided, but `tomtom` is not available on PATH. "
                    "MEME Suite is treated as an external system dependency (not a pip package dependency). "
                    "Install MEME Suite (for example: `conda install -c conda-forge -c bioconda meme`) "
                    "or run report without --meme-db."
                )

        _run_modisco_report(output_h5, report_dir, meme_db)

    if args.run_descriptive_report:
        if not output_h5.exists():
            raise FileNotFoundError(
                f"Cannot run descriptive report: MoDISco output not found at {output_h5}. "
                "Run with --run-motifs first or point --modisco-output to an existing .h5 file."
            )

        report_dir = args.descriptive_report_dir
        if not report_dir.is_absolute():
            report_dir = output_h5.parent / report_dir

        meme_db = args.meme_db
        if meme_db is not None:
            meme_db = meme_db.resolve()
            if not meme_db.exists():
                raise FileNotFoundError(f"Motif database not found: {meme_db}")

        _run_descriptive_report(
            output_h5,
            report_dir,
            meme_db,
            use_tomtom_lite=args.descriptive_report_tomtom_lite,
            top_n_matches=args.descriptive_report_top_n_matches,
            n_examples=args.descriptive_report_n_examples,
            trim_threshold=args.descriptive_report_trim_threshold,
        )


if __name__ == "__main__":
    main()
