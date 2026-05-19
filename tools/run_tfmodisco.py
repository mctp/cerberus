#!/usr/bin/env python
"""Run TF-MoDISco motifs/report on precomputed NPZ inputs.

This script expects precomputed NPZ files with default key ``arr_0`` and
shape ``(N, 4, L)`` for both sequences and attribution scores.
"""

from __future__ import annotations

import argparse
import inspect
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
    try:
        from modiscolite import descriptive_report
    except ImportError as exc:
        raise RuntimeError(
            "Descriptive TF-MoDISco reports require the newer `modisco` package "
            "that exposes modiscolite.descriptive_report."
        ) from exc

    # modisco 2.5.2 calls _plot_weights(..., clamp=True), but some installed
    # builds expose _plot_weights(array, path, figsize) only. Patch the local
    # helper for compatibility.
    if "clamp" not in inspect.signature(descriptive_report._plot_weights).parameters:
        import logomaker
        import matplotlib.pyplot as plt
        import pandas as pd

        def _plot_weights_compat(array, path, figsize=(10, 3), clamp=True):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            df = pd.DataFrame(array, columns=["A", "C", "G", "T"])
            df.index.name = "pos"
            logomaker.Logo(df, ax=ax)
            ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
            if clamp:
                plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
            plt.savefig(path)
            plt.close(fig)

        descriptive_report._plot_weights = _plot_weights_compat

    # The descriptive_report ttl path in modisco 2.5.2 omits the output_dir
    # argument expected by modiscolite.report.tomtomlite_dataframe(). Patch the
    # module-local symbol so HOCOMOCO/MotifCompendium matching works without
    # external MEME Suite tomtom.
    if meme_db is not None and use_tomtom_lite:
        from modiscolite import report as modisco_report

        def _tomtomlite_dataframe_compat(
            modisco_h5py,
            meme_motif_db,
            pattern_groups,
            top_n_matches=3,
            trim_threshold=0.3,
            trim_min_length=3,
        ):
            return modisco_report.tomtomlite_dataframe(
                modisco_h5py,
                report_dir,
                meme_motif_db,
                pattern_groups,
                top_n_matches=top_n_matches,
                trim_threshold=trim_threshold,
                trim_min_length=trim_min_length,
            )

        descriptive_report.tomtomlite_dataframe = _tomtomlite_dataframe_compat

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
            "(or on existing --modisco-output if --no-run-motifs)."
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
                "e.g. `pip install modisco-lite` or `pip install -e .[extras]`."
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
