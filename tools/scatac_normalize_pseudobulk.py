#!/usr/bin/env python
"""Normalize scATAC pseudobulk BigWigs with CPM plus constitutive-peak scaling.

This is a companion preprocessing step for ``tools/scatac_pseudobulk.py``.  It
expects a pseudobulk output directory containing one ``<group>.bw`` per cell
type/sample group and, usually, per-group ``.narrowPeak.bed.gz`` files plus a
``bulk_merge.narrowPeak.bed.gz`` consensus peak set.

The normalization has two stages:

1. Convert raw pseudobulk signal to CPM using group library sizes.
2. Fit a CREsted-style baseline-accessibility scalar.  For each group, the tool
   takes that group's most accessible peaks, keeps the low-Gini peaks among
   them (strong but broadly accessible across groups), averages their CPM
   signal, and rescales groups so these constitutive anchors match a reference
   anchor level.

Output layout:
    output_dir/
        <group>.bw                         # CPM + constitutive-scaled BigWigs
        <group>.narrowPeak.bed.gz          # copied when present
        bulk_merge.narrowPeak.bed.gz       # copied when present
        normalization_summary.tsv
        normalization_metadata.json
        constitutive_anchor_peaks.tsv
        targets.json                       # multi-task training convenience

Usage:
    python tools/scatac_normalize_pseudobulk.py \\
        path/to/pseudobulk path/to/pseudobulk_cpm_constitutive \\
        --group-summary path/to/group_summary.tsv

    # If the input BigWigs were already generated with --normalization CPM:
    python tools/scatac_normalize_pseudobulk.py \\
        path/to/pseudobulk_cpm path/to/pseudobulk_cpm_constitutive \\
        --input-scale cpm
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pybigtools

logger = logging.getLogger(__name__)

ReferenceStrategy = Literal["max", "median", "mean"]

_DEFAULT_EXCLUDED_BIGWIGS = {"bulk.bw"}
_PEAK_SUFFIXES = (
    ".narrowPeak.bed.gz",
    ".narrowPeak.bed.gz.tbi",
    ".narrowPeak.bed",
    ".bed.gz",
    ".bed.gz.tbi",
    ".bed",
)


@dataclass(frozen=True)
class PeakRegion:
    index: int
    chrom: str
    start: int
    end: int
    score_start: int
    score_end: int


@dataclass(frozen=True)
class GroupTrack:
    group: str
    bigwig: Path
    peaks: Path | None


@dataclass(frozen=True)
class GroupNormalization:
    group: str
    input_bigwig: str
    output_bigwig: str
    input_scale: str
    library_size: float | None
    library_size_source: str
    cpm_scale: float
    constitutive_mean: float
    anchor_peaks: int
    used_anchor_fallback: bool
    baseline_weight: float
    final_scale: float


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _outputs_exist(paths: Sequence[Path], stage_name: str) -> bool:
    if not paths:
        return False
    existing = [path for path in paths if path.exists() and path.stat().st_size > 0]
    if len(existing) == len(paths):
        logger.info("Skipping %s: all %d output(s) already exist", stage_name, len(paths))
        for path in existing:
            logger.info("  Found: %s", path)
        return True
    return False


def _find_default_group_summary(pseudobulk_dir: Path) -> Path | None:
    for path in (
        pseudobulk_dir / "group_summary.tsv",
        pseudobulk_dir.parent / "group_summary.tsv",
    ):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _load_library_sizes(
    path: Path,
    group_column: str,
    library_size_column: str,
) -> dict[str, float]:
    sizes: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty group summary: {path}")
        missing = {group_column, library_size_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing column(s) {sorted(missing)} in {path}; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            group = row[group_column]
            try:
                size = float(row[library_size_column])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid library size for group {group!r}: "
                    f"{row[library_size_column]!r}"
                ) from exc
            if size <= 0:
                raise ValueError(f"Library size must be positive for group {group!r}")
            sizes[group] = size
    return sizes


def _bigwig_sum(path: Path) -> float:
    bw = pybigtools.open(str(path))
    try:
        info = bw.info()
    finally:
        bw.close()
    try:
        return float(info["summary"]["sum"])
    except KeyError as exc:
        raise ValueError(f"BigWig summary lacks total sum: {path}") from exc


def _discover_group_tracks(
    pseudobulk_dir: Path,
    groups: Sequence[str] | None,
    include_bulk: bool,
) -> list[GroupTrack]:
    if groups:
        bw_paths = [pseudobulk_dir / f"{group}.bw" for group in groups]
        missing = [path for path in bw_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing requested group BigWig(s): "
                + ", ".join(str(path) for path in missing)
            )
    else:
        bw_paths = sorted(pseudobulk_dir.glob("*.bw"))
        if not include_bulk:
            bw_paths = [
                path for path in bw_paths if path.name not in _DEFAULT_EXCLUDED_BIGWIGS
            ]

    tracks: list[GroupTrack] = []
    for bw_path in bw_paths:
        group = bw_path.name.removesuffix(".bw")
        peaks = _find_group_peaks(pseudobulk_dir, group)
        tracks.append(GroupTrack(group=group, bigwig=bw_path, peaks=peaks))

    if not tracks:
        raise ValueError(f"No pseudobulk BigWigs found in {pseudobulk_dir}")
    return tracks


def _find_group_peaks(pseudobulk_dir: Path, group: str) -> Path | None:
    for suffix in _PEAK_SUFFIXES:
        if suffix.endswith(".tbi"):
            continue
        candidate = pseudobulk_dir / f"{group}{suffix}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _resolve_consensus_peaks(pseudobulk_dir: Path, peaks: Path | None) -> Path:
    if peaks is not None:
        if not peaks.exists() or peaks.stat().st_size == 0:
            raise FileNotFoundError(f"Peak file is missing or empty: {peaks}")
        return peaks

    candidate = pseudobulk_dir / "bulk_merge.narrowPeak.bed.gz"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    raise FileNotFoundError(
        "No --peaks was supplied and bulk_merge.narrowPeak.bed.gz was not found "
        f"in {pseudobulk_dir}"
    )


def _load_peak_regions(
    peaks: Path,
    chrom_sizes: dict[str, int],
    target_region_width: int | None,
) -> list[PeakRegion]:
    regions: list[PeakRegion] = []
    with _open_text(peaks) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in chrom_sizes:
                continue
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            score_start, score_end = _score_interval(
                chrom=chrom,
                start=start,
                end=end,
                fields=parts,
                chrom_sizes=chrom_sizes,
                target_region_width=target_region_width,
            )
            regions.append(
                PeakRegion(
                    index=len(regions),
                    chrom=chrom,
                    start=start,
                    end=end,
                    score_start=score_start,
                    score_end=score_end,
                )
            )
    if not regions:
        raise ValueError(f"No usable peak intervals found in {peaks}")
    return regions


def _score_interval(
    chrom: str,
    start: int,
    end: int,
    fields: Sequence[str],
    chrom_sizes: dict[str, int],
    target_region_width: int | None,
) -> tuple[int, int]:
    if target_region_width is None or target_region_width <= 0:
        return max(0, start), min(chrom_sizes[chrom], end)

    center = (start + end) // 2
    if len(fields) >= 10:
        try:
            summit_offset = int(fields[9])
        except ValueError:
            summit_offset = -1
        if summit_offset >= 0:
            center = start + summit_offset

    width = target_region_width
    half_left = width // 2
    score_start = center - half_left
    score_end = score_start + width
    if score_start < 0:
        score_start = 0
        score_end = min(chrom_sizes[chrom], width)
    if score_end > chrom_sizes[chrom]:
        score_end = chrom_sizes[chrom]
        score_start = max(0, score_end - width)
    if score_end <= score_start:
        score_start = max(0, min(start, chrom_sizes[chrom] - 1))
        score_end = min(chrom_sizes[chrom], score_start + 1)
    return score_start, score_end


def _write_scoring_bed(regions: Sequence[PeakRegion], path: Path) -> None:
    with path.open("w") as handle:
        for region in regions:
            handle.write(
                f"{region.chrom}\t{region.score_start}\t{region.score_end}\t"
                f"peak_{region.index}\n"
            )


def _chrom_sizes_from_bigwig(path: Path) -> dict[str, int]:
    bw = pybigtools.open(str(path))
    try:
        chroms = bw.chroms()
    finally:
        bw.close()
    return {str(chrom): int(size) for chrom, size in chroms.items()}


def _extract_peak_matrix(
    tracks: Sequence[GroupTrack],
    scoring_bed: Path,
    signal_stat: str,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for track in tracks:
        logger.info("Extracting %s signal over peaks: %s", signal_stat, track.group)
        bw = pybigtools.open(str(track.bigwig))
        try:
            values = np.fromiter(
                (float(value) for value in bw.average_over_bed(scoring_bed, stats=signal_stat)),
                dtype=np.float64,
            )
        finally:
            bw.close()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values[values < 0] = 0.0
        columns.append(values)

    lengths = {col.shape[0] for col in columns}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent peak matrix column lengths: {sorted(lengths)}")
    return np.column_stack(columns)


def _gini_per_peak(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
    values = np.maximum(matrix, 0.0)
    n_groups = values.shape[1]
    if n_groups == 0:
        raise ValueError("Cannot compute Gini with zero groups")
    sorted_values = np.sort(values, axis=1)
    totals = sorted_values.sum(axis=1)
    ranks = np.arange(1, n_groups + 1, dtype=np.float64)
    weighted = (sorted_values * ranks).sum(axis=1)
    gini = np.zeros(values.shape[0], dtype=np.float64)
    nonzero = totals > 0
    gini[nonzero] = (2.0 * weighted[nonzero] / (n_groups * totals[nonzero])) - (
        (n_groups + 1.0) / n_groups
    )
    return np.clip(gini, 0.0, 1.0)


def _fit_constitutive_weights(
    cpm_matrix: np.ndarray,
    groups: Sequence[str],
    *,
    top_k_percent: float,
    peak_threshold: float,
    gini_std_threshold: float,
    min_anchor_peaks: int,
    reference_strategy: ReferenceStrategy,
    allow_anchor_fallback: bool,
    max_baseline_weight: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[bool], float]:
    if not 0 < top_k_percent <= 1:
        raise ValueError("--top-k-percent must be in the interval (0, 1]")

    peak_gini = _gini_per_peak(cpm_matrix)
    gini_threshold = float(peak_gini.mean() - gini_std_threshold * peak_gini.std())
    low_gini = peak_gini <= gini_threshold

    constitutive_means: list[float] = []
    anchors_by_group: list[np.ndarray] = []
    fallback_by_group: list[bool] = []

    for col_idx, group in enumerate(groups):
        values = cpm_matrix[:, col_idx]
        candidate_idx = np.flatnonzero(values > peak_threshold)
        if candidate_idx.size == 0:
            raise ValueError(
                f"No peaks above --peak-threshold for group {group!r}; "
                "lower the threshold or remove this group."
            )

        n_top = max(1, int(math.ceil(candidate_idx.size * top_k_percent)))
        candidate_order = np.argsort(values[candidate_idx])[::-1]
        top_idx = candidate_idx[candidate_order[:n_top]]
        anchors = top_idx[low_gini[top_idx]]
        used_fallback = False

        if anchors.size < min_anchor_peaks:
            if not allow_anchor_fallback:
                raise ValueError(
                    f"Only {anchors.size} constitutive anchors found for {group!r}; "
                    "raise --top-k-percent, lower --min-anchor-peaks, or allow fallback."
                )
            fallback_n = min(top_idx.size, max(1, min_anchor_peaks))
            anchors = top_idx[np.argsort(peak_gini[top_idx])[:fallback_n]]
            used_fallback = True
            logger.warning(
                "Group %s had only %d low-Gini anchors; using %d lowest-Gini "
                "top peaks as fallback",
                group,
                int(low_gini[top_idx].sum()),
                anchors.size,
            )

        anchor_mean = float(values[anchors].mean())
        if anchor_mean <= 0:
            raise ValueError(f"Constitutive anchor mean is non-positive for {group!r}")
        constitutive_means.append(anchor_mean)
        anchors_by_group.append(np.sort(anchors))
        fallback_by_group.append(used_fallback)

    means = np.asarray(constitutive_means, dtype=np.float64)
    if reference_strategy == "max":
        reference = float(means.max())
    elif reference_strategy == "median":
        reference = float(np.median(means))
    elif reference_strategy == "mean":
        reference = float(means.mean())
    else:
        raise ValueError(f"Unsupported reference strategy: {reference_strategy!r}")

    weights = reference / means
    if max_baseline_weight is not None:
        if max_baseline_weight <= 0:
            raise ValueError("--max-baseline-weight must be positive when set")
        weights = np.minimum(weights, max_baseline_weight)

    return (
        weights,
        means,
        peak_gini,
        anchors_by_group,
        fallback_by_group,
        gini_threshold,
    )


def _iter_scaled_bigwig_records(path: Path, scale: float):
    bw = pybigtools.open(str(path))
    try:
        chroms = bw.chroms()
        for chrom in chroms:
            for start, end, value in bw.records(chrom):
                scaled = float(value) * scale
                if scaled != 0.0:
                    yield (chrom, int(start), int(end), scaled)
    finally:
        bw.close()


def _write_scaled_bigwig(
    input_bw: Path,
    output_bw: Path,
    chrom_sizes: dict[str, int],
    scale: float,
    overwrite: bool,
) -> None:
    if output_bw.exists() and not overwrite:
        raise FileExistsError(f"Output BigWig exists; use --overwrite: {output_bw}")
    output_bw.parent.mkdir(parents=True, exist_ok=True)
    writer = pybigtools.open(str(output_bw), "w")  # type: ignore[attr-defined]
    try:
        writer.write(chrom_sizes, _iter_scaled_bigwig_records(input_bw, scale))
    finally:
        writer.close()


def _copy_peak_sidecars(
    tracks: Sequence[GroupTrack],
    pseudobulk_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    paths: list[Path] = []
    for track in tracks:
        if track.peaks is not None:
            paths.append(track.peaks)
            tbi = track.peaks.parent / (track.peaks.name + ".tbi")
            if tbi.exists():
                paths.append(tbi)
    for name in (
        "bulk_merge.narrowPeak.bed.gz",
        "bulk_merge.narrowPeak.bed.gz.tbi",
        "bulk_call.narrowPeak.bed.gz",
        "bulk_call.narrowPeak.bed.gz.tbi",
    ):
        path = pseudobulk_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)

    for src in paths:
        dest = output_dir / src.name
        if dest.exists() and not overwrite:
            continue
        shutil.copy2(src, dest)


def _write_summary(
    rows: Sequence[GroupNormalization],
    path: Path,
) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()),
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_anchor_peaks(
    path: Path,
    groups: Sequence[str],
    regions: Sequence[PeakRegion],
    peak_gini: np.ndarray,
    cpm_matrix: np.ndarray,
    anchors_by_group: Sequence[np.ndarray],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "group",
                "peak_index",
                "chrom",
                "start",
                "end",
                "score_start",
                "score_end",
                "gini",
                "cpm_signal",
            ]
        )
        for col_idx, group in enumerate(groups):
            for peak_idx in anchors_by_group[col_idx]:
                region = regions[int(peak_idx)]
                writer.writerow(
                    [
                        group,
                        region.index,
                        region.chrom,
                        region.start,
                        region.end,
                        region.score_start,
                        region.score_end,
                        f"{peak_gini[int(peak_idx)]:.10g}",
                        f"{cpm_matrix[int(peak_idx), col_idx]:.10g}",
                    ]
                )


def _write_targets_json(
    tracks: Sequence[GroupTrack],
    output_dir: Path,
    path: Path,
) -> None:
    targets: dict[str, dict[str, str]] = {}
    for track in sorted(tracks, key=lambda item: item.group):
        output_bw = output_dir / f"{track.group}.bw"
        entry = {"bigwig": str(output_bw)}
        output_peaks = output_dir / f"{track.group}.narrowPeak.bed.gz"
        if output_peaks.exists():
            entry["peaks"] = str(output_peaks)
        elif track.peaks is not None:
            entry["peaks"] = str(track.peaks)
        targets[track.group] = entry

    with path.open("w") as handle:
        json.dump(targets, handle, indent=2)
        handle.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPM-normalize scATAC pseudobulk BigWigs and apply CREsted-style "
            "constitutive-peak baseline scaling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pseudobulk_dir",
        type=Path,
        help="Input directory produced by tools/scatac_pseudobulk.py.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for normalized BigWigs and metadata.",
    )
    parser.add_argument(
        "--peaks",
        type=Path,
        default=None,
        help=(
            "Consensus peak BED/narrowPeak file. Default: "
            "<pseudobulk_dir>/bulk_merge.narrowPeak.bed.gz."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific pseudobulk groups to normalize. Default: all *.bw except bulk.bw.",
    )
    parser.add_argument(
        "--include-bulk",
        action="store_true",
        help="Include bulk.bw as a normalizable group. Default excludes it.",
    )
    parser.add_argument(
        "--input-scale",
        choices=["raw", "cpm"],
        default="raw",
        help=(
            "Scale of input BigWigs. Use raw for scatac_pseudobulk.py's default "
            "outputs and cpm for BigWigs already generated with --normalization CPM."
        ),
    )
    parser.add_argument(
        "--group-summary",
        type=Path,
        default=None,
        help=(
            "TSV with per-group library sizes. Default searches "
            "<pseudobulk_dir>/group_summary.tsv then parent/group_summary.tsv."
        ),
    )
    parser.add_argument(
        "--group-column",
        default="group",
        help="Group name column in --group-summary (default: group).",
    )
    parser.add_argument(
        "--library-size-column",
        default="atac_fragments",
        help="Library size column in --group-summary (default: atac_fragments).",
    )
    parser.add_argument(
        "--library-size-source",
        choices=["auto", "summary", "bigwig-sum"],
        default="auto",
        help=(
            "How to get raw-to-CPM denominators. auto uses --group-summary when "
            "available, otherwise the BigWig total signal."
        ),
    )
    parser.add_argument(
        "--signal-stat",
        choices=["sum", "mean0", "mean", "max"],
        default="sum",
        help="Peak-level BigWig statistic used to fit anchors (default: sum).",
    )
    parser.add_argument(
        "--target-region-width",
        type=int,
        default=1000,
        help=(
            "Fixed bp width centered on narrowPeak summit/region center for "
            "anchor fitting. Use 0 to score full peak intervals. (default: 1000)"
        ),
    )
    parser.add_argument(
        "--top-k-percent",
        type=float,
        default=0.01,
        help="Top fraction of accessible peaks considered per group (default: 0.01).",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.0,
        help="Minimum CPM peak signal before a peak can be an anchor candidate.",
    )
    parser.add_argument(
        "--gini-std-threshold",
        type=float,
        default=1.0,
        help=(
            "Low-Gini cutoff is mean(gini) - this * std(gini), matching the "
            "CREsted-style shared-accessibility filter. (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--min-anchor-peaks",
        type=int,
        default=50,
        help=(
            "Minimum anchors per group. If fewer pass the Gini cutoff, the tool "
            "uses the lowest-Gini top peaks as fallback unless disabled. (default: 50)"
        ),
    )
    parser.add_argument(
        "--no-anchor-fallback",
        action="store_true",
        help="Raise an error instead of using lowest-Gini top peaks when anchors are sparse.",
    )
    parser.add_argument(
        "--reference-strategy",
        choices=["max", "median", "mean"],
        default="max",
        help=(
            "Reference constitutive mean used for baseline weights. max matches "
            "CREsted-style upscaling. (default: max)"
        ),
    )
    parser.add_argument(
        "--max-baseline-weight",
        type=float,
        default=None,
        help="Optional cap on constitutive baseline weights.",
    )
    parser.add_argument(
        "--copy-peaks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy matching peak files and bulk_merge into output_dir (default: true).",
    )
    parser.add_argument(
        "--write-targets-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output_dir/targets.json for multi-task training (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized BigWigs and metadata.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def _resolve_cpm_scales(
    tracks: Sequence[GroupTrack],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float | None], list[str]]:
    if args.input_scale == "cpm":
        return (
            np.ones(len(tracks), dtype=np.float64),
            [None] * len(tracks),
            ["input-cpm"] * len(tracks),
        )

    group_summary = args.group_summary or _find_default_group_summary(args.pseudobulk_dir)
    summary_sizes: dict[str, float] = {}
    if group_summary is not None and args.library_size_source in {"auto", "summary"}:
        logger.info("Loading library sizes from %s", group_summary)
        summary_sizes = _load_library_sizes(
            group_summary,
            group_column=args.group_column,
            library_size_column=args.library_size_column,
        )
    elif args.library_size_source == "summary":
        raise FileNotFoundError(
            "--library-size-source summary was requested, but no group summary "
            "was found. Pass --group-summary."
        )

    cpm_scales: list[float] = []
    library_sizes: list[float | None] = []
    sources: list[str] = []
    for track in tracks:
        if track.group in summary_sizes:
            library_size = summary_sizes[track.group]
            source = "summary"
        elif args.library_size_source == "summary":
            raise ValueError(
                f"Group {track.group!r} is missing from group summary; "
                "cannot compute CPM scale."
            )
        else:
            library_size = _bigwig_sum(track.bigwig)
            source = "bigwig-sum"
            logger.warning(
                "Using BigWig total signal as library size for %s: %.6g",
                track.group,
                library_size,
            )
        if library_size <= 0:
            raise ValueError(f"Library size must be positive for group {track.group!r}")
        library_sizes.append(library_size)
        sources.append(source)
        cpm_scales.append(1_000_000.0 / library_size)
    return np.asarray(cpm_scales, dtype=np.float64), library_sizes, sources


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pseudobulk_dir: Path = args.pseudobulk_dir
    output_dir: Path = args.output_dir
    if not pseudobulk_dir.exists():
        raise FileNotFoundError(f"Input pseudobulk directory does not exist: {pseudobulk_dir}")
    if pseudobulk_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "Refusing to normalize in place. Choose a distinct output_dir so the "
            "raw pseudobulk BigWigs remain available."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = _discover_group_tracks(
        pseudobulk_dir,
        groups=args.groups,
        include_bulk=args.include_bulk,
    )
    groups = [track.group for track in tracks]
    expected_bigwigs = [output_dir / f"{group}.bw" for group in groups]
    expected_metadata = [
        output_dir / "normalization_summary.tsv",
        output_dir / "normalization_metadata.json",
    ]
    if not args.overwrite and _outputs_exist(
        [*expected_bigwigs, *expected_metadata],
        "normalized pseudobulks",
    ):
        logger.info("All normalized outputs already exist. Nothing to do.")
        return

    consensus_peaks = _resolve_consensus_peaks(pseudobulk_dir, args.peaks)
    chrom_sizes = _chrom_sizes_from_bigwig(tracks[0].bigwig)
    target_region_width = (
        None if args.target_region_width <= 0 else int(args.target_region_width)
    )
    regions = _load_peak_regions(consensus_peaks, chrom_sizes, target_region_width)
    logger.info("Loaded %d scoring regions from %s", len(regions), consensus_peaks)

    cpm_scales, library_sizes, library_sources = _resolve_cpm_scales(tracks, args)

    with tempfile.TemporaryDirectory(prefix="cerberus_scatac_norm_") as tmpdir:
        scoring_bed = Path(tmpdir) / "scoring_regions.bed"
        _write_scoring_bed(regions, scoring_bed)
        raw_matrix = _extract_peak_matrix(tracks, scoring_bed, args.signal_stat)

    cpm_matrix = raw_matrix * cpm_scales[np.newaxis, :]
    (
        baseline_weights,
        constitutive_means,
        peak_gini,
        anchors_by_group,
        fallback_by_group,
        gini_threshold,
    ) = _fit_constitutive_weights(
        cpm_matrix,
        groups,
        top_k_percent=args.top_k_percent,
        peak_threshold=args.peak_threshold,
        gini_std_threshold=args.gini_std_threshold,
        min_anchor_peaks=args.min_anchor_peaks,
        reference_strategy=args.reference_strategy,
        allow_anchor_fallback=not args.no_anchor_fallback,
        max_baseline_weight=args.max_baseline_weight,
    )

    final_scales = cpm_scales * baseline_weights
    rows: list[GroupNormalization] = []
    logger.info("Writing %d normalized BigWig(s) to %s", len(tracks), output_dir)
    for idx, track in enumerate(tracks):
        output_bw = output_dir / f"{track.group}.bw"
        _write_scaled_bigwig(
            track.bigwig,
            output_bw,
            chrom_sizes,
            scale=float(final_scales[idx]),
            overwrite=args.overwrite,
        )
        rows.append(
            GroupNormalization(
                group=track.group,
                input_bigwig=str(track.bigwig),
                output_bigwig=str(output_bw),
                input_scale=args.input_scale,
                library_size=library_sizes[idx],
                library_size_source=library_sources[idx],
                cpm_scale=float(cpm_scales[idx]),
                constitutive_mean=float(constitutive_means[idx]),
                anchor_peaks=int(anchors_by_group[idx].size),
                used_anchor_fallback=bool(fallback_by_group[idx]),
                baseline_weight=float(baseline_weights[idx]),
                final_scale=float(final_scales[idx]),
            )
        )

    if args.copy_peaks:
        _copy_peak_sidecars(tracks, pseudobulk_dir, output_dir, args.overwrite)

    _write_summary(rows, output_dir / "normalization_summary.tsv")
    _write_anchor_peaks(
        output_dir / "constitutive_anchor_peaks.tsv",
        groups,
        regions,
        peak_gini,
        cpm_matrix,
        anchors_by_group,
    )
    if args.write_targets_json:
        _write_targets_json(tracks, output_dir, output_dir / "targets.json")

    metadata = {
        "input_dir": str(pseudobulk_dir),
        "output_dir": str(output_dir),
        "consensus_peaks": str(consensus_peaks),
        "groups": groups,
        "n_groups": len(groups),
        "n_peaks": len(regions),
        "input_scale": args.input_scale,
        "signal_stat": args.signal_stat,
        "target_region_width": args.target_region_width,
        "top_k_percent": args.top_k_percent,
        "peak_threshold": args.peak_threshold,
        "gini_std_threshold": args.gini_std_threshold,
        "gini_threshold": gini_threshold,
        "min_anchor_peaks": args.min_anchor_peaks,
        "reference_strategy": args.reference_strategy,
        "max_baseline_weight": args.max_baseline_weight,
    }
    with (output_dir / "normalization_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    logger.info("Wrote summary: %s", output_dir / "normalization_summary.tsv")
    logger.info("Wrote anchor diagnostics: %s", output_dir / "constitutive_anchor_peaks.tsv")
    logger.info("All done.")


if __name__ == "__main__":
    main()
