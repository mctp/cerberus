#!/usr/bin/env python
"""Generate pseudobulk BigWig files and call peaks from scATAC-seq fragments using SnapATAC2.

Uses SnapATAC2's Rust-backed fragment importer, coverage exporter, and MACS3
peak caller for fast pseudobulk BigWig generation and peak calling.

Tn5 shift correction
---------------------
Tn5 transposase creates a 9 bp staggered double-strand cut. Standard 10x
pipelines (Cell Ranger ATAC, cellranger-arc) already shift fragment
coordinates by +4 bp (left/start) and -5 bp (right/end) in the output
fragments.tsv.gz, placing each cut site at the center of the 9 bp overhang.

Recent work (Mao et al. 2024, PRINT) shows that a symmetric +4/-4 shift
better models Tn5 sequence bias. By default, this tool applies an additional
+1 bp shift to the right end (--shift-right=1), converting from the 10x
+4/-5 convention to the +4/-4 convention. Use --no-shift to keep the
original 10x coordinates.

Output layout (all files in output_dir/):
    cell_type_A.bw
    cell_type_A.narrowPeak.bed.gz
    cell_type_A.narrowPeak.bed.gz.tbi
    ...
    bulk.bw
    bulk_call.narrowPeak.bed.gz     (with --bulk-peaks --call-peaks)
    bulk_call.narrowPeak.bed.gz.tbi
    bulk_merge.narrowPeak.bed.gz    (with --call-peaks, disable with --no-merge)
    bulk_merge.narrowPeak.bed.gz.tbi

Usage:
    python tools/scatac_pseudobulk.py \\
        fragments.tsv.bgz gene_activity.h5ad output_dir/ \\
        --genome hg38 --groupby cell_type --call-peaks

    # Re-run from scratch even if outputs already exist:
    python tools/scatac_pseudobulk.py \\
        fragments.tsv.bgz gene_activity.h5ad output_dir/ \\
        --genome hg38 --groupby cell_type --call-peaks --overwrite

    # Keep original 10x +4/-5 coordinates (no additional correction):
    python tools/scatac_pseudobulk.py \\
        fragments.tsv.bgz gene_activity.h5ad output_dir/ \\
        --genome hg38 --groupby cell_type --no-shift
"""

import argparse
import gzip
import logging
import multiprocessing
import os
import shutil
import statistics
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

# Disable HDF5 POSIX file locking before any HDF5-backed library is imported.
# POSIX fcntl locks are unreliable on NFS and cause deadlocks when multiple
# processes (from ProcessPoolExecutor) open the same .h5ad file concurrently.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import anndata as ad
import pysam  # type: ignore
import snapatac2 as snap  # type: ignore

logger = logging.getLogger(__name__)

# Built-in genome aliases supported by snapatac2 (Genome type lacks stubs)
GENOME_ALIASES: dict[str, Any] = {
    "hg38": snap.genome.hg38,  # type: ignore[attr-defined]
    "hg19": snap.genome.hg19,  # type: ignore[attr-defined]
    "mm10": snap.genome.mm10,  # type: ignore[attr-defined]
    "mm39": snap.genome.mm39,  # type: ignore[attr-defined]
}

# Constant column name used for bulk (all-cells) grouping
_BULK_GROUP_COL = "_bulk_group"
_BULK_GROUP_VAL = "bulk"
_BULK_CALL_PREFIX = "bulk_call"
_BULK_MERGE_PREFIX = "bulk_merge"


def load_chrom_sizes(path: Path) -> dict[str, int]:
    """Load chromosome sizes from a .fai or .chrom.sizes file.

    Accepts two formats:
    - FASTA index (.fai): 5 tab-separated columns, name + length in cols 0-1.
    - chrom.sizes: 2 tab-separated columns, name + length.
    """
    chrom_sizes: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def _bgzip_and_tabix(bed_path: Path) -> Path:
    """Sort, bgzip, and tabix-index a BED file. Returns path to .bed.gz.

    The original uncompressed BED file is removed after compression.
    Uses pysam for bgzip/tabix instead of requiring external binaries.

    Args:
        bed_path: Path to an uncompressed BED file.

    Returns:
        Path to the bgzipped file (.bed.gz).
    """
    gz_path = bed_path.parent / (bed_path.name + ".gz")

    # Sort by chrom, start, end in memory then write sorted file back
    with open(bed_path) as f:
        lines = f.readlines()
    lines.sort(key=lambda line: (line.split("\t")[0], int(line.split("\t")[1])))
    with open(bed_path, "w") as f:
        f.writelines(lines)

    # bgzip and tabix via pysam (no external binaries needed)
    pysam.tabix_compress(str(bed_path), str(gz_path), force=True)  # type: ignore[attr-defined]
    bed_path.unlink()
    pysam.tabix_index(str(gz_path), preset="bed", force=True)  # type: ignore[attr-defined]

    logger.info(f"  Indexed: {gz_path}")
    return gz_path


def _outputs_exist(paths: list[Path], stage_name: str) -> bool:
    """Check if all expected output files for a stage already exist and are non-empty.

    Args:
        paths: Expected output file paths.
        stage_name: Human-readable stage name for log messages.

    Returns:
        True if all paths exist and are non-empty (stage can be skipped),
        False otherwise.
    """
    if not paths:
        return False
    existing = [p for p in paths if p.exists() and p.stat().st_size > 0]
    if len(existing) == len(paths):
        logger.info(
            f"Skipping {stage_name}: all {len(paths)} output(s) already exist"
        )
        for p in existing:
            logger.info(f"  Found: {p}")
        return True
    return False


_Normalization = Literal["RPKM", "CPM", "BPM"]


def _export_coverage(
    snap_h5ad: Path,
    groupby: str,
    selections: list[str] | None,
    args: argparse.Namespace,
    normalization: "_Normalization | None",
    out_dir: Path,
    n_jobs: int | None = None,
) -> dict[str, str]:
    """Export pseudobulk BigWig files for a given groupby column.

    Opens its own read-only handle to the h5ad file, making it safe to call
    from a child process (each process gets an independent file descriptor).

    Internally, snapatac2 ``export_coverage`` uses ``n_jobs`` threads
    (Rust-backed) for parallel I/O within this single process.

    Args:
        snap_h5ad: Path to the SnapATAC2 h5ad file with imported fragments.
        groupby: Column in data.obs to group by.
        selections: Optional subset of groups to export.
        args: Parsed CLI arguments.
        normalization: Normalization method or None for raw.
        out_dir: Output directory path.
        n_jobs: Number of threads for coverage export. Defaults to args.n_jobs.

    Returns:
        Dict mapping group name to output file path.
    """
    # Configure logging for spawned processes (no-op if already configured)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    jobs = n_jobs if n_jobs is not None else args.n_jobs
    data = snap.read(snap_h5ad, backed="r")
    output_files = snap.ex.export_coverage(
        data,
        groupby=groupby,
        selections=selections,
        bin_size=args.bin_size,
        normalization=normalization,
        min_frag_length=args.min_frag_length,
        max_frag_length=args.max_frag_length,
        counting_strategy=args.counting_strategy,
        out_dir=out_dir,
        suffix=".bw",
        n_jobs=jobs,
    )
    data.close()
    for group_name, filepath in sorted(output_files.items()):
        logger.info(f"  Written: {group_name} -> {filepath}")
    return output_files


def _call_peaks(
    snap_h5ad: Path,
    groupby: str,
    selections: list[str] | None,
    args: argparse.Namespace,
    out_dir: Path,
    n_jobs: int | None = None,
    name_map: dict[str, str] | None = None,
) -> None:
    """Call peaks with MACS3 and write bgzipped+tabixed narrowPeak BED files.

    Opens its own read-only handle to the h5ad file, making it safe to call
    from a child process (each process gets an independent file descriptor).

    Internally, ``snap.tl.macs3`` spawns up to ``n_jobs`` MACS3 subprocesses
    (one per group). The bgzip/tabix post-processing uses ``n_jobs`` threads
    via ThreadPoolExecutor.

    Writes one {group_name}.narrowPeak.bed.gz + .tbi per group, all in out_dir.

    Args:
        snap_h5ad: Path to the SnapATAC2 h5ad file with imported fragments.
        groupby: Column in data.obs to group by.
        selections: Optional subset of groups to call peaks for.
        args: Parsed CLI arguments.
        out_dir: Output directory path.
        n_jobs: Number of MACS3 subprocesses and bgzip/tabix threads.
            Defaults to args.n_jobs.
        name_map: Optional mapping from group name to output file prefix.
            Groups not in the map keep their original name.
    """
    # Configure logging for spawned processes (no-op if already configured)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    jobs = n_jobs if n_jobs is not None else args.n_jobs
    data = snap.read(snap_h5ad, backed="r")
    logger.info("Calling peaks with MACS3...")
    peaks = snap.tl.macs3(
        data,
        groupby=groupby,
        qvalue=args.peak_qvalue,
        selections=set(selections) if selections else None,
        n_jobs=jobs,
        inplace=False,
    )
    data.close()
    assert peaks is not None, "macs3 with inplace=False must return a dict"

    # Write BED files, then bgzip+tabix in parallel
    bed_paths: list[Path] = []
    for group_name, peak_df in sorted(peaks.items()):
        file_prefix = (name_map or {}).get(group_name, group_name)
        bed_path = out_dir / f"{file_prefix}.narrowPeak.bed"
        peak_df.write_csv(str(bed_path), separator="\t", include_header=False)
        logger.info(f"  Peaks: {group_name} ({len(peak_df)} peaks)")
        bed_paths.append(bed_path)

    if bed_paths:
        workers = min(jobs, len(bed_paths))
        logger.info(
            f"Compressing and indexing {len(bed_paths)} peak files "
            f"({workers} workers)..."
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_bgzip_and_tabix, bed_paths))
    else:
        logger.info("No peaks to compress.")


def _merge_peaks(
    peak_files: list[Path],
    out_dir: Path,
    out_name: str = "merged",
) -> Path:
    """Merge overlapping narrowPeak peaks across groups into a single peak set.

    Overlapping peaks are collapsed into one interval spanning the union.
    The merged summit is the median absolute summit position of all
    constituent peaks (reported as offset from the merged chromStart).
    Score, signalValue, pValue, and qValue are taken as the max across
    constituents.

    Args:
        peak_files: List of bgzipped narrowPeak BED files to merge.
        out_dir: Output directory for the merged file.
        out_name: Base name for the output file (default: "merged").

    Returns:
        Path to the bgzipped+tabixed merged file.
    """
    # Read all peaks: (chrom, start, end, score, signal, pval, qval, abs_summit)
    peaks: list[tuple[str, int, int, int, float, float, float, int]] = []
    for pf in peak_files:
        with gzip.open(pf, "rt") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                score = int(parts[4])
                signal = float(parts[6])
                pval = float(parts[7])
                qval = float(parts[8])
                summit_offset = int(parts[9])
                peaks.append(
                    (chrom, start, end, score, signal, pval, qval, start + summit_offset)
                )

    logger.info(f"Read {len(peaks)} peaks from {len(peak_files)} files for merging")
    if not peaks:
        logger.info("No peaks to merge.")
        bed_path = out_dir / f"{out_name}.narrowPeak.bed"
        bed_path.write_text("")
        return _bgzip_and_tabix(bed_path)

    peaks.sort(key=lambda p: (p[0], p[1]))

    # Merge overlapping intervals
    merged: list[tuple[str, int, int, str, int, str, float, float, float, int]] = []
    cur_chrom, cur_start, cur_end = peaks[0][0], peaks[0][1], peaks[0][2]
    cur_scores = [peaks[0][3]]
    cur_signals = [peaks[0][4]]
    cur_pvals = [peaks[0][5]]
    cur_qvals = [peaks[0][6]]
    cur_summits = [peaks[0][7]]

    for chrom, start, end, score, signal, pval, qval, abs_summit in peaks[1:]:
        if chrom == cur_chrom and start <= cur_end:
            # Overlapping — extend
            cur_end = max(cur_end, end)
            cur_scores.append(score)
            cur_signals.append(signal)
            cur_pvals.append(pval)
            cur_qvals.append(qval)
            cur_summits.append(abs_summit)
        else:
            # Emit merged peak
            median_summit = int(statistics.median(cur_summits))
            merged.append((
                cur_chrom, cur_start, cur_end,
                ".", min(max(cur_scores), 1000), ".",
                max(cur_signals), max(cur_pvals), max(cur_qvals),
                median_summit - cur_start,
            ))
            cur_chrom, cur_start, cur_end = chrom, start, end
            cur_scores = [score]
            cur_signals = [signal]
            cur_pvals = [pval]
            cur_qvals = [qval]
            cur_summits = [abs_summit]

    # Emit last
    median_summit = int(statistics.median(cur_summits))
    merged.append((
        cur_chrom, cur_start, cur_end,
        ".", min(max(cur_scores), 1000), ".",
        max(cur_signals), max(cur_pvals), max(cur_qvals),
        median_summit - cur_start,
    ))

    logger.info(f"Merged into {len(merged)} peaks")

    bed_path = out_dir / f"{out_name}.narrowPeak.bed"
    with open(bed_path, "w") as f:
        for row in merged:
            f.write("\t".join(str(x) for x in row) + "\n")

    return _bgzip_and_tabix(bed_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pseudobulk BigWig files and call peaks from scATAC-seq fragments (SnapATAC2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fragments",
        type=Path,
        help="Fragment file (.tsv.gz or .tsv.bgz)",
    )
    parser.add_argument(
        "h5ad",
        type=Path,
        help="AnnData h5ad file with cell metadata",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for BigWig files and peaks",
    )

    # Genome / chrom sizes (mutually exclusive)
    genome_group = parser.add_mutually_exclusive_group(required=True)
    genome_group.add_argument(
        "--genome",
        type=str,
        choices=list(GENOME_ALIASES.keys()),
        help="Built-in genome name for chromosome sizes",
    )
    genome_group.add_argument(
        "--chrom-sizes",
        type=Path,
        help="Chromosome sizes file (.fai or .chrom.sizes)",
    )

    # Grouping
    parser.add_argument(
        "--groupby",
        type=str,
        default="cell_type",
        help="obs column to group cells by (default: cell_type)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Specific group names to process (default: all groups)",
    )
    parser.add_argument(
        "--bulk-peaks",
        action="store_true",
        help="Also call bulk peaks using all cells (slow). Bulk BigWig is always exported.",
    )

    # Coverage mode
    parser.add_argument(
        "--counting-strategy",
        type=str,
        choices=["insertion", "fragment", "paired-insertion"],
        default="insertion",
        help=(
            "Counting strategy. 'insertion': Tn5 cut sites. "
            "'fragment': full fragment coverage. "
            "'paired-insertion': counts insertion pair once if both "
            "fall within the same region. (default: insertion)"
        ),
    )

    # Normalization
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["CPM", "RPKM", "BPM", "raw"],
        default="raw",
        help="Normalization method (default: raw). Use 'raw' for no normalization.",
    )

    # Bin size
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1,
        help="Bin size in bp for coverage track (default: 1)",
    )

    # Fragment size filtering
    parser.add_argument(
        "--min-frag-length",
        type=int,
        default=None,
        help="Minimum fragment length in bp (default: no minimum)",
    )
    parser.add_argument(
        "--max-frag-length",
        type=int,
        default=2000,
        help="Maximum fragment length in bp (default: 2000)",
    )

    # Cell filtering
    parser.add_argument(
        "--min-num-fragments",
        type=int,
        default=200,
        help="Minimum number of fragments per cell to keep (default: 200)",
    )

    # Tn5 shift correction
    #
    # Background: Tn5 transposase creates a 9 bp staggered cut. Standard
    # pipelines (Cell Ranger ATAC, cellranger-arc) already apply a +4/-5 bp
    # shift to fragment coordinates in their fragments.tsv.gz output, placing
    # each end at the center of the 9 bp overhang.
    #
    # However, recent work (e.g. Mao et al. 2024, PRINT) shows that a +4/-4
    # shift better models Tn5 sequence bias symmetry. Since 10x fragments are
    # pre-shifted by +4/-5, the default shift_right=+1 here converts them to
    # the +4/-4 convention (+4/-5 +1 = +4/-4).
    #
    # Use --no-shift (sets both to 0) to skip any additional correction and
    # keep the original +4/-5 coordinates from 10x pipelines.
    tn5_group = parser.add_argument_group(
        "Tn5 shift correction",
        description=(
            "Adjust fragment coordinates for Tn5 insertion bias. "
            "10x Cell Ranger fragments are pre-shifted +4/-5. The defaults "
            "convert to the newer +4/-4 convention (shift_right=+1). "
            "Use --no-shift to keep the original 10x +4/-5 coordinates."
        ),
    )
    tn5_group.add_argument(
        "--shift-left",
        type=int,
        default=0,
        help=(
            "Additional bp shift for the left (start) end of each fragment. "
            "Positive values shift right (downstream). (default: 0, i.e. "
            "keep the +4 already applied by 10x pipelines)"
        ),
    )
    tn5_group.add_argument(
        "--shift-right",
        type=int,
        default=1,
        help=(
            "Additional bp shift for the right (end) end of each fragment. "
            "Positive values shift right (downstream). Default +1 converts "
            "10x's +4/-5 to the +4/-4 convention. (default: 1)"
        ),
    )
    tn5_group.add_argument(
        "--no-shift",
        action="store_true",
        help=(
            "Disable Tn5 shift correction (equivalent to --shift-left 0 "
            "--shift-right 0). Keeps original fragment coordinates as-is."
        ),
    )

    # Peak calling
    parser.add_argument(
        "--call-peaks",
        action="store_true",
        help="Call peaks with MACS3 after generating BigWig files",
    )
    parser.add_argument(
        "--peak-qvalue",
        type=float,
        default=0.05,
        help="MACS3 q-value cutoff for peak calling (default: 0.05)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help=(
            "Disable merging of per-group narrowPeak files. By default, when "
            "--call-peaks is set, all per-group narrowPeak files are merged into "
            "bulk_merge.narrowPeak.bed.gz. Overlapping peaks are collapsed and "
            "the summit is set to the median of constituent summits."
        ),
    )

    # Parallelism
    _cpu = os.cpu_count() or 8
    _default_n_jobs = max(4, _cpu // 2)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_default_n_jobs,
        help=(
            f"Parallelism budget passed to each stage "
            f"(default: max(4, cpu_count//2) = {_default_n_jobs}). "
            "Fragment import and coverage export use this many threads "
            "(Rust-backed). Peak calling spawns this many MACS3 "
            "subprocesses (one per group). When --bulk-peaks --call-peaks "
            "enables stage overlap, up to three stages run as separate "
            "processes via ProcessPoolExecutor, each receiving the full "
            "budget. Use --sequential to run stages one at a time."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Disable stage overlap; run all stages strictly sequentially "
            "in a single process. Each stage gets the full --n-jobs budget."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run all stages even if output files already exist. "
            "By default, stages whose outputs are already present are skipped."
        ),
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # --no-shift overrides both shift values to 0
    if args.no_shift:
        args.shift_left = 0
        args.shift_right = 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Log Tn5 shift settings
    if args.shift_left == 0 and args.shift_right == 0:
        logger.info("Tn5 shift: none (keeping original fragment coordinates)")
    else:
        logger.info(
            f"Tn5 shift: left={args.shift_left:+d}, right={args.shift_right:+d} "
            f"(applied on top of any pre-existing shift in the fragment file)"
        )

    # Resolve genome / chrom sizes
    if args.genome:
        chrom_sizes = GENOME_ALIASES[args.genome]
        logger.info(f"Using built-in genome: {args.genome}")
    else:
        chrom_sizes = load_chrom_sizes(args.chrom_sizes)
        logger.info(f"Loaded {len(chrom_sizes)} chromosomes from {args.chrom_sizes}")

    # Load cell metadata first to use as whitelist during import
    logger.info(f"Loading cell metadata from {args.h5ad}...")
    adata_meta = ad.read_h5ad(args.h5ad, backed="r")
    meta_barcodes = list(adata_meta.obs_names)
    groupby_series = adata_meta.obs[args.groupby]
    barcode_to_group = dict(zip(meta_barcodes, groupby_series, strict=True))
    adata_meta.file.close()
    logger.info(f"Loaded {len(meta_barcodes)} barcodes from metadata")

    # Resolve normalization (snapatac2 expects None for raw)
    normalization: _Normalization | None = (
        None if args.normalization == "raw"
        else args.normalization  # type: ignore[assignment]  # validated by argparse choices
    )
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine expected group names from metadata for skip checks.
    # This is an approximation — actual groups after import may differ
    # due to min_num_fragments filtering, but it is sufficient to detect
    # whether a previous run's outputs are present.
    expected_groups: list[str] = sorted(
        set(args.groups) if args.groups else set(str(g) for g in barcode_to_group.values())
    )

    # --- Determine which stages need to run (skip check) ---
    skip_group_cov = (
        not args.overwrite
        and _outputs_exist(
            [out_dir / f"{g}.bw" for g in expected_groups],
            "per-group coverage",
        )
    )
    skip_group_peaks = (
        not args.call_peaks
        or (
            not args.overwrite
            and _outputs_exist(
                [out_dir / f"{g}.narrowPeak.bed.gz" for g in expected_groups],
                "per-group peaks",
            )
        )
    )
    skip_bulk_cov = (
        not args.overwrite
        and _outputs_exist(
            [out_dir / f"{_BULK_GROUP_VAL}.bw"],
            "bulk coverage",
        )
    )
    skip_bulk_peaks = (
        not (args.bulk_peaks and args.call_peaks)
        or (
            not args.overwrite
            and _outputs_exist(
                [out_dir / f"{_BULK_CALL_PREFIX}.narrowPeak.bed.gz"],
                "bulk peaks",
            )
        )
    )
    skip_merge = (
        args.no_merge
        or not args.call_peaks
        or (
            not args.overwrite
            and _outputs_exist(
                [out_dir / f"{_BULK_MERGE_PREFIX}.narrowPeak.bed.gz"],
                "merged peaks",
            )
        )
    )

    all_skipped = (
        skip_group_cov and skip_group_peaks
        and skip_bulk_cov and skip_bulk_peaks and skip_merge
    )
    if all_skipped:
        logger.info("All outputs already exist. Nothing to do.")
        logger.info("All done.")
        return

    # Import fragments into a backed AnnData via snapatac2
    # Reuse existing snap h5ad from a previous run if present (unless --overwrite)
    snap_h5ad = out_dir / "_snap_fragments.h5ad"

    data: Any = None
    if not args.overwrite and snap_h5ad.exists() and snap_h5ad.stat().st_size > 0:
        try:
            data = snap.read(snap_h5ad, backed="r+")
            logger.info(f"Reusing existing fragment store: {snap_h5ad}")
        except (Exception, BaseException):
            logger.warning(
                f"Existing fragment store is corrupt, re-importing: {snap_h5ad}"
            )
            snap_h5ad.unlink(missing_ok=True)
            data = None
    if data is None:
        logger.info(f"Importing fragments from {args.fragments}...")
        data = snap.pp.import_fragments(
            args.fragments,
            chrom_sizes=chrom_sizes,
            file=snap_h5ad,
            sorted_by_barcode=False,
            whitelist=meta_barcodes,
            min_num_fragments=args.min_num_fragments,
            shift_left=args.shift_left,
            shift_right=args.shift_right,
            n_jobs=args.n_jobs,
        )
    logger.info(f"Fragment store has {data.n_obs} cells")

    if data.n_obs == 0:
        raise ValueError(
            "No overlapping barcodes between fragment file and h5ad metadata. "
            "Check that barcode formats match."
        )

    # Transfer groupby column to snap data
    imported_barcodes = list(data.obs_names)
    data.obs[args.groupby] = [str(barcode_to_group[b]) for b in imported_barcodes]

    # Add bulk group column (all cells get the same label)
    data.obs[_BULK_GROUP_COL] = [_BULK_GROUP_VAL] * data.n_obs

    # Log group sizes (snapatac2 obs uses Polars)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="_import_from_c", category=DeprecationWarning)
        group_counts = data.obs[args.groupby].value_counts()
    for row in group_counts.iter_rows():
        logger.info(f"  {row[0]}: {row[1]} cells")

    # Filter to requested groups
    selections = None
    if args.groups:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="_import_from_c", category=DeprecationWarning)
            available = set(data.obs[args.groupby].unique().to_list())
        missing = set(args.groups) - available
        if missing:
            raise ValueError(
                f"Groups not found: {missing}. Available: {sorted(available)}"
            )
        selections = args.groups

    # Close the main handle; all subsequent functions open their own
    # read-only handles, which is safe for concurrent multiprocessing.
    n_obs = data.n_obs
    data.close()

    need_group_peaks = not skip_group_peaks
    need_bulk_cov = not skip_bulk_cov
    need_bulk_peaks = not skip_bulk_peaks

    if args.sequential:
        # --- Sequential mode: one stage at a time, full n_jobs each ---
        if not skip_group_cov:
            logger.info("Exporting per-group pseudobulk BigWig files...")
            _export_coverage(
                snap_h5ad, args.groupby, selections, args, normalization, out_dir,
            )
        if need_group_peaks:
            _call_peaks(snap_h5ad, args.groupby, selections, args, out_dir)
        if need_bulk_cov:
            logger.info(f"Exporting bulk BigWig ({n_obs} cells)...")
            _export_coverage(
                snap_h5ad, _BULK_GROUP_COL, None, args, normalization, out_dir,
            )
        if need_bulk_peaks:
            _call_peaks(snap_h5ad, _BULK_GROUP_COL, None, args, out_dir,
                        name_map={_BULK_GROUP_VAL: _BULK_CALL_PREFIX})
    else:
        # --- Parallel mode ---
        # Bulk peaks is the slowest stage (all cells, single MACS3 call),
        # so it starts first as a background process.  Per-group coverage
        # runs in the main process while bulk peaks warms up.  Then the
        # remaining stages (per-group peaks, bulk coverage) overlap with
        # the still-running bulk peaks.  Each stage gets the full n_jobs
        # budget; MACS3 subprocesses and Rust I/O threads are lightweight,
        # so mild oversubscription is fine.
        mp_ctx = multiprocessing.get_context("forkserver")
        bg_futures: dict[str, Any] = {}
        pool = ProcessPoolExecutor(max_workers=3, mp_context=mp_ctx)

        # Start bulk peaks first — it has the longest wall-clock time
        if need_bulk_peaks:
            logger.info("Starting bulk peaks in background (slowest stage)...")
            bg_futures["bulk peaks"] = pool.submit(
                _call_peaks,
                snap_h5ad, _BULK_GROUP_COL, None, args, out_dir, 1,
                {_BULK_GROUP_VAL: _BULK_CALL_PREFIX},
            )

        # Per-group coverage in main process (overlaps with bulk peaks)
        if not skip_group_cov:
            logger.info("Exporting per-group pseudobulk BigWig files...")
            _export_coverage(
                snap_h5ad, args.groupby, selections, args, normalization, out_dir,
            )

        # Submit remaining stages to pool (overlap with bulk peaks)
        if need_group_peaks:
            bg_futures["per-group peaks"] = pool.submit(
                _call_peaks,
                snap_h5ad, args.groupby, selections, args, out_dir,
                args.n_jobs,
            )
        if need_bulk_cov:
            bg_futures["bulk coverage"] = pool.submit(
                _export_coverage,
                snap_h5ad, _BULK_GROUP_COL, None, args, normalization,
                out_dir, args.n_jobs,
            )

        if bg_futures:
            logger.info(
                f"Waiting for {len(bg_futures)} background stage(s): "
                f"{', '.join(bg_futures)}..."
            )

        # Collect all exceptions
        errors: dict[str, BaseException] = {}
        for name, fut in bg_futures.items():
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"  {name} failed: {exc}")
                errors[name] = exc
        pool.shutdown(wait=False)
        if errors:
            raise RuntimeError(
                f"Parallel stage failed for: {', '.join(errors)}. "
                f"See log output above for details."
            ) from next(iter(errors.values()))

    # --- Merge peaks ---
    if not skip_merge:
        merge_out = f"{_BULK_MERGE_PREFIX}.narrowPeak"
        peak_files = sorted(out_dir.glob("*.narrowPeak.bed.gz"))
        # Exclude previous merge output and bulk called peaks (all-cells aggregate,
        # not a biological group — including it would double-count every peak)
        _exclude = {
            f"{_BULK_MERGE_PREFIX}.narrowPeak.bed.gz",
            f"{_BULK_CALL_PREFIX}.narrowPeak.bed.gz",
        }
        peak_files = [p for p in peak_files if p.name not in _exclude]
        if len(peak_files) > 1:
            logger.info(
                f"Merging {len(peak_files)} narrowPeak files into "
                f"{merge_out}.bed.gz..."
            )
            _merge_peaks(peak_files, out_dir, out_name=_BULK_MERGE_PREFIX)
        elif len(peak_files) == 1:
            logger.info(
                f"Only one peak file ({peak_files[0].name}), "
                f"copying as {merge_out}.bed.gz"
            )
            merged_path = out_dir / f"{merge_out}.bed.gz"
            shutil.copy2(peak_files[0], merged_path)
            tbi_src = peak_files[0].parent / (peak_files[0].name + ".tbi")
            if tbi_src.exists():
                shutil.copy2(tbi_src, out_dir / f"{merge_out}.bed.gz.tbi")
        else:
            logger.info("No peak files found to merge.")

    # Keep snap h5ad for future reuse (skipped on next run without --overwrite).
    # Delete with --overwrite on the *next* run (import_fragments overwrites it).
    logger.info(f"Fragment store retained at {snap_h5ad}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
