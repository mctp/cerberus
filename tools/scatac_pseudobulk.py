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
    bulk.bw                         (with --bulk)
    bulk.narrowPeak.bed.gz          (with --bulk --call-peaks)
    bulk.narrowPeak.bed.gz.tbi
    merged.narrowPeak.bed.gz        (with --merge --call-peaks)
    merged.narrowPeak.bed.gz.tbi

Usage:
    python tools/scatac_pseudobulk.py \\
        fragments.tsv.bgz gene_activity.h5ad output_dir/ \\
        --genome hg38 --groupby cell_type --call-peaks --bulk

    # Keep original 10x +4/-5 coordinates (no additional correction):
    python tools/scatac_pseudobulk.py \\
        fragments.tsv.bgz gene_activity.h5ad output_dir/ \\
        --genome hg38 --groupby cell_type --no-shift
"""

import argparse
import gzip
import logging
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import anndata as ad
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

    Args:
        bed_path: Path to an uncompressed BED file.

    Returns:
        Path to the bgzipped file (.bed.gz).
    """
    gz_path = bed_path.parent / (bed_path.name + ".gz")

    # Sort by chrom, start, end then bgzip
    subprocess.run(
        f"sort -k1,1 -k2,2n '{bed_path}' | bgzip -c > '{gz_path}'",
        shell=True,
        check=True,
    )
    bed_path.unlink()

    # Tabix index
    subprocess.run(
        ["tabix", "-p", "bed", str(gz_path)],
        check=True,
    )
    logger.info(f"  Indexed: {gz_path}")
    return gz_path


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

    Opens its own read-only handle to the h5ad file, making it safe for
    multiprocessing (each process gets an independent file descriptor).

    Args:
        snap_h5ad: Path to the SnapATAC2 h5ad file with imported fragments.
        groupby: Column in data.obs to group by.
        selections: Optional subset of groups to export.
        args: Parsed CLI arguments.
        normalization: Normalization method or None for raw.
        out_dir: Output directory path.
        n_jobs: Thread count for coverage export. Defaults to args.n_jobs.

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
) -> None:
    """Call peaks with MACS3 and write bgzipped+tabixed narrowPeak BED files.

    Opens its own read-only handle to the h5ad file, making it safe for
    multiprocessing (each process gets an independent file descriptor).

    Writes one {group_name}.narrowPeak.bed.gz + .tbi per group, all in out_dir.
    The bgzip/tabix post-processing runs in parallel across groups.

    Args:
        snap_h5ad: Path to the SnapATAC2 h5ad file with imported fragments.
        groupby: Column in data.obs to group by.
        selections: Optional subset of groups to call peaks for.
        args: Parsed CLI arguments.
        out_dir: Output directory path.
        n_jobs: Thread count for MACS3 and bgzip/tabix parallelism.
            Defaults to args.n_jobs.
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
        bed_path = out_dir / f"{group_name}.narrowPeak.bed"
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
        "--bulk",
        action="store_true",
        help="Also generate a bulk BigWig and peak set using all cells",
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
        "--merge",
        action="store_true",
        help=(
            "Merge all per-group (and bulk, if --bulk) narrowPeak files into "
            "a single merged.narrowPeak.bed.gz. Overlapping peaks are collapsed "
            "and the summit is set to the median of constituent summits. "
            "Requires --call-peaks."
        ),
    )

    # Parallelism
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help=(
            "Max total concurrent threads/processes across all stages "
            "(default: 8). When stages overlap, the budget is split "
            "between them."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable stage overlap; run all stages strictly sequentially",
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
    barcode_to_group = dict(zip(meta_barcodes, groupby_series))
    adata_meta.file.close()
    logger.info(f"Loaded {len(meta_barcodes)} barcodes from metadata")

    # Import fragments into a backed AnnData via snapatac2
    # Use metadata barcodes as whitelist so only relevant cells are imported
    logger.info(f"Importing fragments from {args.fragments}...")
    snap_h5ad = args.output_dir / "_snap_fragments.h5ad"
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    logger.info(f"Imported {data.n_obs} cells from fragment file")

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
    group_counts = data.obs[args.groupby].value_counts()
    for row in group_counts.iter_rows():
        logger.info(f"  {row[0]}: {row[1]} cells")

    # Filter to requested groups
    selections = None
    if args.groups:
        available = set(data.obs[args.groupby].unique().to_list())
        missing = set(args.groups) - available
        if missing:
            raise ValueError(
                f"Groups not found: {missing}. Available: {sorted(available)}"
            )
        selections = args.groups

    # Resolve normalization (snapatac2 expects None for raw)
    normalization: _Normalization | None = (
        None if args.normalization == "raw"
        else args.normalization  # type: ignore[assignment]  # validated by argparse choices
    )
    out_dir: Path = args.output_dir

    # Close the main handle; all subsequent functions open their own
    # read-only handles, which is safe for concurrent multiprocessing.
    n_obs = data.n_obs
    data.close()

    # --- Per-group BigWig files ---
    logger.info("Exporting per-group pseudobulk BigWig files...")
    _export_coverage(snap_h5ad, args.groupby, selections, args, normalization, out_dir)

    # --- Parallel stage overlap ---
    # Uses ProcessPoolExecutor so each worker gets its own process and
    # independent HDF5 file descriptor (read-only), avoiding thread-safety
    # issues.  When --bulk --call-peaks, three tasks run concurrently:
    #   1. Per-group peak calling (uses n_jobs workers internally via MACS3)
    #   2. Bulk coverage export (uses n_jobs workers internally)
    #   3. Bulk peak calling (always 1 MACS3 process — single group)
    # Bulk peaks is always 1 process regardless of n_jobs, so the remaining
    # budget is split between per-group peaks and bulk coverage.
    can_overlap = args.bulk and args.call_peaks and not args.sequential
    if can_overlap:
        half = max(1, (args.n_jobs - 1) // 2)
        logger.info(
            f"Running per-group peaks (n_jobs={half}), bulk coverage "
            f"(n_jobs={half}), and bulk peaks (n_jobs=1) in parallel "
            f"(total budget: {args.n_jobs})..."
        )
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = {
                "bulk coverage": pool.submit(
                    _export_coverage,
                    snap_h5ad, _BULK_GROUP_COL, None, args, normalization,
                    out_dir, half,
                ),
                "per-group peaks": pool.submit(
                    _call_peaks,
                    snap_h5ad, args.groupby, selections, args, out_dir,
                    half,
                ),
                "bulk peaks": pool.submit(
                    _call_peaks,
                    snap_h5ad, _BULK_GROUP_COL, None, args, out_dir,
                    1,
                ),
            }
            # Collect all exceptions instead of failing on the first one
            errors: dict[str, BaseException] = {}
            for name, fut in futures.items():
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"  {name} failed: {exc}")
                    errors[name] = exc
            if errors:
                raise RuntimeError(
                    f"Parallel stage failed for: {', '.join(errors)}. "
                    f"See log output above for details."
                ) from next(iter(errors.values()))

    else:
        # --- Sequential fallback ---
        if args.call_peaks:
            _call_peaks(snap_h5ad, args.groupby, selections, args, out_dir)

        if args.bulk:
            logger.info(f"Exporting bulk BigWig ({n_obs} cells)...")
            _export_coverage(
                snap_h5ad, _BULK_GROUP_COL, None, args, normalization, out_dir,
            )
            if args.call_peaks:
                _call_peaks(
                    snap_h5ad, _BULK_GROUP_COL, None, args, out_dir,
                )

    # --- Merge peaks ---
    if args.merge:
        if not args.call_peaks:
            raise ValueError("--merge requires --call-peaks")
        peak_files = sorted(out_dir.glob("*.narrowPeak.bed.gz"))
        # Exclude any previous merged file
        peak_files = [p for p in peak_files if p.name != "merged.narrowPeak.bed.gz"]
        if peak_files:
            logger.info(
                f"Merging {len(peak_files)} narrowPeak files into "
                f"merged.narrowPeak.bed.gz..."
            )
            _merge_peaks(peak_files, out_dir)

    # Clean up temporary snap h5ad
    snap_h5ad.unlink(missing_ok=True)
    logger.info("All done.")


if __name__ == "__main__":
    main()
