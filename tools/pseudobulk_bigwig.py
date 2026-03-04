#!/usr/bin/env python
"""Generate pseudobulk BigWig files from scATAC-seq fragment files.

Reads a tabix-indexed fragment file and an AnnData h5ad for cell metadata,
groups cells by a chosen obs column (e.g. cell_type), and writes one BigWig
per group.

The Tn5 +4/-5 shift is already applied in Cell Ranger ATAC fragment files.
This tool does NOT apply an additional shift by default.

Usage:
    python tools/pseudobulk_bigwig.py \\
        fragments.tsv.bgz gene_activity.h5ad chrom.sizes output_dir/ \\
        --groupby cell_type \\
        --mode insertion \\
        --normalization cpm \\
        --min-fragsize 0 --max-fragsize 147

    # Using a cerberus .fai file for chrom sizes:
    python tools/pseudobulk_bigwig.py \\
        fragments.tsv.bgz gene_activity.h5ad hg38.fa.fai output_dir/
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pybigtools  # type: ignore
import pysam

logger = logging.getLogger(__name__)


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


def load_barcode_groups(
    h5ad_path: Path,
    groupby: str,
) -> dict[str, set[str]]:
    """Load barcode-to-group mapping from an h5ad file.

    Args:
        h5ad_path: Path to the AnnData h5ad file.
        groupby: Column name in obs to group by (e.g. 'cell_type').

    Returns:
        Dict mapping group name to set of barcodes.
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")
    groups: dict[str, set[str]] = {}
    for barcode, group in zip(adata.obs_names, adata.obs[groupby]):
        groups.setdefault(str(group), set()).add(str(barcode))
    adata.file.close()
    logger.info(
        f"Loaded {sum(len(v) for v in groups.values())} barcodes "
        f"across {len(groups)} groups from '{groupby}'"
    )
    for name, barcodes in sorted(groups.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {name}: {len(barcodes)} cells")
    return groups


def _sanitize_filename(name: str) -> str:
    """Convert a group name to a filesystem-safe filename."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def generate_pseudobulk_bigwig(
    fragments_path: Path,
    chrom_sizes: dict[str, int],
    barcodes: set[str],
    output_path: Path,
    mode: str,
    normalization: str,
    min_fragsize: int,
    max_fragsize: int,
    tn5_shift: int,
) -> None:
    """Generate a single pseudobulk BigWig from a fragment file.

    Args:
        fragments_path: Path to tabix-indexed fragment file.
        chrom_sizes: Dict of chrom -> size.
        barcodes: Set of barcodes to include.
        output_path: Output BigWig path.
        mode: Coverage mode - 'insertion' (Tn5 cut sites) or 'fragment'
              (full fragment coverage).
        normalization: 'cpm' (counts per million), 'rpkm', or 'raw'.
        min_fragsize: Minimum fragment size (inclusive).
        max_fragsize: Maximum fragment size (inclusive). 0 means no upper limit.
        tn5_shift: Additional Tn5 shift to apply (0 if already shifted).
    """
    tbx = pysam.TabixFile(str(fragments_path))
    available_contigs = set(tbx.contigs)

    # Filter to chroms present in both fragment file and chrom sizes
    chroms = [c for c in chrom_sizes if c in available_contigs]

    # Accumulate per-chrom arrays, then normalize and stream to BigWig
    chrom_arrays: dict[str, np.ndarray] = {}

    logger.info(f"Processing {len(chroms)} chromosomes for {len(barcodes)} barcodes...")
    total_fragments = 0
    total_kept = 0

    for chrom in chroms:
        size = chrom_sizes[chrom]
        arr = np.zeros(size, dtype=np.float32)

        for line in tbx.fetch(chrom):
            parts = line.split("\t")
            total_fragments += 1

            barcode = parts[3]
            if barcode not in barcodes:
                continue

            start = int(parts[1])
            end = int(parts[2])
            fragsize = end - start

            # Fragment size filter
            if fragsize < min_fragsize:
                continue
            if max_fragsize > 0 and fragsize > max_fragsize:
                continue

            count = int(parts[4]) if len(parts) > 4 else 1
            total_kept += 1

            if mode == "insertion":
                # +1 at each Tn5 insertion site (fragment endpoints)
                left = start + tn5_shift
                right = end - 1 - tn5_shift
                if 0 <= left < size:
                    arr[left] += count
                if 0 <= right < size:
                    arr[right] += count
            elif mode == "fragment":
                # Full fragment coverage
                adj_start = max(0, start)
                adj_end = min(size, end)
                arr[adj_start:adj_end] += count

        chrom_arrays[chrom] = arr

    tbx.close()

    logger.info(
        f"Processed {total_fragments} total fragments, "
        f"kept {total_kept} after barcode + size filtering"
    )

    # Compute total signal for normalization
    total_signal = sum(float(arr.sum()) for arr in chrom_arrays.values())

    if normalization == "cpm":
        scale = 1_000_000.0 / total_signal if total_signal > 0 else 1.0
        logger.info(f"CPM normalization: total_signal={total_signal:.0f}, scale={scale:.6f}")
    elif normalization == "rpkm":
        total_reads = total_kept
        scale = 1_000_000_000.0 / total_reads if total_reads > 0 else 1.0
        logger.info(f"RPKM normalization: total_reads={total_reads}, scale={scale:.6f}")
    else:
        scale = 1.0

    if scale != 1.0:
        for arr in chrom_arrays.values():
            arr *= scale

    # Write BigWig using pybigtools streaming interface
    bw_chrom_sizes = {c: chrom_sizes[c] for c in chroms}

    def value_stream():
        for chrom in chroms:
            arr = chrom_arrays[chrom]
            # Find runs of nonzero values and emit them as bedGraph intervals
            nonzero = np.nonzero(arr)[0]
            if len(nonzero) == 0:
                continue

            # Group consecutive positions into intervals for efficiency
            # Split where gap > 1
            splits = np.where(np.diff(nonzero) > 1)[0] + 1
            for group in np.split(nonzero, splits):
                start = int(group[0])
                end = int(group[-1]) + 1
                for pos in range(start, end):
                    yield (chrom, pos, pos + 1, float(arr[pos]))

    logger.info(f"Writing BigWig to {output_path}...")
    bw = pybigtools.open(str(output_path), "w")
    bw.write(bw_chrom_sizes, value_stream())
    logger.info(f"Done: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pseudobulk BigWig files from scATAC-seq fragments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "fragments",
        type=Path,
        help="Tabix-indexed fragment file (.tsv.bgz)",
    )
    parser.add_argument(
        "h5ad",
        type=Path,
        help="AnnData h5ad file with cell metadata",
    )
    parser.add_argument(
        "chrom_sizes",
        type=Path,
        help="Chromosome sizes file (.fai or .chrom.sizes)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for BigWig files",
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
        "--all-cells",
        action="store_true",
        help="Also generate a combined BigWig with all cells",
    )

    # Coverage mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["insertion", "fragment"],
        default="insertion",
        help=(
            "Coverage mode. 'insertion': count at Tn5 cut sites (fragment endpoints). "
            "'fragment': count across full fragment length. (default: insertion)"
        ),
    )

    # Normalization
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["cpm", "rpkm", "raw"],
        default="cpm",
        help="Normalization method: cpm, rpkm, or raw counts (default: cpm)",
    )

    # Fragment size filtering
    parser.add_argument(
        "--min-fragsize",
        type=int,
        default=0,
        help="Minimum fragment size in bp, inclusive (default: 0)",
    )
    parser.add_argument(
        "--max-fragsize",
        type=int,
        default=0,
        help="Maximum fragment size in bp, inclusive. 0 = no limit (default: 0)",
    )

    # Tn5 shift
    parser.add_argument(
        "--tn5-shift",
        type=int,
        default=0,
        help=(
            "Additional Tn5 shift to apply to insertion sites. "
            "Cell Ranger ATAC fragments already have the +4/-5 shift applied, "
            "so this defaults to 0. Set to 4 if your fragments are unshifted."
        ),
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load inputs
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    logger.info(f"Loaded {len(chrom_sizes)} chromosomes from {args.chrom_sizes}")

    barcode_groups = load_barcode_groups(args.h5ad, args.groupby)

    # Filter to requested groups
    if args.groups:
        barcode_groups = {
            k: v for k, v in barcode_groups.items() if k in args.groups
        }
        if not barcode_groups:
            raise ValueError(
                f"None of the requested groups {args.groups} found. "
                f"Available: {list(load_barcode_groups(args.h5ad, args.groupby).keys())}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-group BigWigs
    for group_name, barcodes in sorted(barcode_groups.items()):
        safe_name = _sanitize_filename(group_name)
        output_path = args.output_dir / f"{safe_name}.bw"
        if output_path.exists():
            logger.info(f"Skipping {group_name} (already exists: {output_path})")
            continue
        logger.info(f"Generating BigWig for '{group_name}' ({len(barcodes)} cells)...")
        generate_pseudobulk_bigwig(
            fragments_path=args.fragments,
            chrom_sizes=chrom_sizes,
            barcodes=barcodes,
            output_path=output_path,
            mode=args.mode,
            normalization=args.normalization,
            min_fragsize=args.min_fragsize,
            max_fragsize=args.max_fragsize,
            tn5_shift=args.tn5_shift,
        )

    # Optionally generate all-cells BigWig
    if args.all_cells:
        all_barcodes: set[str] = set()
        for barcodes in barcode_groups.values():
            all_barcodes |= barcodes
        output_path = args.output_dir / "all_cells.bw"
        if output_path.exists():
            logger.info(f"Skipping all_cells (already exists: {output_path})")
        else:
            logger.info(f"Generating BigWig for all cells ({len(all_barcodes)} cells)...")
            generate_pseudobulk_bigwig(
                fragments_path=args.fragments,
                chrom_sizes=chrom_sizes,
                barcodes=all_barcodes,
                output_path=output_path,
                mode=args.mode,
                normalization=args.normalization,
                min_fragsize=args.min_fragsize,
                max_fragsize=args.max_fragsize,
                tn5_shift=args.tn5_shift,
            )

    logger.info("All done.")


if __name__ == "__main__":
    main()
