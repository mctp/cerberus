#!/usr/bin/env python
"""Benchmark pseudobulk_bigwig.py to identify computational bottlenecks.

Uses test data from tests/data/dataset/kidney_scatac/ and profiles each
stage of generate_pseudobulk_bigwig independently.
"""

import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pysam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("tests/data/dataset/kidney_scatac")
FRAGMENTS = DATA_DIR / "fragments.tsv.bgz"
H5AD = DATA_DIR / "gene_activity.h5ad"
CHROM_SIZES_FILE = Path("tests/data/genome/hg38/hg38.fa.fai")

# Limit to one small chromosome for faster per-stage benchmarks
BENCH_CHROM = "chr21"  # smallest autosome


def timed(label: str):
    """Context manager that prints elapsed time."""

    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.t0
            logger.info(f"[{label}] {self.elapsed:.3f}s")

    return Timer()


def load_chrom_sizes() -> dict[str, int]:
    chrom_sizes: dict[str, int] = {}
    with open(CHROM_SIZES_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def bench_load_h5ad():
    """Benchmark loading barcode groups from h5ad."""
    with timed("load h5ad (backed=r)"):
        adata = ad.read_h5ad(H5AD, backed="r")
        groups: dict[str, set[str]] = {}
        for barcode, group in zip(adata.obs_names, adata.obs["cell_type"], strict=True):
            groups.setdefault(str(group), set()).add(str(barcode))
        adata.file.close()
    logger.info(
        f"  {sum(len(v) for v in groups.values())} barcodes, {len(groups)} groups"
    )
    return groups


def bench_tabix_fetch_count():
    """Benchmark raw tabix iteration speed (no processing)."""
    tbx = pysam.TabixFile(str(FRAGMENTS))

    with timed(f"tabix fetch + count ({BENCH_CHROM})"):
        count = 0
        for _ in tbx.fetch(BENCH_CHROM):
            count += 1
    logger.info(f"  {count:,} lines in {BENCH_CHROM}")

    with timed("tabix fetch + count (all chroms)"):
        total = 0
        for chrom in tbx.contigs:
            for _ in tbx.fetch(chrom):
                total += 1
    logger.info(f"  {total:,} total lines")

    tbx.close()
    return total


def bench_fragment_parsing(groups: dict[str, set[str]]):
    """Benchmark the fragment parsing + barcode lookup loop."""
    all_barcodes = set()
    for v in groups.values():
        all_barcodes |= v

    tbx = pysam.TabixFile(str(FRAGMENTS))
    chrom_sizes = load_chrom_sizes()
    size = chrom_sizes[BENCH_CHROM]

    # Stage 1: just string split + barcode lookup (no array writes)
    with timed(f"parse + barcode filter ({BENCH_CHROM})"):
        kept = 0
        total = 0
        for line in tbx.fetch(BENCH_CHROM):
            parts = line.split("\t")
            total += 1
            if parts[3] in all_barcodes:
                kept += 1
    logger.info(f"  {kept:,}/{total:,} fragments matched barcodes")

    # Stage 2: full processing including array writes (insertion mode)
    arr = np.zeros(size, dtype=np.float32)
    with timed(f"parse + filter + array write insertion ({BENCH_CHROM})"):
        for line in tbx.fetch(BENCH_CHROM):
            parts = line.split("\t")
            if parts[3] not in all_barcodes:
                continue
            start = int(parts[1])
            end = int(parts[2])
            count = int(parts[4]) if len(parts) > 4 else 1
            if 0 <= start < size:
                arr[start] += count
            right = end - 1
            if 0 <= right < size:
                arr[right] += count

    tbx.close()


def bench_array_allocation():
    """Benchmark allocating full-genome float32 arrays."""
    chrom_sizes = load_chrom_sizes()
    total_bp = sum(chrom_sizes.values())
    logger.info(
        f"  Total genome size: {total_bp:,} bp = {total_bp * 4 / 1e9:.2f} GB float32"
    )

    with timed("allocate all chrom arrays (float32)"):
        arrays = {}
        for chrom, size in chrom_sizes.items():
            arrays[chrom] = np.zeros(size, dtype=np.float32)

    with timed("sum all arrays"):
        total = sum(float(a.sum()) for a in arrays.values())
    logger.info(f"  Total signal: {total}")


def bench_value_stream():
    """Benchmark the BigWig value_stream generator (nonzero finding + yielding)."""
    chrom_sizes = load_chrom_sizes()

    # Create a realistic sparse array for chr21
    size = chrom_sizes[BENCH_CHROM]
    arr = np.zeros(size, dtype=np.float32)
    # Simulate ~1% coverage with random positions
    rng = np.random.default_rng(42)
    nz = rng.integers(0, size, size=size // 100)
    arr[nz] = rng.uniform(0.1, 10.0, size=len(nz)).astype(np.float32)
    nonzero_count = int(np.count_nonzero(arr))
    logger.info(f"  Simulated array: {size:,} positions, {nonzero_count:,} nonzero")

    # Original approach: yield per-position tuples
    with timed(f"value_stream per-position yield ({BENCH_CHROM})"):
        count = 0
        nonzero = np.nonzero(arr)[0]
        splits = np.where(np.diff(nonzero) > 1)[0] + 1
        for group in np.split(nonzero, splits):
            start = int(group[0])
            end = int(group[-1]) + 1
            for pos in range(start, end):
                count += 1
                _ = (BENCH_CHROM, pos, pos + 1, float(arr[pos]))
    logger.info(f"  Yielded {count:,} single-position intervals")

    # Alternative: emit contiguous intervals (bedGraph-style)
    with timed(f"value_stream bedgraph intervals ({BENCH_CHROM})"):
        count = 0
        nonzero = np.nonzero(arr)[0]
        if len(nonzero) > 0:
            splits = np.where(np.diff(nonzero) > 1)[0] + 1
            for group in np.split(nonzero, splits):
                start = int(group[0])
                end = int(group[-1]) + 1
                vals = arr[start:end]
                # Find sub-runs with same value
                changes = np.where(np.diff(vals) != 0)[0] + 1
                prev = 0
                for ch in changes:
                    count += 1
                    _ = (BENCH_CHROM, start + prev, start + ch, float(vals[prev]))
                    prev = ch
                count += 1
                _ = (BENCH_CHROM, start + prev, end, float(vals[prev]))
    logger.info(f"  Emitted {count:,} bedGraph intervals")


def bench_multi_group_overhead(groups: dict[str, set[str]]):
    """Benchmark the cost of re-reading fragments for each group."""
    tbx = pysam.TabixFile(str(FRAGMENTS))

    # Time reading chr21 once
    with timed(f"single read of {BENCH_CHROM}"):
        count = 0
        for _ in tbx.fetch(BENCH_CHROM):
            count += 1

    n_groups = len(groups)
    logger.info(
        f"  With {n_groups} groups, the fragment file is read {n_groups}x. "
        f"Estimated I/O overhead: {n_groups}x the single-read time."
    )
    tbx.close()


def main():
    logger.info("=" * 60)
    logger.info("Pseudobulk BigWig Benchmark")
    logger.info("=" * 60)

    logger.info("\n--- Stage 1: Load h5ad barcode groups ---")
    groups = bench_load_h5ad()

    logger.info("\n--- Stage 2: Raw tabix iteration speed ---")
    bench_tabix_fetch_count()

    logger.info("\n--- Stage 3: Fragment parsing + barcode filter ---")
    bench_fragment_parsing(groups)

    logger.info("\n--- Stage 4: Array allocation ---")
    bench_array_allocation()

    logger.info("\n--- Stage 5: BigWig value_stream ---")
    bench_value_stream()

    logger.info("\n--- Stage 6: Multi-group re-read overhead ---")
    bench_multi_group_overhead(groups)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY OF BOTTLENECKS")
    logger.info("=" * 60)
    logger.info(
        "1. FRAGMENT ITERATION: Reading 389M lines through pysam in Python is the "
        "dominant cost. Each group re-reads the entire file."
    )
    logger.info(
        "2. MULTI-GROUP RE-READ: With N groups, the file is read N times. "
        "A single-pass approach accumulating all groups simultaneously would "
        "reduce I/O by ~Nx."
    )
    logger.info(
        "3. VALUE_STREAM PER-POSITION YIELD: Yielding one tuple per nonzero "
        "position is slow. Merging into bedGraph intervals would reduce "
        "Python overhead."
    )
    logger.info(
        "4. ARRAY ALLOCATION: Full-genome float32 arrays use ~12 GB for hg38. "
        "With N groups this becomes ~12*N GB. Consider sparse or per-chrom "
        "streaming."
    )


if __name__ == "__main__":
    main()
