#!/usr/bin/env python3
"""
Benchmark: sampler throughput vs. context length.

Measures how increasing context length affects extraction throughput when
reading from FASTA, BigWig, or both. Uses real GRCh38 FASTA and MDA-PCA-2B
AR ChIP-seq BigWig files.

Usage:
    RUN_SLOW_TESTS=1 python tests/benchmark/bench_sampler_context_length.py

    # Override data directory:
    CERBERUS_DATA_DIR=/path/to/data RUN_SLOW_TESTS=1 python tests/benchmark/bench_sampler_context_length.py

    # Change number of iterations:
    RUN_SLOW_TESTS=1 python tests/benchmark/bench_sampler_context_length.py --num-intervals 200
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from cerberus.download import download_dataset, download_reference_genome
from cerberus.genome import create_genome_config
from cerberus.interval import Interval
from cerberus.sequence import SequenceExtractor
from cerberus.signal import SignalExtractor

logger = logging.getLogger(__name__)

# Context lengths to benchmark: 1 kb to 128 kb (powers of two)
CONTEXT_LENGTHS = [1_024 * (2**i) for i in range(8)]  # 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k

DEFAULT_NUM_INTERVALS = 100
DEFAULT_SEED = 42


def resolve_data_paths() -> tuple[Path, Path]:
    """Locate or download the GRCh38 FASTA and MDA-PCA-2B AR BigWig."""
    base_dir = Path(os.environ.get("CERBERUS_DATA_DIR", "tests/data"))

    genome_dir = base_dir / "genome"
    genome_files = download_reference_genome(genome_dir, genome="hg38")
    fasta_path = genome_files["fasta"]

    dataset_dir = base_dir / "dataset"
    dataset_files = download_dataset(dataset_dir, name="mdapca2b_ar")
    bigwig_path = dataset_files["bigwig"]

    return Path(fasta_path), Path(bigwig_path)


def generate_random_intervals(
    fasta_path: Path,
    num_intervals: int,
    context_length: int,
    seed: int,
) -> list[Interval]:
    """Generate random genomic intervals of specified length on GRCh38 autosomes."""
    genome_config = create_genome_config(
        name="hg38",
        fasta_path=fasta_path,
        species="human",
    )
    # Use only autosomes for consistent benchmarking
    chrom_sizes = {
        c: s for c, s in genome_config.chrom_sizes.items()
        if c.replace("chr", "").isdigit()
    }

    rng = np.random.RandomState(seed)

    # Build weighted chromosome list
    chroms = list(chrom_sizes.keys())
    sizes = np.array([chrom_sizes[c] for c in chroms], dtype=np.int64)
    # Only chromosomes large enough for the context
    valid = sizes >= context_length
    chroms = [c for c, v in zip(chroms, valid, strict=True) if v]
    sizes = sizes[valid]
    weights = sizes / sizes.sum()

    intervals: list[Interval] = []
    for _ in range(num_intervals):
        idx = rng.choice(len(chroms), p=weights)
        chrom = chroms[idx]
        max_start = chrom_sizes[chrom] - context_length
        start = int(rng.randint(0, max_start))
        end = start + context_length
        intervals.append(Interval(chrom, start, end))

    return intervals


def bench_fasta(
    extractor: SequenceExtractor,
    intervals: list[Interval],
) -> float:
    """Benchmark FASTA extraction. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    for iv in intervals:
        extractor.extract(iv)
    return time.perf_counter() - t0


def bench_bigwig(
    extractor: SignalExtractor,
    intervals: list[Interval],
) -> float:
    """Benchmark BigWig extraction. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    for iv in intervals:
        extractor.extract(iv)
    return time.perf_counter() - t0


def bench_both(
    seq_extractor: SequenceExtractor,
    sig_extractor: SignalExtractor,
    intervals: list[Interval],
) -> float:
    """Benchmark FASTA + BigWig extraction together. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    for iv in intervals:
        seq_extractor.extract(iv)
        sig_extractor.extract(iv)
    return time.perf_counter() - t0


def format_length(bp: int) -> str:
    """Format base pairs as human-readable string (e.g. 1kb, 128kb)."""
    if bp >= 1_024:
        return f"{bp // 1_024}kb"
    return f"{bp}bp"


def run_benchmark(num_intervals: int, seed: int) -> None:
    """Run the full benchmark suite and print results."""
    fasta_path, bigwig_path = resolve_data_paths()
    logger.info("FASTA: %s", fasta_path)
    logger.info("BigWig: %s", bigwig_path)

    seq_extractor = SequenceExtractor(fasta_path)
    sig_extractor = SignalExtractor({"ar": bigwig_path})

    # Warm up extractors (trigger lazy loading)
    warmup_iv = Interval("chr1", 0, 1000)
    seq_extractor.extract(warmup_iv)
    sig_extractor.extract(warmup_iv)

    # Header
    print(
        f"\n{'length':>10s}  "
        f"{'fasta_s':>10s}  {'fasta_it/s':>10s}  {'fasta_bp/s':>12s}  "
        f"{'bigwig_s':>10s}  {'bigwig_it/s':>10s}  {'bigwig_bp/s':>12s}  "
        f"{'both_s':>10s}  {'both_it/s':>10s}  {'both_bp/s':>12s}"
    )
    print("-" * 130)

    results: list[dict] = []

    for ctx_len in CONTEXT_LENGTHS:
        intervals = generate_random_intervals(
            fasta_path, num_intervals, ctx_len, seed
        )

        t_fasta = bench_fasta(seq_extractor, intervals)
        t_bigwig = bench_bigwig(sig_extractor, intervals)
        t_both = bench_both(seq_extractor, sig_extractor, intervals)

        n = len(intervals)
        row = {
            "context_length": ctx_len,
            "num_intervals": n,
            "fasta_s": t_fasta,
            "bigwig_s": t_bigwig,
            "both_s": t_both,
        }
        results.append(row)

        def _throughput(elapsed: float, _n: int = n, _ctx_len: int = ctx_len) -> tuple[float, float]:
            its = _n / elapsed if elapsed > 0 else float("inf")
            bps = _n * _ctx_len / elapsed if elapsed > 0 else float("inf")
            return its, bps

        f_its, f_bps = _throughput(t_fasta)
        b_its, b_bps = _throughput(t_bigwig)
        a_its, a_bps = _throughput(t_both)

        print(
            f"{format_length(ctx_len):>10s}  "
            f"{t_fasta:10.3f}  {f_its:10.1f}  {f_bps:12.0f}  "
            f"{t_bigwig:10.3f}  {b_its:10.1f}  {b_bps:12.0f}  "
            f"{t_both:10.3f}  {a_its:10.1f}  {a_bps:12.0f}"
        )

    # Summary
    print(f"\n{'='*130}")
    print(f"Intervals per context length: {num_intervals}")
    print(f"Seed: {seed}")
    print(f"FASTA: {fasta_path}")
    print(f"BigWig: {bigwig_path}")

    # Scaling factors (relative to smallest context length)
    print(f"\n--- Scaling relative to {format_length(CONTEXT_LENGTHS[0])} ---")
    print(f"{'length':>10s}  {'fasta_x':>10s}  {'bigwig_x':>10s}  {'both_x':>10s}")
    base = results[0]
    for row in results:
        f_x = row["fasta_s"] / base["fasta_s"] if base["fasta_s"] > 0 else 0
        b_x = row["bigwig_s"] / base["bigwig_s"] if base["bigwig_s"] > 0 else 0
        a_x = row["both_s"] / base["both_s"] if base["both_s"] > 0 else 0
        print(
            f"{format_length(row['context_length']):>10s}  "
            f"{f_x:10.2f}  {b_x:10.2f}  {a_x:10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sampler extraction throughput vs. context length"
    )
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=DEFAULT_NUM_INTERVALS,
        help=f"Number of random intervals per context length (default: {DEFAULT_NUM_INTERVALS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if os.environ.get("RUN_SLOW_TESTS") is None:
        print(
            "This benchmark requires real genome data. "
            "Set RUN_SLOW_TESTS=1 to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_benchmark(num_intervals=args.num_intervals, seed=args.seed)


if __name__ == "__main__":
    main()
