import os
import random
import sys
import time
from pathlib import Path

import pyfaidx

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cerberus.interval import Interval
from cerberus.sequence import compute_intervals_gc


def generate_random_intervals(chrom_sizes, num_intervals, width=2048, seed=42):
    rng = random.Random(seed)
    intervals = []
    chroms = list(chrom_sizes.keys())
    weights = [chrom_sizes[c] for c in chroms]

    # Pre-select chroms
    selected_chroms = rng.choices(chroms, weights=weights, k=num_intervals)

    for chrom in selected_chroms:
        size = chrom_sizes[chrom]
        if size < width:
            continue
        start = rng.randint(0, size - width)
        end = start + width
        intervals.append(Interval(chrom, start, end))

    return intervals


def main():
    fasta_path = Path("tests/data/genome/hg38/hg38.fa")
    if not fasta_path.exists():
        print(f"Error: {fasta_path} not found.")
        return

    print(f"Loading genome index from {fasta_path}...")
    fasta = pyfaidx.Fasta(str(fasta_path))
    chrom_sizes = {k: len(fasta[k]) for k in fasta.keys()}

    # Filter for standard chroms to avoid tiny scaffolds if any
    chrom_sizes = {k: v for k, v in chrom_sizes.items() if "chr" in k and "_" not in k}

    N = 1_000_000

    print(f"Generating {N} random intervals...")
    intervals = generate_random_intervals(chrom_sizes, N, width=2048)

    # Baseline: Unsorted (Random)
    random.shuffle(intervals)
    intervals_unsorted = list(intervals)  # Copy
    intervals_sorted = sorted(intervals, key=lambda x: (x.chrom, x.start))

    iterations = 3
    print(f"Running {iterations} iterations to account for warmup...")

    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")

        # Unsorted
        print("Benchmarking Unsorted (Random Access)...")
        start_time = time.time()
        _ = compute_intervals_gc(intervals_unsorted, fasta_path)
        unsorted_time = time.time() - start_time
        print(f"Unsorted time: {unsorted_time:.2f} s")

        # Sorted
        print("Benchmarking Sorted (Sequential Access)...")
        start_time = time.time()
        _ = compute_intervals_gc(intervals_sorted, fasta_path)
        sorted_time = time.time() - start_time
        print(f"Sorted time: {sorted_time:.2f} s")

        improvement = unsorted_time / sorted_time if sorted_time > 0 else 0
        print(f"Speedup: {improvement:.2f}x")


if __name__ == "__main__":
    main()
