#!/usr/bin/env python3
"""
Benchmark: DataLoader throughput vs. context length.

Measures how increasing context length affects end-to-end data pipeline
throughput (multi-worker extraction + transforms + batch collation) when
reading from FASTA, BigWig, or both. Uses real GRCh38 FASTA and MDA-PCA-2B
AR ChIP-seq BigWig files via CerberusDataModule.

Usage:
    RUN_SLOW_TESTS=1 python tests/benchmark/bench_dataloader_context_length.py

    # Customize workers, batch size, batches to measure:
    RUN_SLOW_TESTS=1 python tests/benchmark/bench_dataloader_context_length.py \
        --num-workers 8 --batch-size 32 --num-batches 50

    # Override data directory:
    CERBERUS_DATA_DIR=/path/to/data RUN_SLOW_TESTS=1 python tests/benchmark/bench_dataloader_context_length.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the project root is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from torch.utils.data import DataLoader

from cerberus.config import GenomeConfig, DataConfig, SamplerConfig
from cerberus.datamodule import CerberusDataModule
from cerberus.dataset import CerberusDataset
from cerberus.download import download_dataset, download_reference_genome
from cerberus.genome import create_genome_config
from cerberus.sequence import InMemorySequenceExtractor, BaseSequenceExtractor
from cerberus.signal import InMemorySignalExtractor, BaseSignalExtractor

logger = logging.getLogger(__name__)

# Context lengths to benchmark: 1 kb to 128 kb (powers of two)
CONTEXT_LENGTHS = [1_024 * (2**i) for i in range(8)]

SCENARIOS = ["fasta_only", "bigwig_only", "fasta_bigwig"]
SCENARIO_LABELS = {"fasta_only": "fasta", "bigwig_only": "bigwig", "fasta_bigwig": "both"}

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

def make_data_config(
    scenario: str,
    input_len: int,
    max_jitter: int,
    bigwig_path: Path,
) -> DataConfig:
    """Build a DataConfig dict for the given scenario and context length."""
    if scenario == "fasta_only":
        inputs: dict[str, Path] = {}
        targets: dict[str, Path] = {"ar": bigwig_path}
        use_sequence = True
    elif scenario == "bigwig_only":
        inputs = {"ar": bigwig_path}
        targets = {"ar": bigwig_path}
        use_sequence = False
    elif scenario == "fasta_bigwig":
        inputs = {"ar": bigwig_path}
        targets = {"ar": bigwig_path}
        use_sequence = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return DataConfig(
        inputs=inputs,
        targets=targets,
        input_len=input_len,
        output_len=input_len,
        max_jitter=max_jitter,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        use_sequence=use_sequence,
        target_scale=1.0,
    )

def make_sampler_config(input_len: int, max_jitter: int, num_intervals: int) -> SamplerConfig:
    """Build a SamplerConfig dict."""
    padded_size = input_len + 2 * max_jitter
    return SamplerConfig(
        sampler_type="random",
        padded_size=padded_size,
        sampler_args={"num_intervals": num_intervals},
    )

def bench_dataloader(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    warmup_batches: int,
    seed: int,
    in_memory: bool = False,
) -> dict[str, float | int]:
    """
    Create a CerberusDataModule, iterate num_batches from train_dataloader,
    and return timing results.
    """
    dm = CerberusDataModule(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        test_fold=0,
        val_fold=1,
        seed=seed,
        pin_memory=False,
        persistent_workers=False,
    )
    dm.setup(batch_size=batch_size, num_workers=num_workers, in_memory=in_memory)
    loader = dm.train_dataloader()
    it = iter(loader)

    # Warmup (not timed) — absorbs worker spawn + lazy file handle init
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            break

    # Timed pass
    batches_consumed = 0
    t0 = time.perf_counter()
    for _ in range(num_batches):
        try:
            next(it)
            batches_consumed += 1
        except StopIteration:
            break
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": elapsed,
        "batches": batches_consumed,
        "it_per_s": batches_consumed / elapsed if elapsed > 0 else float("inf"),
    }

def format_length(bp: int) -> str:
    """Format base pairs as human-readable string (e.g. 1kb, 128kb)."""
    if bp >= 1_024:
        return f"{bp // 1_024}kb"
    return f"{bp}bp"

def bench_dataloader_inmemory(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    warmup_batches: int,
    seed: int,
    seq_extractor: BaseSequenceExtractor | None,
    sig_extractor: BaseSignalExtractor | None,
) -> dict[str, float | int]:
    """
    Build a CerberusDataset with pre-loaded in-memory extractors, wrap in a
    DataLoader, and time iteration.
    """
    use_sequence = data_config.use_sequence
    has_inputs = bool(data_config.inputs)

    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        sequence_extractor=seq_extractor if use_sequence else None,
        input_signal_extractor=sig_extractor if has_inputs else None,
        target_signal_extractor=sig_extractor,
        seed=seed,
    )
    # Split folds to get a training dataset
    train_ds, _, _ = dataset.split_folds(test_fold=0, val_fold=1)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    it = iter(loader)

    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            break

    batches_consumed = 0
    t0 = time.perf_counter()
    for _ in range(num_batches):
        try:
            next(it)
            batches_consumed += 1
        except StopIteration:
            break
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": elapsed,
        "batches": batches_consumed,
        "it_per_s": batches_consumed / elapsed if elapsed > 0 else float("inf"),
    }

def run_disk_pass(
    genome_config: GenomeConfig,
    bigwig_path: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """Run disk-backed benchmark across all context lengths and scenarios."""
    total_batches = args.warmup_batches + args.num_batches
    num_intervals = total_batches * args.batch_size * 2

    all_results: list[dict] = []

    for ctx_len in CONTEXT_LENGTHS:
        row: dict = {"context_length": ctx_len}

        for scenario in SCENARIOS:
            data_config = make_data_config(
                scenario=scenario,
                input_len=ctx_len,
                max_jitter=args.max_jitter,
                bigwig_path=bigwig_path,
            )
            sampler_config = make_sampler_config(
                input_len=ctx_len,
                max_jitter=args.max_jitter,
                num_intervals=num_intervals,
            )
            result = bench_dataloader(
                genome_config=genome_config,
                data_config=data_config,
                sampler_config=sampler_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_batches=args.num_batches,
                warmup_batches=args.warmup_batches,
                seed=args.seed,
            )
            row[scenario] = result

        all_results.append(row)

    return all_results

def run_inmemory_pass(
    genome_config: GenomeConfig,
    fasta_path: Path,
    bigwig_path: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """Run in-memory benchmark. Loads genome + BigWig once, reuses across all configs."""
    total_batches = args.warmup_batches + args.num_batches
    num_intervals = total_batches * args.batch_size * 2

    # Load once — this is the expensive part
    print("  Loading genome into memory...")
    seq_extractor = InMemorySequenceExtractor(fasta_path=fasta_path)
    print("  Loading BigWig into memory...")
    sig_extractor = InMemorySignalExtractor(bigwig_paths={"ar": bigwig_path})

    all_results: list[dict] = []

    for ctx_len in CONTEXT_LENGTHS:
        row: dict = {"context_length": ctx_len}

        for scenario in SCENARIOS:
            data_config = make_data_config(
                scenario=scenario,
                input_len=ctx_len,
                max_jitter=args.max_jitter,
                bigwig_path=bigwig_path,
            )
            sampler_config = make_sampler_config(
                input_len=ctx_len,
                max_jitter=args.max_jitter,
                num_intervals=num_intervals,
            )
            result = bench_dataloader_inmemory(
                genome_config=genome_config,
                data_config=data_config,
                sampler_config=sampler_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_batches=args.num_batches,
                warmup_batches=args.warmup_batches,
                seed=args.seed,
                seq_extractor=seq_extractor,
                sig_extractor=sig_extractor,
            )
            row[scenario] = result

        all_results.append(row)

    return all_results

def print_results_table(
    all_results: list[dict],
    label: str,
    args: argparse.Namespace,
    fasta_path: Path,
    bigwig_path: Path,
) -> None:
    """Print results table and scaling summary for one pass."""
    # Header
    header_parts = [f"{'length':>10s}"]
    for sc in SCENARIOS:
        sl = SCENARIO_LABELS[sc]
        header_parts.extend([f"{sl + '_s':>10s}", f"{sl + '_it/s':>10s}"])
    header = "  ".join(header_parts)

    print(f"\n=== {label} ===")
    print(header)
    print("-" * len(header))

    for row in all_results:
        parts = [f"{format_length(row['context_length']):>10s}"]
        for sc in SCENARIOS:
            r = row[sc]
            parts.extend([f"{r['elapsed_s']:10.3f}", f"{r['it_per_s']:10.1f}"])
        print("  ".join(parts))

    # Summary
    print(
        f"\nbatch_size={args.batch_size}  num_workers={args.num_workers}  "
        f"num_batches={args.num_batches}  warmup={args.warmup_batches}  "
        f"max_jitter={args.max_jitter}  seed={args.seed}"
    )
    print(f"FASTA: {fasta_path}")
    print(f"BigWig: {bigwig_path}")

    # Scaling table
    if len(all_results) > 1:
        print(f"\n--- Scaling relative to {format_length(CONTEXT_LENGTHS[0])} ---")
        scale_parts = [f"{'length':>10s}"]
        for sc in SCENARIOS:
            scale_parts.append(f"{SCENARIO_LABELS[sc] + '_x':>10s}")
        print("  ".join(scale_parts))

        base = all_results[0]
        for row in all_results:
            parts = [f"{format_length(row['context_length']):>10s}"]
            for sc in SCENARIOS:
                base_t = base[sc]["elapsed_s"]
                curr_t = row[sc]["elapsed_s"]
                ratio = curr_t / base_t if base_t > 0 else 0.0
                parts.append(f"{ratio:10.2f}")
            print("  ".join(parts))

def print_comparison_table(
    disk_results: list[dict],
    mem_results: list[dict],
) -> None:
    """Print side-by-side speedup table (in-memory it/s / disk it/s)."""
    print("\n=== Speedup: in-memory / disk ===")
    header_parts = [f"{'length':>10s}"]
    for sc in SCENARIOS:
        sl = SCENARIO_LABELS[sc]
        header_parts.extend([f"{sl + '_disk':>10s}", f"{sl + '_mem':>10s}", f"{sl + '_x':>8s}"])
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    for d_row, m_row in zip(disk_results, mem_results):
        parts = [f"{format_length(d_row['context_length']):>10s}"]
        for sc in SCENARIOS:
            d_its = d_row[sc]["it_per_s"]
            m_its = m_row[sc]["it_per_s"]
            speedup = m_its / d_its if d_its > 0 else float("inf")
            parts.extend([f"{d_its:10.1f}", f"{m_its:10.1f}", f"{speedup:8.2f}"])
        print("  ".join(parts))

def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark suite and print results."""
    fasta_path, bigwig_path = resolve_data_paths()

    genome_config = create_genome_config(
        name="hg38",
        fasta_path=fasta_path,
        species="human",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )

    # Disk pass
    print("Running disk-backed benchmark...")
    disk_results = run_disk_pass(genome_config, bigwig_path, args)
    print_results_table(disk_results, "disk", args, fasta_path, bigwig_path)

    if args.in_memory:
        print("\nRunning in-memory benchmark...")
        mem_results = run_inmemory_pass(genome_config, fasta_path, bigwig_path, args)
        print_results_table(mem_results, "in-memory", args, fasta_path, bigwig_path)
        print_comparison_table(disk_results, mem_results)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader throughput vs. context length"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--num-batches", type=int, default=20,
        help="Batches to time per configuration (default: 20)",
    )
    parser.add_argument(
        "--warmup-batches", type=int, default=3,
        help="Warmup batches before timing (default: 3)",
    )
    parser.add_argument(
        "--max-jitter", type=int, default=0,
        help="Max jitter in bp (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--in-memory", action="store_true",
        help="Also run in-memory benchmark and compare with disk",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if os.environ.get("RUN_SLOW_TESTS") is None:
        print(
            "This benchmark requires real genome data. "
            "Set RUN_SLOW_TESTS=1 to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_benchmark(args)

if __name__ == "__main__":
    main()
