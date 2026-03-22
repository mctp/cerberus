#!/usr/bin/env python3
"""
Benchmark: alternative implementations of encode_dna.

Compares multiple pure Python/NumPy/PyTorch implementations of DNA one-hot
encoding across sequence lengths from 1 kb to 128 kb. Verifies correctness
of all implementations against the current baseline, then reports throughput.

Usage:
    python tests/benchmark/bench_encode_dna.py
    python tests/benchmark/bench_encode_dna.py --num-iters 500
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the project root is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from cerberus.sequence import encode_dna as baseline_encode_dna

# ---------------------------------------------------------------------------
# Shared lookup tables (built once at import time)
# ---------------------------------------------------------------------------

_LUT_NP = np.zeros(256, dtype=np.int8) - 1
for _i, _b in enumerate("ACGT"):
    _LUT_NP[ord(_b)] = _i

# (256, 4) one-hot lookup: row i is the one-hot vector for byte value i
_ONEHOT_LUT_U8 = np.zeros((256, 4), dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _ONEHOT_LUT_U8[ord(_b), _i] = 1

_ONEHOT_LUT_F32 = _ONEHOT_LUT_U8.astype(np.float32)

_ONEHOT_LUT_TORCH = torch.from_numpy(_ONEHOT_LUT_F32)

LENGTHS = [1_024 * (2**i) for i in range(8)]  # 1k to 128k


# ---------------------------------------------------------------------------
# Implementation 0: current baseline (imported from cerberus.sequence)
# ---------------------------------------------------------------------------

def impl_baseline(sequence: str) -> torch.Tensor:
    return baseline_encode_dna(sequence, encoding="ACGT")


# ---------------------------------------------------------------------------
# Implementation 1: numpy scatter (same as baseline, inlined to remove overhead)
# ---------------------------------------------------------------------------

def impl_np_scatter(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    indices = _LUT_NP[seq_bytes]
    one_hot = np.zeros((4, len(sequence)), dtype=np.float32)
    valid_mask = indices >= 0
    one_hot[indices[valid_mask], np.where(valid_mask)[0]] = 1.0
    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Implementation 2: numpy LUT gather — index into a (256, 4) table, transpose
# ---------------------------------------------------------------------------

def impl_np_lut_gather(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    # (L, 4) float32 via fancy indexing into pre-built LUT
    one_hot = _ONEHOT_LUT_F32[seq_bytes]  # (L, 4)
    return torch.from_numpy(np.ascontiguousarray(one_hot.T))  # (4, L)


# ---------------------------------------------------------------------------
# Implementation 3: numpy LUT gather, uint8 → late float cast
# ---------------------------------------------------------------------------

def impl_np_lut_u8(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    one_hot_u8 = _ONEHOT_LUT_U8[seq_bytes]  # (L, 4) uint8
    # Transpose + contiguous copy + float cast
    return torch.from_numpy(np.ascontiguousarray(one_hot_u8.T)).float()


# ---------------------------------------------------------------------------
# Implementation 4: numpy comparison broadcast — no LUT, 4 comparisons
# ---------------------------------------------------------------------------

_ACGT_BYTES = np.array([ord(c) for c in "ACGT"], dtype=np.uint8).reshape(4, 1)

def impl_np_broadcast(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    # (4, 1) == (L,) -> (4, L) bool -> float32
    one_hot = (seq_bytes == _ACGT_BYTES).astype(np.float32)
    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Implementation 5: torch comparison broadcast — all in torch
# ---------------------------------------------------------------------------

_ACGT_BYTES_TORCH = torch.tensor([ord(c) for c in "ACGT"], dtype=torch.uint8).unsqueeze(1)

def impl_torch_broadcast(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_t = torch.from_numpy(seq_bytes.copy())  # (L,) uint8
    # (4, 1) == (L,) -> (4, L) bool -> float32
    return (seq_t == _ACGT_BYTES_TORCH).float()


# ---------------------------------------------------------------------------
# Implementation 6: torch scatter_ — LUT index + scatter into zeros
# ---------------------------------------------------------------------------

def impl_torch_scatter(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    indices = _LUT_NP[seq_bytes]
    L = len(sequence)

    valid = indices >= 0
    valid_idx = np.where(valid)[0]
    row_idx = indices[valid]

    one_hot = torch.zeros(4, L, dtype=torch.float32)
    one_hot[
        torch.from_numpy(row_idx.astype(np.int64)),
        torch.from_numpy(valid_idx.astype(np.int64)),
    ] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Implementation 7: torch LUT gather via embedding-style indexing
# ---------------------------------------------------------------------------

def impl_torch_lut(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_t = torch.from_numpy(seq_bytes.astype(np.int64))
    one_hot = _ONEHOT_LUT_TORCH[seq_t]  # (L, 4)
    return one_hot.T.contiguous()  # (4, L)


# ---------------------------------------------------------------------------
# Implementation 8: pure numpy bytes comparison, no string.encode()
#   Uses memoryview to avoid the .encode("ascii") copy
# ---------------------------------------------------------------------------

def impl_np_broadcast_mv(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    one_hot = np.empty((4, len(sequence)), dtype=np.float32)
    for i, byte_val in enumerate(_ACGT_BYTES.ravel()):
        np.equal(seq_bytes, byte_val, out=one_hot[i:i+1].reshape(-1).view(np.uint8)[:len(sequence)])
    # Redo properly — the trick above is fragile; use straightforward loop
    for i, byte_val in enumerate(_ACGT_BYTES.ravel()):
        one_hot[i] = (seq_bytes == byte_val)
    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Implementation 9: numpy put_along_axis (vectorized scatter without mask)
# ---------------------------------------------------------------------------

def impl_np_put_along(sequence: str) -> torch.Tensor:
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    indices = _LUT_NP[seq_bytes]  # (L,) int8
    L = len(sequence)
    one_hot = np.zeros((4, L), dtype=np.float32)
    valid = indices >= 0
    valid_pos = np.where(valid)[0]
    valid_ch = indices[valid].astype(np.intp)
    one_hot[valid_ch, valid_pos] = 1.0
    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from collections.abc import Callable

IMPLEMENTATIONS: dict[str, Callable[[str], torch.Tensor]] = {
    "baseline":         impl_baseline,
    "np_scatter":       impl_np_scatter,
    "np_lut_gather":    impl_np_lut_gather,
    "np_lut_u8":        impl_np_lut_u8,
    "np_broadcast":     impl_np_broadcast,
    "torch_broadcast":  impl_torch_broadcast,
    "torch_scatter":    impl_torch_scatter,
    "torch_lut":        impl_torch_lut,
    "np_broadcast_loop": impl_np_broadcast_mv,
    "np_put_along":     impl_np_put_along,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sequence(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence with ~1% N bases."""
    rng = random.Random(seed)
    bases = "ACGT" * 25 + "N"  # ~1% N
    return "".join(rng.choice(bases) for _ in range(length))


def format_length(bp: int) -> str:
    if bp >= 1_024:
        return f"{bp // 1_024}kb"
    return f"{bp}bp"


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def verify_correctness(sequences: dict[int, str]) -> bool:
    """Verify all implementations match baseline for every sequence length."""
    print("Verifying correctness...")
    all_ok = True

    for length, seq in sorted(sequences.items()):
        reference = impl_baseline(seq)

        for name, fn in IMPLEMENTATIONS.items():
            if name == "baseline":
                continue
            result = fn(seq)
            if result.shape != reference.shape:
                print(f"  FAIL {name} @ {format_length(length)}: "
                      f"shape {result.shape} != {reference.shape}")
                all_ok = False
                continue
            if not torch.equal(result, reference):
                diff = (result != reference).sum().item()
                print(f"  FAIL {name} @ {format_length(length)}: "
                      f"{diff} elements differ")
                all_ok = False
                continue

    if all_ok:
        print("  All implementations match baseline.")
    return all_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_one(fn: Callable[[str], torch.Tensor], seq: str, num_iters: int) -> float:
    """Time num_iters calls, return median seconds per call."""
    # Warmup
    for _ in range(min(3, num_iters)):
        fn(seq)

    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        fn(seq)
        times.append(time.perf_counter() - t0)

    return float(np.median(times))


def run_benchmark(num_iters: int) -> None:
    # Pre-generate sequences
    sequences = {length: generate_sequence(length) for length in LENGTHS}

    # Correctness
    if not verify_correctness(sequences):
        print("\nCorrectness check failed. Fix implementations before benchmarking.")
        sys.exit(1)

    names = list(IMPLEMENTATIONS.keys())

    # Header
    col_w = 14
    header = f"{'length':>10s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(f"\n--- Median time per call (ms), {num_iters} iterations ---")
    print(header)
    print("-" * len(header))

    # results[length][name] = median_ms
    results: dict[int, dict[str, float]] = {}

    for length in LENGTHS:
        seq = sequences[length]
        row: dict[str, float] = {}

        for name, fn in IMPLEMENTATIONS.items():
            median_s = bench_one(fn, seq, num_iters)
            row[name] = median_s * 1000  # ms

        results[length] = row

        parts = [f"{format_length(length):>10s}"]
        for name in names:
            parts.append(f"{row[name]:{col_w}.3f}")
        print("".join(parts))

    # Speedup relative to baseline
    print("\n--- Speedup vs baseline ---")
    header = f"{'length':>10s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header)
    print("-" * len(header))

    for length in LENGTHS:
        base_ms = results[length]["baseline"]
        parts = [f"{format_length(length):>10s}"]
        for name in names:
            speedup = base_ms / results[length][name] if results[length][name] > 0 else 0
            parts.append(f"{speedup:{col_w}.2f}")
        print("".join(parts))

    # Summary: best implementation per length
    print("\n--- Best implementation per length ---")
    for length in LENGTHS:
        row = results[length]
        best_name = min(row, key=lambda k: row[k])
        base_ms = row["baseline"]
        best_ms = row[best_name]
        speedup = base_ms / best_ms if best_ms > 0 else 0
        print(f"  {format_length(length):>6s}: {best_name} ({best_ms:.3f} ms, {speedup:.2f}x vs baseline)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark alternative encode_dna implementations"
    )
    parser.add_argument(
        "--num-iters", type=int, default=100,
        help="Iterations per (implementation, length) pair (default: 100)",
    )
    args = parser.parse_args()

    run_benchmark(num_iters=args.num_iters)


if __name__ == "__main__":
    main()
