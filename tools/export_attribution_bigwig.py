#!/usr/bin/env python
"""Export attribution NPZ arrays to a genome-browser BigWig track.

This tool follows chrombpnet-style projected attribution export:
1. Compute projected attributions as ``(ohe * attr).sum(axis=1)``.
2. Map each row to its genomic interval from ``intervals.tsv``.
3. Resolve overlaps using a fixed nearest-center midpoint split strategy.
4. Write one scalar value per base to BigWig.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybigtools

import cerberus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntervalRow:
    index: int
    chrom: str
    start: int
    end: int
    strand: str
    interval_source: str

    @property
    def mid(self) -> int:
        return (self.start + self.end) // 2

    @property
    def length(self) -> int:
        return self.end - self.start


def _load_modisco_array(npz_path: Path, label: str) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"{label} NPZ not found: {npz_path}")
    with np.load(npz_path) as data:
        if "arr_0" not in data:
            raise ValueError(
                f"{label} NPZ '{npz_path}' is missing key 'arr_0'."
            )
        arr = data["arr_0"]
    if arr.ndim != 3:
        raise ValueError(f"{label} array must be 3D (N, 4, L); got shape {arr.shape}")
    return arr


def _load_chrom_sizes(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"chrom sizes file not found: {path}")
    chrom_sizes: dict[str, int] = {}
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid chrom sizes line {lineno} in {path}: {raw!r}"
                )
            chrom, size = parts[0], int(parts[1])
            chrom_sizes[chrom] = size
    if not chrom_sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return chrom_sizes


def _load_intervals_tsv(path: Path) -> list[IntervalRow]:
    if not path.exists():
        raise FileNotFoundError(f"interval TSV not found: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"index", "chrom", "start", "end", "strand", "interval_source"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must contain columns: {sorted(required)}; got {reader.fieldnames}"
            )
        rows: list[IntervalRow] = []
        for row in reader:
            idx = int(row["index"])
            start = int(row["start"])
            end = int(row["end"])
            if start >= end:
                raise ValueError(
                    f"Invalid interval with start >= end at index={idx}: {start}-{end}"
                )
            rows.append(
                IntervalRow(
                    index=idx,
                    chrom=row["chrom"],
                    start=start,
                    end=end,
                    strand=row["strand"],
                    interval_source=row["interval_source"],
                )
            )

    if not rows:
        raise ValueError(f"No interval rows found in {path}")

    rows = sorted(rows, key=lambda r: r.index)
    expected = list(range(len(rows)))
    observed = [row.index for row in rows]
    if observed != expected:
        raise ValueError(
            f"Interval indices must be contiguous [0..N-1]. "
            f"Expected {expected[:5]}... got {observed[:5]}..."
        )
    return rows


def _project_observed_base_attributions(ohe: np.ndarray, attr: np.ndarray) -> np.ndarray:
    if ohe.shape != attr.shape:
        raise ValueError(f"Shape mismatch: ohe {ohe.shape} vs attr {attr.shape}")
    if ohe.shape[1] != 4:
        raise ValueError(f"Expected 4 nucleotide channels; got shape {ohe.shape}")
    projected = (ohe * attr).sum(axis=1)
    return projected.astype(np.float32, copy=False)


def _iter_nearest_center_entries(
    projected: np.ndarray,
    intervals: list[IntervalRow],
    chrom_sizes: dict[str, int],
):
    if projected.ndim != 2:
        raise ValueError(f"Projected attribution must have shape (N, L), got {projected.shape}")
    n_rows, seq_len = projected.shape
    if n_rows != len(intervals):
        raise ValueError(
            f"Projected rows ({n_rows}) do not match interval rows ({len(intervals)})."
        )

    chrom_to_order = {chrom: i for i, chrom in enumerate(chrom_sizes.keys())}

    for row in intervals:
        if row.chrom not in chrom_to_order:
            raise ValueError(
                f"Interval chromosome '{row.chrom}' is missing from chrom sizes file."
            )
        if row.length != seq_len:
            raise ValueError(
                f"Interval length mismatch at index={row.index}: "
                f"interval length={row.length}, projected length={seq_len}."
            )
        chrom_len = chrom_sizes[row.chrom]
        if row.start < 0 or row.end > chrom_len:
            raise ValueError(
                f"Interval out of bounds for {row.chrom} (size={chrom_len}): "
                f"{row.start}-{row.end}"
            )

    sorted_idx = sorted(
        range(len(intervals)),
        key=lambda i: (chrom_to_order[intervals[i].chrom], intervals[i].start),
    )

    cur_chrom = ""
    cur_end = 0
    for rank, idx in enumerate(sorted_idx):
        region = intervals[idx]
        if region.chrom != cur_chrom:
            cur_chrom = region.chrom
            cur_end = 0

        if cur_end < region.start:
            cur_end = region.start

        if region.end < cur_end:
            continue

        next_end = region.end
        if rank + 1 < len(sorted_idx):
            next_region = intervals[sorted_idx[rank + 1]]
            if next_region.chrom == region.chrom and next_region.start < region.end:
                next_end = (region.mid + next_region.mid) // 2

        if next_end <= cur_end:
            continue

        start_offset = cur_end - region.start
        end_offset = next_end - region.start
        vals = projected[idx, start_offset:end_offset]
        if vals.shape[0] != next_end - cur_end:
            raise RuntimeError(
                f"Projected slicing mismatch for interval index={region.index}: "
                f"slice {start_offset}:{end_offset} from length {seq_len}."
            )

        pos = cur_end
        for val in vals:
            yield region.chrom, pos, pos + 1, float(val)
            pos += 1

        cur_end = next_end


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export projected attribution BigWig from ohe/attribution NPZ and "
            "aligned interval metadata TSV."
        )
    )
    parser.add_argument(
        "--ohe-path",
        type=Path,
        required=True,
        help="Path to sequence NPZ with key arr_0 and shape (N, 4, L).",
    )
    parser.add_argument(
        "--attr-path",
        type=Path,
        required=True,
        help="Path to attribution NPZ with key arr_0 and shape (N, 4, L).",
    )
    parser.add_argument(
        "--intervals-tsv",
        type=Path,
        required=True,
        help="Path to intervals TSV exported by tools/export_tfmodisco_inputs.py.",
    )
    parser.add_argument(
        "--chrom-sizes",
        type=Path,
        required=True,
        help="Tab-delimited chromosome sizes file: chrom\\tlength.",
    )
    parser.add_argument(
        "--output-bw",
        type=Path,
        required=True,
        help="Output BigWig path.",
    )
    return parser


def main() -> None:
    cerberus.setup_logging()
    args = _build_arg_parser().parse_args()

    ohe = _load_modisco_array(args.ohe_path.resolve(), "Sequence")
    attrs = _load_modisco_array(args.attr_path.resolve(), "Attribution")
    intervals = _load_intervals_tsv(args.intervals_tsv.resolve())
    chrom_sizes = _load_chrom_sizes(args.chrom_sizes.resolve())

    projected = _project_observed_base_attributions(ohe, attrs)

    interval_chroms = {row.chrom for row in intervals}
    write_chrom_sizes = {
        chrom: size for chrom, size in chrom_sizes.items() if chrom in interval_chroms
    }
    if not write_chrom_sizes:
        raise ValueError(
            "No interval chromosomes matched the provided chrom sizes file."
        )

    output_bw = args.output_bw.resolve()
    output_bw.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing BigWig with chrombpnet-style projected attribution. rows=%d, length=%d",
        projected.shape[0],
        projected.shape[1],
    )

    bw = pybigtools.open(str(output_bw), "w")  # type: ignore[attr-defined]
    try:
        bw.write(
            write_chrom_sizes,
            _iter_nearest_center_entries(projected, intervals, write_chrom_sizes),
        )
    finally:
        bw.close()

    logger.info("Saved BigWig: %s", output_bw)


if __name__ == "__main__":
    main()
