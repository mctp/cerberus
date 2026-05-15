"""Tests for tools/scatac_normalize_pseudobulk.py."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pybigtools


def _load_tool_module():
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "scatac_normalize_pseudobulk.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scatac_normalize_pseudobulk",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_bigwig(path: Path, entries: list[tuple[str, int, int, float]]) -> None:
    writer = pybigtools.open(str(path), "w")  # type: ignore[attr-defined]
    try:
        writer.write({"chr1": 100}, entries)
    finally:
        writer.close()


def _peak_sums(bigwig: Path, bed: Path) -> list[float]:
    reader = pybigtools.open(str(bigwig))
    try:
        return [float(v) for v in reader.average_over_bed(str(bed), stats="sum")]
    finally:
        reader.close()


def test_gini_per_peak_identifies_even_and_specific_rows() -> None:
    mod = _load_tool_module()

    matrix = np.array(
        [
            [10.0, 10.0, 10.0],
            [30.0, 0.0, 0.0],
        ]
    )

    gini = mod._gini_per_peak(matrix)

    assert gini[0] == 0.0
    assert gini[1] > gini[0]


def test_scatac_normalizer_cpm_and_constitutive_scaling(tmp_path: Path) -> None:
    mod = _load_tool_module()
    input_dir = tmp_path / "pseudobulk"
    output_dir = tmp_path / "normalized"
    input_dir.mkdir()

    # Raw peak sums before CPM:
    # A shared anchors: 10, 12 with library size 100 -> CPM anchor mean 110000.
    # B shared anchors: 2.5, 3 with library size 50 -> CPM anchor mean 55000.
    # The CREsted-style max reference should therefore give B a baseline weight of 2.
    _write_bigwig(
        input_dir / "A.bw",
        [
            ("chr1", 0, 10, 1.0),
            ("chr1", 10, 20, 1.2),
            ("chr1", 20, 30, 10.0),
        ],
    )
    _write_bigwig(
        input_dir / "B.bw",
        [
            ("chr1", 0, 10, 0.25),
            ("chr1", 10, 20, 0.3),
            ("chr1", 30, 40, 6.0),
        ],
    )
    _write_bigwig(input_dir / "bulk.bw", [("chr1", 0, 40, 1.0)])

    peaks = input_dir / "bulk_merge.narrowPeak.bed"
    peaks.write_text(
        "chr1\t0\t10\t.\t0\t.\t0\t0\t0\t5\n"
        "chr1\t10\t20\t.\t0\t.\t0\t0\t0\t5\n"
        "chr1\t20\t30\t.\t0\t.\t0\t0\t0\t5\n"
        "chr1\t30\t40\t.\t0\t.\t0\t0\t0\t5\n"
    )
    group_summary = tmp_path / "group_summary.tsv"
    group_summary.write_text("group\tatac_fragments\nA\t100\nB\t50\n")

    mod.main(
        [
            str(input_dir),
            str(output_dir),
            "--peaks",
            str(peaks),
            "--group-summary",
            str(group_summary),
            "--target-region-width",
            "0",
            "--top-k-percent",
            "1.0",
            "--min-anchor-peaks",
            "1",
            "--no-copy-peaks",
            "--no-write-targets-json",
        ]
    )

    with (output_dir / "normalization_summary.tsv").open(newline="") as handle:
        rows = {
            row["group"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }

    assert float(rows["A"]["cpm_scale"]) == 10_000.0
    assert float(rows["B"]["cpm_scale"]) == 20_000.0
    assert float(rows["A"]["baseline_weight"]) == 1.0
    assert float(rows["B"]["baseline_weight"]) == 2.0
    assert float(rows["B"]["final_scale"]) == 40_000.0

    a_sums = _peak_sums(output_dir / "A.bw", peaks)
    b_sums = _peak_sums(output_dir / "B.bw", peaks)

    assert a_sums[:2] == [100_000.0, 120_000.0]
    assert b_sums[:2] == [100_000.0, 120_000.0]
    assert not (output_dir / "bulk.bw").exists()
