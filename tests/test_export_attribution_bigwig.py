"""Tests for tools/export_attribution_bigwig.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pybigtools


def _load_tool_module():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "export_attribution_bigwig.py"
    spec = importlib.util.spec_from_file_location("export_attribution_bigwig", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dense_values_from_records(records, start: int, end: int) -> list[float]:
    values = [0.0] * (end - start)
    for rec_start, rec_end, val in records:
        for pos in range(rec_start, rec_end):
            values[pos - start] = float(val)
    return values


def test_project_observed_base_attributions():
    mod = _load_tool_module()

    ohe = np.zeros((1, 4, 3), dtype=np.float32)
    ohe[0, 0, 0] = 1.0  # A
    ohe[0, 1, 1] = 1.0  # C
    ohe[0, 2, 2] = 1.0  # G

    attr = np.zeros((1, 4, 3), dtype=np.float32)
    attr[0, 0, 0] = 1.0
    attr[0, 1, 1] = 20.0
    attr[0, 2, 2] = 300.0
    attr[0, 3, :] = 999.0  # Should be zeroed by one-hot projection.

    projected = mod._project_observed_base_attributions(ohe, attr)
    np.testing.assert_allclose(projected, np.array([[1.0, 20.0, 300.0]], dtype=np.float32))


def test_nearest_center_overlap_entries():
    mod = _load_tool_module()

    projected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # region 0: chr1:0-4, mid=2
            [2.0, 2.0, 2.0, 2.0],  # region 1: chr1:2-6, mid=4
        ],
        dtype=np.float32,
    )
    intervals = [
        mod.IntervalRow(0, "chr1", 0, 4, "+", "IntervalSampler"),
        mod.IntervalRow(1, "chr1", 2, 6, "+", "IntervalSampler"),
    ]
    chrom_sizes = {"chr1": 10}

    entries = list(mod._iter_nearest_center_entries(projected, intervals, chrom_sizes))
    per_base = [(start, val) for chrom, start, end, val in entries if chrom == "chr1"]

    assert per_base == [
        (0, 1.0),
        (1, 1.0),
        (2, 1.0),
        (3, 2.0),
        (4, 2.0),
        (5, 2.0),
    ]


def test_bigwig_write_projected_track(tmp_path):
    mod = _load_tool_module()

    ohe = np.zeros((2, 4, 4), dtype=np.float32)
    ohe[:, 0, :] = 1.0  # all A
    attrs = np.zeros((2, 4, 4), dtype=np.float32)
    attrs[0, 0, :] = 1.0
    attrs[1, 0, :] = 2.0

    ohe_path = tmp_path / "ohe.npz"
    attr_path = tmp_path / "shap.npz"
    np.savez_compressed(ohe_path, ohe)
    np.savez_compressed(attr_path, attrs)

    intervals_path = tmp_path / "intervals.tsv"
    intervals_path.write_text(
        "index\tchrom\tstart\tend\tstrand\tinterval_source\n"
        "0\tchr1\t0\t4\t+\tIntervalSampler\n"
        "1\tchr1\t2\t6\t+\tIntervalSampler\n"
    )
    chrom_sizes_path = tmp_path / "chrom.sizes"
    chrom_sizes_path.write_text("chr1\t10\n")

    loaded_ohe = mod._load_modisco_array(ohe_path, "Sequence")
    loaded_attr = mod._load_modisco_array(attr_path, "Attribution")
    intervals = mod._load_intervals_tsv(intervals_path)
    chrom_sizes = mod._load_chrom_sizes(chrom_sizes_path)
    projected = mod._project_observed_base_attributions(loaded_ohe, loaded_attr)

    out_bw = tmp_path / "shap.bw"
    bw = pybigtools.open(str(out_bw), "w")  # type: ignore[attr-defined]
    try:
        bw.write(
            chrom_sizes,
            mod._iter_nearest_center_entries(projected, intervals, chrom_sizes),
        )
    finally:
        bw.close()

    reader = pybigtools.open(str(out_bw))  # type: ignore[attr-defined]
    try:
        records = list(reader.records("chr1", 0, 6))
    finally:
        reader.close()
    dense = _dense_values_from_records(records, 0, 6)
    assert dense == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
