"""Tests for cerberus.differential — log2FC computation, label loading, and index."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cerberus.differential import (
    DifferentialRecord,
    DifferentialTargetIndex,
    compute_bigwig_counts,
    compute_log2fc_cpm,
    compute_log2fc_from_bigwigs,
    load_differential_targets,
    write_differential_targets,
)
from cerberus.interval import Interval


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_tsv(tmp_path: Path, content: str, name: str = "targets.tsv") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


SAMPLE_TSV = """\
    chrom\tstart\tend\tlog2fc\tbase_mean
    chr1\t1000\t2000\t1.5\t320.0
    chr2\t500\t1500\t-0.8\t180.5
    chr3\t200\t800\t0.0\t50.0
    """


# ---------------------------------------------------------------------------
# compute_log2fc_cpm
# ---------------------------------------------------------------------------


def test_compute_log2fc_cpm_both_zero() -> None:
    """Both-zero peaks get log2FC = 0 (not ±inf) due to pseudocount."""
    a = np.array([0.0])
    b = np.array([0.0])
    result = compute_log2fc_cpm(a, b, normalize=False)
    assert result[0] == pytest.approx(0.0)


def test_compute_log2fc_cpm_equal_counts() -> None:
    """Equal counts → log2FC = 0 regardless of magnitude."""
    a = np.array([10.0, 100.0, 500.0])
    b = np.array([10.0, 100.0, 500.0])
    result = compute_log2fc_cpm(a, b, normalize=False)
    assert np.allclose(result, 0.0)


def test_compute_log2fc_cpm_doubling() -> None:
    """High-count peak where B = 2×A → log2FC close to 1.0."""
    a = np.array([1000.0])
    b = np.array([2000.0])
    result = compute_log2fc_cpm(a, b, pseudocount=1.0, normalize=False)
    # (2000+1)/(1000+1) ≈ 2.0 → log2 ≈ 1.0
    assert result[0] == pytest.approx(np.log2(2001 / 1001))


def test_compute_log2fc_cpm_low_count_shrinkage() -> None:
    """Zero-count reference peak yields finite positive log2FC (not +inf)."""
    a = np.array([0.0])
    b = np.array([5.0])
    result = compute_log2fc_cpm(a, b, pseudocount=1.0, normalize=False)
    # Without pseudocount this would be log2(5/0) = +inf.
    # With pseudocount=1 it is log2(6/1) — finite.
    assert np.isfinite(result[0])
    assert result[0] > 0.0


def test_compute_log2fc_cpm_pseudocount_shrinkage_stronger() -> None:
    """Larger pseudocount → smaller |log2FC| for low-count peaks."""
    a = np.array([0.0])
    b = np.array([5.0])
    fc1 = compute_log2fc_cpm(a, b, pseudocount=1.0, normalize=False)[0]
    fc5 = compute_log2fc_cpm(a, b, pseudocount=5.0, normalize=False)[0]
    assert abs(fc5) < abs(fc1)


def test_compute_log2fc_cpm_sign() -> None:
    """B < A → negative log2FC; B > A → positive."""
    a = np.array([100.0, 50.0])
    b = np.array([50.0, 100.0])
    result = compute_log2fc_cpm(a, b, normalize=False)
    assert result[0] < 0.0
    assert result[1] > 0.0


def test_compute_log2fc_cpm_normalize_true() -> None:
    """With normalize=True, library size differences are corrected."""
    # A has 2× the total counts of B but the same per-peak proportions.
    # After CPM normalization every peak has identical coverage → log2FC = 0.
    a = np.array([200.0, 400.0, 400.0])   # total 1000
    b = np.array([100.0, 200.0, 200.0])   # total 500, same proportions
    result = compute_log2fc_cpm(a, b, pseudocount=1.0, normalize=True)
    # CPM_a = CPM_b for all peaks → (CPM_b + 1) / (CPM_a + 1) = 1 → log2 = 0
    assert np.allclose(result, 0.0, atol=1e-6)


def test_compute_log2fc_cpm_normalize_false_raw() -> None:
    """With normalize=False the raw values are used directly."""
    a = np.array([1.0])
    b = np.array([3.0])
    result = compute_log2fc_cpm(a, b, pseudocount=1.0, normalize=False)
    expected = np.log2((3.0 + 1.0) / (1.0 + 1.0))
    assert result[0] == pytest.approx(expected)


def test_compute_log2fc_cpm_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_log2fc_cpm(np.array([1.0, 2.0]), np.array([1.0]))


def test_compute_log2fc_cpm_invalid_pseudocount_raises() -> None:
    a = np.array([1.0])
    b = np.array([1.0])
    with pytest.raises(ValueError, match="pseudocount"):
        compute_log2fc_cpm(a, b, pseudocount=0.0)


def test_compute_log2fc_cpm_zero_total_raises() -> None:
    """All-zero library with normalize=True must raise."""
    a = np.zeros(5)
    b = np.ones(5)
    with pytest.raises(ValueError, match="zero"):
        compute_log2fc_cpm(a, b, normalize=True)


def test_compute_log2fc_cpm_output_shape() -> None:
    a = np.ones(10)
    b = np.ones(10) * 2
    result = compute_log2fc_cpm(a, b, normalize=False)
    assert result.shape == (10,)


# ---------------------------------------------------------------------------
# DifferentialRecord
# ---------------------------------------------------------------------------


def test_differential_record_fields() -> None:
    r = DifferentialRecord(chrom="chr1", start=100, end=200, log2fc=1.5, base_mean=300.0)
    assert r.chrom == "chr1"
    assert r.start == 100
    assert r.end == 200
    assert r.log2fc == 1.5
    assert r.base_mean == 300.0


def test_differential_record_interval_property() -> None:
    r = DifferentialRecord(chrom="chrX", start=0, end=500, log2fc=0.3)
    iv = r.interval
    assert isinstance(iv, Interval)
    assert iv.chrom == "chrX"
    assert iv.start == 0
    assert iv.end == 500


def test_differential_record_optional_base_mean_default_none() -> None:
    r = DifferentialRecord(chrom="chr1", start=0, end=100, log2fc=0.0)
    assert r.base_mean is None


def test_differential_record_is_frozen() -> None:
    r = DifferentialRecord(chrom="chr1", start=0, end=100, log2fc=1.0)
    with pytest.raises(Exception):
        r.log2fc = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_differential_targets
# ---------------------------------------------------------------------------


def test_load_basic(tmp_path: Path) -> None:
    p = _write_tsv(tmp_path, SAMPLE_TSV)
    records = load_differential_targets(p)
    assert len(records) == 3
    assert records[0].chrom == "chr1"
    assert records[0].start == 1000
    assert records[0].end == 2000
    assert records[0].log2fc == pytest.approx(1.5)
    assert records[0].base_mean == pytest.approx(320.0)


def test_load_missing_required_column_raises(tmp_path: Path) -> None:
    bad_tsv = "chrom\tstart\tlog2fc\n"  # missing "end"
    p = _write_tsv(tmp_path, bad_tsv)
    with pytest.raises(ValueError, match="Missing required column"):
        load_differential_targets(p)


def test_load_na_base_mean(tmp_path: Path) -> None:
    tsv = "chrom\tstart\tend\tlog2fc\tbase_mean\n" "chr1\t0\t500\t1.0\tNA\n"
    p = _write_tsv(tmp_path, tsv)
    records = load_differential_targets(p)
    assert records[0].base_mean is None


def test_load_no_base_mean_column(tmp_path: Path) -> None:
    tsv = "chrom\tstart\tend\tlog2fc\n" "chr1\t0\t500\t2.0\n"
    p = _write_tsv(tmp_path, tsv)
    records = load_differential_targets(p)
    assert len(records) == 1
    assert records[0].base_mean is None


def test_load_custom_log2fc_col(tmp_path: Path) -> None:
    tsv = "chrom\tstart\tend\tlog2cpm\n" "chr1\t0\t100\t3.14\n"
    p = _write_tsv(tmp_path, tsv)
    records = load_differential_targets(p, log2fc_col="log2cpm")
    assert records[0].log2fc == pytest.approx(3.14)


def test_load_comment_lines_skipped(tmp_path: Path) -> None:
    tsv = "chrom\tstart\tend\tlog2fc\n" "# a comment\n" "chr1\t0\t100\t1.0\n"
    p = _write_tsv(tmp_path, tsv)
    records = load_differential_targets(p)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# write_differential_targets + roundtrip
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(tmp_path: Path) -> None:
    records = [
        DifferentialRecord("chr1", 1000, 2000, 1.5, base_mean=300.0),
        DifferentialRecord("chr2", 0, 500, -2.0, base_mean=None),
    ]
    p = tmp_path / "out.tsv"
    write_differential_targets(p, records)
    reloaded = load_differential_targets(p)
    assert len(reloaded) == 2
    assert reloaded[0].log2fc == pytest.approx(1.5)
    assert reloaded[0].base_mean == pytest.approx(300.0)
    assert reloaded[1].base_mean is None


def test_write_na_for_missing_base_mean(tmp_path: Path) -> None:
    records = [DifferentialRecord("chr1", 0, 100, 0.5)]
    p = tmp_path / "out.tsv"
    write_differential_targets(p, records)
    assert "NA" in p.read_text()


# ---------------------------------------------------------------------------
# DifferentialTargetIndex
# ---------------------------------------------------------------------------


def test_index_lookup_hit() -> None:
    records = [DifferentialRecord("chr1", 1000, 2000, 1.5)]
    idx = DifferentialTargetIndex(records)
    assert idx.get(Interval("chr1", 1000, 2000)) == pytest.approx(1.5)


def test_index_lookup_miss_returns_default() -> None:
    idx = DifferentialTargetIndex([], default=0.0)
    assert idx.get(Interval("chr1", 0, 500)) == pytest.approx(0.0)


def test_index_custom_default() -> None:
    idx = DifferentialTargetIndex([], default=-99.0)
    assert idx.get(Interval("chrX", 0, 1)) == pytest.approx(-99.0)


def test_index_len() -> None:
    records = [
        DifferentialRecord("chr1", 0, 100, 1.0),
        DifferentialRecord("chr2", 0, 100, 2.0),
    ]
    idx = DifferentialTargetIndex(records)
    assert len(idx) == 2


def test_index_contains() -> None:
    records = [DifferentialRecord("chr1", 0, 100, 1.0)]
    idx = DifferentialTargetIndex(records)
    assert Interval("chr1", 0, 100) in idx
    assert Interval("chr1", 0, 200) not in idx


def test_index_from_tsv(tmp_path: Path) -> None:
    p = _write_tsv(tmp_path, SAMPLE_TSV)
    idx = DifferentialTargetIndex.from_tsv(p)
    assert len(idx) == 3
    assert idx.get(Interval("chr1", 1000, 2000)) == pytest.approx(1.5)


def test_index_exact_coordinate_match() -> None:
    """Off-by-one coordinates must NOT match."""
    records = [DifferentialRecord("chr1", 1000, 2000, 1.5)]
    idx = DifferentialTargetIndex(records)
    assert idx.get(Interval("chr1", 1000, 2001)) == pytest.approx(0.0)
    assert idx.get(Interval("chr1", 1001, 2000)) == pytest.approx(0.0)


def test_index_get_wrong_type_raises() -> None:
    idx = DifferentialTargetIndex([])
    with pytest.raises(TypeError, match="Interval"):
        idx.get("chr1:0-100")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end: compute → record → write → load → index
# ---------------------------------------------------------------------------


def test_end_to_end_cpm_to_index(tmp_path: Path) -> None:
    """Full pipeline: counts → log2fc → records → TSV → index → lookup."""
    chroms = ["chr1", "chr2", "chr3"]
    starts = [0, 1000, 5000]
    ends = [500, 1500, 5500]
    counts_a = np.array([100.0, 0.0, 50.0])
    counts_b = np.array([200.0, 0.0, 50.0])

    log2fc = compute_log2fc_cpm(counts_a, counts_b, pseudocount=1.0, normalize=False)

    records = [
        DifferentialRecord(chrom=c, start=s, end=e, log2fc=float(fc))
        for c, s, e, fc in zip(chroms, starts, ends, log2fc)
    ]

    tsv = tmp_path / "targets.tsv"
    write_differential_targets(tsv, records)

    idx = DifferentialTargetIndex.from_tsv(tsv)
    assert len(idx) == 3

    # chr1: counts 100→200 with pc=1: log2(201/101) ≈ 0.993
    assert idx.get(Interval("chr1", 0, 500)) == pytest.approx(np.log2(201 / 101))
    # chr2: both zero → 0.0
    assert idx.get(Interval("chr2", 1000, 1500)) == pytest.approx(0.0)
    # chr3: equal counts → 0.0
    assert idx.get(Interval("chr3", 5000, 5500)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_bigwig_counts
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pybigtools_differential():
    with patch("cerberus.differential.pybigtools") as mock:
        yield mock


def _make_bw_mock(values_map: dict[tuple, np.ndarray]) -> MagicMock:
    """Return a mock bigwig whose .values() returns from a dict keyed by (chrom, start, end)."""
    bw = MagicMock()
    bw.values.side_effect = lambda chrom, start, end: values_map[(chrom, start, end)]
    return bw


def test_compute_bigwig_counts_basic(mock_pybigtools_differential, tmp_path: Path) -> None:
    """Sum over uniform bigwig bins matches interval length × value."""
    intervals = [Interval("chr1", 0, 5), Interval("chr2", 10, 15)]
    bw = _make_bw_mock(
        {
            ("chr1", 0, 5): np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            ("chr2", 10, 15): np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        }
    )
    mock_pybigtools_differential.open.return_value = bw

    counts = compute_bigwig_counts(tmp_path / "a.bw", intervals)

    assert counts.shape == (2,)
    assert counts[0] == pytest.approx(10.0)  # 5 × 2.0
    assert counts[1] == pytest.approx(15.0)  # 5 × 3.0


def test_compute_bigwig_counts_nan_treated_as_zero(mock_pybigtools_differential, tmp_path: Path) -> None:
    """NaN positions are ignored (nansum), not propagated."""
    intervals = [Interval("chr1", 0, 4)]
    bw = _make_bw_mock(
        {("chr1", 0, 4): np.array([1.0, float("nan"), 1.0, 1.0])}
    )
    mock_pybigtools_differential.open.return_value = bw

    counts = compute_bigwig_counts(tmp_path / "a.bw", intervals)
    assert counts[0] == pytest.approx(3.0)


def test_compute_bigwig_counts_missing_chrom_returns_zero(
    mock_pybigtools_differential, tmp_path: Path
) -> None:
    """RuntimeError (chromosome absent) returns 0 for that interval."""
    intervals = [Interval("chrUn", 0, 100)]
    bw = MagicMock()
    bw.values.side_effect = RuntimeError("chromosome not found")
    mock_pybigtools_differential.open.return_value = bw

    counts = compute_bigwig_counts(tmp_path / "a.bw", intervals)
    assert counts[0] == pytest.approx(0.0)


def test_compute_bigwig_counts_empty_intervals(mock_pybigtools_differential, tmp_path: Path) -> None:
    """Empty interval list returns empty array."""
    mock_pybigtools_differential.open.return_value = MagicMock()
    counts = compute_bigwig_counts(tmp_path / "a.bw", [])
    assert counts.shape == (0,)


def test_compute_bigwig_counts_output_dtype(mock_pybigtools_differential, tmp_path: Path) -> None:
    """Output is float64."""
    intervals = [Interval("chr1", 0, 3)]
    bw = _make_bw_mock({("chr1", 0, 3): np.array([1.0, 1.0, 1.0], dtype=np.float32)})
    mock_pybigtools_differential.open.return_value = bw
    counts = compute_bigwig_counts(tmp_path / "a.bw", intervals)
    assert counts.dtype == np.float64


# ---------------------------------------------------------------------------
# compute_log2fc_from_bigwigs
# ---------------------------------------------------------------------------


def test_compute_log2fc_from_bigwigs_doubling(mock_pybigtools_differential, tmp_path: Path) -> None:
    """B = 2×A → log2FC ≈ 1.0 for high-signal peaks."""
    intervals = [Interval("chr1", 0, 4)]
    bw_a = _make_bw_mock({("chr1", 0, 4): np.array([250.0, 250.0, 250.0, 250.0])})
    bw_b = _make_bw_mock({("chr1", 0, 4): np.array([500.0, 500.0, 500.0, 500.0])})

    def open_side_effect(path: str):
        return bw_a if "a.bw" in path else bw_b

    mock_pybigtools_differential.open.side_effect = open_side_effect

    log2fc = compute_log2fc_from_bigwigs(tmp_path / "a.bw", tmp_path / "b.bw", intervals)

    # counts_a=1000, counts_b=2000, pc=1: log2(2001/1001)
    assert log2fc[0] == pytest.approx(np.log2(2001 / 1001))


def test_compute_log2fc_from_bigwigs_equal_signal_is_zero(
    mock_pybigtools_differential, tmp_path: Path
) -> None:
    """Equal signal in A and B → log2FC = 0 for all peaks."""
    intervals = [Interval("chr1", 0, 3), Interval("chr2", 100, 103)]
    signal = np.array([10.0, 10.0, 10.0])
    bw = _make_bw_mock(
        {
            ("chr1", 0, 3): signal.copy(),
            ("chr2", 100, 103): signal.copy(),
        }
    )
    mock_pybigtools_differential.open.return_value = bw

    log2fc = compute_log2fc_from_bigwigs(tmp_path / "a.bw", tmp_path / "b.bw", intervals)

    assert np.allclose(log2fc, 0.0)


def test_compute_log2fc_from_bigwigs_uses_normalize_false(
    mock_pybigtools_differential, tmp_path: Path
) -> None:
    """Library-size normalization is NOT applied (bigwigs are pre-normalised)."""
    # A has 2× total signal across peaks but same proportions as B.
    # With normalize=True this would give log2FC=0; with normalize=False
    # it gives a non-zero value because raw sums differ.
    intervals = [Interval("chr1", 0, 2), Interval("chr2", 0, 2)]
    bw_a = _make_bw_mock(
        {
            ("chr1", 0, 2): np.array([200.0, 200.0]),  # total 400
            ("chr2", 0, 2): np.array([300.0, 300.0]),  # total 600
        }
    )
    bw_b = _make_bw_mock(
        {
            ("chr1", 0, 2): np.array([100.0, 100.0]),  # total 200
            ("chr2", 0, 2): np.array([150.0, 150.0]),  # total 300
        }
    )

    def open_side_effect(path: str):
        return bw_a if "a.bw" in path else bw_b

    mock_pybigtools_differential.open.side_effect = open_side_effect

    log2fc = compute_log2fc_from_bigwigs(tmp_path / "a.bw", tmp_path / "b.bw", intervals)

    # normalize=False: counts_a=[400,600], counts_b=[200,300]
    # log2((200+1)/(400+1)) and log2((300+1)/(600+1)) — both negative
    expected_0 = np.log2(201 / 401)
    expected_1 = np.log2(301 / 601)
    assert log2fc[0] == pytest.approx(expected_0)
    assert log2fc[1] == pytest.approx(expected_1)


def test_compute_log2fc_from_bigwigs_pseudocount(mock_pybigtools_differential, tmp_path: Path) -> None:
    """Larger pseudocount shrinks low-signal peaks more aggressively."""
    intervals = [Interval("chr1", 0, 1)]
    bw_a = _make_bw_mock({("chr1", 0, 1): np.array([0.0])})
    bw_b = _make_bw_mock({("chr1", 0, 1): np.array([5.0])})

    def open_side_effect(path: str):
        return bw_a if "a.bw" in path else bw_b

    mock_pybigtools_differential.open.side_effect = open_side_effect

    fc1 = compute_log2fc_from_bigwigs(
        tmp_path / "a.bw", tmp_path / "b.bw", intervals, pseudocount=1.0
    )
    fc5 = compute_log2fc_from_bigwigs(
        tmp_path / "a.bw", tmp_path / "b.bw", intervals, pseudocount=5.0
    )
    assert abs(fc5[0]) < abs(fc1[0])


def test_compute_log2fc_from_bigwigs_output_shape(mock_pybigtools_differential, tmp_path: Path) -> None:
    """Output shape matches number of intervals."""
    n = 7
    intervals = [Interval("chr1", i * 10, i * 10 + 5) for i in range(n)]
    vals = {("chr1", i * 10, i * 10 + 5): np.ones(5) for i in range(n)}
    bw = _make_bw_mock(vals)
    mock_pybigtools_differential.open.return_value = bw

    log2fc = compute_log2fc_from_bigwigs(tmp_path / "a.bw", tmp_path / "b.bw", intervals)
    assert log2fc.shape == (n,)


def test_compute_log2fc_from_bigwigs_end_to_end(
    mock_pybigtools_differential, tmp_path: Path
) -> None:
    """Full pipeline: bigwig sums → log2fc → records → TSV → index → lookup."""
    intervals = [Interval("chr1", 0, 4), Interval("chr2", 100, 104)]
    bw_a = _make_bw_mock(
        {
            ("chr1", 0, 4): np.array([100.0, 100.0, 100.0, 100.0]),
            ("chr2", 100, 104): np.array([50.0, 50.0, 50.0, 50.0]),
        }
    )
    bw_b = _make_bw_mock(
        {
            ("chr1", 0, 4): np.array([200.0, 200.0, 200.0, 200.0]),
            ("chr2", 100, 104): np.array([50.0, 50.0, 50.0, 50.0]),
        }
    )

    def open_side_effect(path: str):
        return bw_a if "a.bw" in path else bw_b

    mock_pybigtools_differential.open.side_effect = open_side_effect

    log2fc = compute_log2fc_from_bigwigs(tmp_path / "a.bw", tmp_path / "b.bw", intervals)

    records = [
        DifferentialRecord(chrom=iv.chrom, start=iv.start, end=iv.end, log2fc=float(fc))
        for iv, fc in zip(intervals, log2fc)
    ]
    tsv = tmp_path / "targets.tsv"
    write_differential_targets(tsv, records)

    idx = DifferentialTargetIndex.from_tsv(tsv)
    # chr1: counts_a=400, counts_b=800, pc=1: log2(801/401)
    assert idx.get(Interval("chr1", 0, 4)) == pytest.approx(np.log2(801 / 401))
    # chr2: equal counts → 0.0
    assert idx.get(Interval("chr2", 100, 104)) == pytest.approx(0.0)
