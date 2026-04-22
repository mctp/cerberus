"""Tests for the reads-equivalent and quantile-based pseudocount helpers.

Covers the two helpers added to :mod:`cerberus.pseudocount`:

- :func:`resolve_reads_equivalent_pseudocount` — Rec 1 (Phase 1 / single-task).
  Converts a reads-equivalent specification into a scaled pseudocount, taking
  read length, bin size, target scale, and (for CPM) library depth into
  account.
- :func:`resolve_quantile_pseudocount` — Rec 2 (Phase 2 differential).
  Pulls the shrinkage prior from a quantile of training-region per-condition
  total counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from cerberus.pseudocount import (
    resolve_quantile_pseudocount,
    resolve_reads_equivalent_pseudocount,
)


# ---------------------------------------------------------------------------
# resolve_reads_equivalent_pseudocount
# ---------------------------------------------------------------------------


def test_reads_equiv_raw_bpnet_defaults() -> None:
    """BPNet scale: 150 bp reads × 1 bp bins × target_scale=1 → 150 per read."""
    pc = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0
    )
    assert pc == pytest.approx(150.0)


def test_reads_equiv_raw_asap_defaults() -> None:
    """ASAP scale: 100 bp fragments × 4 bp bins × target_scale=1 → 25 per read."""
    pc = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=100, bin_size=4, target_scale=1.0
    )
    assert pc == pytest.approx(25.0)


def test_reads_equiv_scales_linearly_with_reads() -> None:
    """Doubling reads_equiv must exactly double the returned pseudocount."""
    pc1 = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0
    )
    pc10 = resolve_reads_equivalent_pseudocount(
        reads_equiv=10.0, read_length=150, bin_size=1, target_scale=1.0
    )
    assert pc10 == pytest.approx(10 * pc1)


def test_reads_equiv_respects_target_scale() -> None:
    """``target_scale`` multiplies the final pseudocount."""
    base = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0
    )
    scaled = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=0.01
    )
    assert scaled == pytest.approx(0.01 * base)


def test_reads_equiv_cpm_requires_total_reads() -> None:
    """CPM input without a library size must raise."""
    with pytest.raises(ValueError, match="total_reads"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
            input_scale="cpm",
        )


def test_reads_equiv_cpm_matches_raw_divided_by_depth() -> None:
    """CPM: pc(CPM, N) × (N / 1e6) should equal raw pc — factor of 1e6/N."""
    raw_pc = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
        input_scale="raw",
    )
    total_reads = 50_000_000.0
    cpm_pc = resolve_reads_equivalent_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
        input_scale="cpm", total_reads=total_reads,
    )
    assert cpm_pc == pytest.approx(raw_pc * (1e6 / total_reads))


def test_reads_equiv_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="reads_equiv"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=0.0, read_length=150, bin_size=1, target_scale=1.0
        )
    with pytest.raises(ValueError, match="read_length"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=1.0, read_length=0, bin_size=1, target_scale=1.0
        )
    with pytest.raises(ValueError, match="bin_size"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=0, target_scale=1.0
        )
    with pytest.raises(ValueError, match="target_scale"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=0.0
        )


def test_reads_equiv_rejects_unknown_input_scale() -> None:
    with pytest.raises(ValueError, match="input_scale"):
        resolve_reads_equivalent_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
            input_scale="rpkm",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# resolve_quantile_pseudocount
# ---------------------------------------------------------------------------


class _FakeDataModule:
    """Stub that mimics :class:`CerberusDataModule.compute_count_quantile_samples`.

    :func:`resolve_quantile_pseudocount` only calls that one method, so tests
    do not need a full datamodule + genome + bigWig setup.
    """

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples
        self.calls: list[dict] = []

    def compute_count_quantile_samples(
        self,
        n_samples: int = 2000,
        per_channel: bool = True,
    ) -> np.ndarray:
        self.calls.append({"n_samples": n_samples, "per_channel": per_channel})
        return self._samples


def test_quantile_matches_numpy_quantile() -> None:
    samples = np.arange(100, dtype=np.float64)  # 0..99
    dm = _FakeDataModule(samples)
    pc = resolve_quantile_pseudocount(dm, quantile=0.10)
    # numpy's default quantile on 0..99 at 0.10 is 9.9.
    assert pc == pytest.approx(float(np.quantile(samples, 0.10)))


def test_quantile_rejects_out_of_range() -> None:
    dm = _FakeDataModule(np.arange(10.0))
    with pytest.raises(ValueError, match="quantile"):
        resolve_quantile_pseudocount(dm, quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        resolve_quantile_pseudocount(dm, quantile=1.0)


def test_quantile_raises_on_empty_samples() -> None:
    dm = _FakeDataModule(np.array([], dtype=np.float64))
    with pytest.raises(RuntimeError, match="No training regions"):
        resolve_quantile_pseudocount(dm)


def test_quantile_passes_through_n_samples_and_per_channel() -> None:
    dm = _FakeDataModule(np.arange(50.0))
    resolve_quantile_pseudocount(
        dm, quantile=0.25, n_samples=500, per_channel=False
    )
    assert dm.calls[-1] == {"n_samples": 500, "per_channel": False}


def test_quantile_returns_float() -> None:
    """``ModelConfig.count_pseudocount`` is a float; the helper must emit one."""
    dm = _FakeDataModule(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    pc = resolve_quantile_pseudocount(dm, quantile=0.5)
    assert isinstance(pc, float)


def test_quantile_is_scale_adaptive() -> None:
    """Multiplying all input counts by k must multiply the returned pc by k.

    This is what makes the helper correct across raw, CPM, and different
    library depths without the user reasoning about those explicitly.
    """
    base_samples = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    pc_base = resolve_quantile_pseudocount(_FakeDataModule(base_samples), quantile=0.25)
    pc_scaled = resolve_quantile_pseudocount(
        _FakeDataModule(base_samples * 37.0), quantile=0.25
    )
    assert pc_scaled == pytest.approx(37.0 * pc_base)
