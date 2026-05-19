"""Tests for the read-coverage and noise-floor pseudocount helpers.

Covers the two functions in :mod:`cerberus.pseudocount`:

- :func:`resolve_read_coverage_pseudocount` — for ``log(count + pc)``
  losses. Converts a read-coverage specification into a scaled
  pseudocount, taking read length, bin size, target scale, and (for
  CPM) library depth into account.
- :func:`resolve_noise_floor_pseudocount` — for log-fold-change losses
  like ``DifferentialCountLoss``. Pulls the shrinkage prior from a
  per-channel quantile of training-region totals and combines channels
  with ``max``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cerberus.pseudocount import (
    resolve_noise_floor_pseudocount,
    resolve_read_coverage_pseudocount,
)


# ---------------------------------------------------------------------------
# resolve_read_coverage_pseudocount
# ---------------------------------------------------------------------------


def test_read_coverage_raw_bpnet_defaults() -> None:
    """BPNet scale: 150 bp reads × 1 bp bins × target_scale=1 → 150 per read."""
    pc = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
    )
    assert pc == pytest.approx(150.0)


def test_read_coverage_raw_asap_defaults() -> None:
    """ASAP scale: 100 bp fragments × 4 bp bins × target_scale=1 → 25 per read."""
    pc = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=100, bin_size=4, target_scale=1.0,
    )
    assert pc == pytest.approx(25.0)


def test_read_coverage_scales_linearly_with_reads() -> None:
    """Doubling reads_equiv must exactly double the returned pseudocount."""
    pc1 = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
    )
    pc10 = resolve_read_coverage_pseudocount(
        reads_equiv=10.0, read_length=150, bin_size=1, target_scale=1.0,
    )
    assert pc10 == pytest.approx(10 * pc1)


def test_read_coverage_respects_target_scale() -> None:
    """``target_scale`` multiplies the final pseudocount."""
    base = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
    )
    scaled = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=0.01,
    )
    assert scaled == pytest.approx(0.01 * base)


def test_read_coverage_cpm_requires_total_reads() -> None:
    """CPM input without a library size must raise."""
    with pytest.raises(ValueError, match="total_reads"):
        resolve_read_coverage_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
            input_scale="cpm",
        )


def test_read_coverage_cpm_matches_raw_divided_by_depth() -> None:
    """CPM: pc(CPM, N) × (N / 1e6) should equal raw pc — factor of 1e6/N."""
    raw_pc = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
        input_scale="raw",
    )
    total_reads = 50_000_000.0
    cpm_pc = resolve_read_coverage_pseudocount(
        reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
        input_scale="cpm", total_reads=total_reads,
    )
    assert cpm_pc == pytest.approx(raw_pc * (1e6 / total_reads))


def test_read_coverage_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="reads_equiv"):
        resolve_read_coverage_pseudocount(
            reads_equiv=0.0, read_length=150, bin_size=1, target_scale=1.0,
        )
    with pytest.raises(ValueError, match="read_length"):
        resolve_read_coverage_pseudocount(
            reads_equiv=1.0, read_length=0, bin_size=1, target_scale=1.0,
        )
    with pytest.raises(ValueError, match="bin_size"):
        resolve_read_coverage_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=0, target_scale=1.0,
        )
    with pytest.raises(ValueError, match="target_scale"):
        resolve_read_coverage_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=0.0,
        )


def test_read_coverage_rejects_unknown_input_scale() -> None:
    with pytest.raises(ValueError, match="input_scale"):
        resolve_read_coverage_pseudocount(
            reads_equiv=1.0, read_length=150, bin_size=1, target_scale=1.0,
            input_scale="rpkm",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# resolve_noise_floor_pseudocount
# ---------------------------------------------------------------------------


class _FakeDataModule:
    """Stub mimicking :meth:`CerberusDataModule.compute_count_quantile_samples`.

    :func:`resolve_noise_floor_pseudocount` only calls that one method, so
    tests do not need a full datamodule + genome + bigWig setup.
    """

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples
        self.calls: list[dict] = []

    def compute_count_quantile_samples(
        self,
        n_samples: int = 2000,
        per_channel: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        self.calls.append(
            {"n_samples": n_samples, "per_channel": per_channel, "seed": seed},
        )
        return self._samples


def test_noise_floor_single_channel_matches_numpy_quantile() -> None:
    """With one channel, the result is just np.quantile(column, q)."""
    samples = np.arange(100, dtype=np.float64).reshape(-1, 1)  # (100, 1)
    dm = _FakeDataModule(samples)
    pc = resolve_noise_floor_pseudocount(dm, quantile=0.10)
    assert pc == pytest.approx(float(np.quantile(samples[:, 0], 0.10)))


def test_noise_floor_picks_max_across_per_channel_quantiles() -> None:
    """Two channels with very different scales → quantile reflects the deeper one.

    Channel 0 ranges [1, 100], channel 1 ranges [100, 10_000]. The 10th
    percentile is roughly ~10.9 for ch 0 and ~1090 for ch 1.  The helper
    must return the *max* of the two so the deeper condition's noise floor
    still gets shrunk.
    """
    rng = np.random.default_rng(0)
    ch0 = rng.uniform(1.0, 100.0, size=200)
    ch1 = rng.uniform(100.0, 10_000.0, size=200)
    samples = np.stack([ch0, ch1], axis=1)  # (200, 2)
    dm = _FakeDataModule(samples)

    pc = resolve_noise_floor_pseudocount(dm, quantile=0.10)
    q0, q1 = np.quantile(samples, 0.10, axis=0)
    assert q1 > q0  # sanity: ch 1 quantile is much larger
    assert pc == pytest.approx(float(q1))


def test_noise_floor_rejects_out_of_range_quantile() -> None:
    dm = _FakeDataModule(np.arange(10.0).reshape(-1, 1))
    with pytest.raises(ValueError, match="quantile"):
        resolve_noise_floor_pseudocount(dm, quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        resolve_noise_floor_pseudocount(dm, quantile=1.0)


def test_noise_floor_raises_on_empty_samples() -> None:
    dm = _FakeDataModule(np.empty((0, 1), dtype=np.float64))
    with pytest.raises(RuntimeError, match="No training regions"):
        resolve_noise_floor_pseudocount(dm)


def test_noise_floor_forwards_kwargs_to_datamodule() -> None:
    """n_samples and seed must reach the datamodule for reproducibility.

    per_channel is fixed to True by the helper -- it always needs the
    2D shape so it can compute per-channel quantiles.
    """
    dm = _FakeDataModule(np.arange(50.0).reshape(-1, 1))
    resolve_noise_floor_pseudocount(
        dm, quantile=0.25, n_samples=500, seed=42,
    )
    assert dm.calls[-1] == {
        "n_samples": 500,
        "per_channel": True,
        "seed": 42,
    }


def test_noise_floor_returns_float() -> None:
    """``ModelConfig.count_pseudocount`` is a float; the helper must emit one."""
    dm = _FakeDataModule(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    pc = resolve_noise_floor_pseudocount(dm, quantile=0.5)
    assert isinstance(pc, float)


def test_noise_floor_is_scale_adaptive() -> None:
    """Multiplying all input counts by k must multiply the returned pc by k.

    Quantile is a linear operator; max is too.  This is the property
    that makes the helper correct across raw, CPM, and different library
    depths without the user reasoning about those explicitly.
    """
    base = np.array([[1.0, 5.0], [2.0, 7.0], [5.0, 11.0], [10.0, 20.0],
                     [20.0, 35.0], [50.0, 60.0], [100.0, 150.0]])
    pc_base = resolve_noise_floor_pseudocount(
        _FakeDataModule(base), quantile=0.25,
    )
    pc_scaled = resolve_noise_floor_pseudocount(
        _FakeDataModule(base * 37.0), quantile=0.25,
    )
    assert pc_scaled == pytest.approx(37.0 * pc_base)
