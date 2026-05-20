"""Tests for the scATAC pseudobulk normaliser's math helpers.

Pins the Gini calculation, the constitutive-anchor fit, and the
argument-parser contract.  The full BigWig I/O pipeline is not
exercised here (it requires real BigWig fixtures), but the pure
numerical helpers are pinned in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.scatac_normalize_pseudobulk import (  # noqa: E402
    _fit_constitutive_weights,
    _gini_per_peak,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Gini coefficient (per-peak accessibility-distribution dispersion)
# ---------------------------------------------------------------------------


def test_gini_per_peak_zero_for_uniform_rows():
    """A row with equal values across all groups has Gini == 0 (perfectly
    even distribution); the constitutive-anchor heuristic uses this."""
    matrix = np.full((4, 5), 2.0)
    gini = _gini_per_peak(matrix)
    assert gini.shape == (4,)
    assert np.allclose(gini, 0.0)


def test_gini_per_peak_max_for_single_nonzero_group():
    """A row where one group carries all signal has Gini approaching
    (n-1)/n; for 5 groups that's 0.8."""
    matrix = np.zeros((1, 5))
    matrix[0, 2] = 10.0
    gini = _gini_per_peak(matrix)
    assert gini.shape == (1,)
    assert gini[0] == pytest_approx(0.8)


def test_gini_per_peak_handles_all_zero_rows():
    """Rows with no signal should not divide-by-zero; Gini is 0 there."""
    matrix = np.zeros((2, 4))
    matrix[1, 0] = 1.0
    gini = _gini_per_peak(matrix)
    assert gini[0] == pytest_approx(0.0)
    assert gini[1] > 0


def test_gini_per_peak_negative_inputs_are_clipped_to_zero():
    """Negative bin signals (numerical artefacts) are clipped before Gini
    so they don't blow up the sum; row stays effectively zero."""
    matrix = np.array([[-1.0, -2.0, -0.5]])
    gini = _gini_per_peak(matrix)
    assert gini[0] == pytest_approx(0.0)


def test_gini_per_peak_rejects_non_2d():
    import pytest
    with pytest.raises(ValueError, match="2D matrix"):
        _gini_per_peak(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Constitutive-anchor fitting (CREsted-style baseline rescaling)
# ---------------------------------------------------------------------------


def test_fit_constitutive_weights_returns_reference_unity_when_max_strategy():
    """With reference_strategy='max', the group with the highest anchor mean
    gets weight==1.0; other groups get weight = max/group_mean (>=1)."""
    # Three groups, four peaks.  Group 0 dominates: anchor means are 10, 5, 5
    # -- so group 0 is the reference at weight 1.0; groups 1 and 2 rescale 2x.
    cpm_matrix = np.array(
        [
            [10.0, 5.0, 5.0],
            [10.0, 5.0, 5.0],
            [10.0, 5.0, 5.0],
            [10.0, 5.0, 5.0],
        ]
    )
    (weights, means, peak_gini, anchors, fallback, gini_threshold) = (
        _fit_constitutive_weights(
            cpm_matrix,
            groups=["a", "b", "c"],
            top_k_percent=1.0,             # use all peaks as anchor candidates
            peak_threshold=0.0,
            gini_std_threshold=10.0,       # don't filter by Gini
            min_anchor_peaks=1,
            allow_anchor_fallback=True,
            reference_strategy="max",
            max_baseline_weight=None,
        )
    )
    assert all(a.size > 0 for a in anchors)
    assert weights[0] == pytest_approx(1.0)
    assert weights[1] == pytest_approx(2.0)
    assert weights[2] == pytest_approx(2.0)
    # Means are recoverable and the Gini per-peak vector has the right shape.
    assert means[0] == pytest_approx(10.0)
    assert peak_gini.shape == (4,)


def test_fit_constitutive_weights_caps_baseline_weight_when_requested():
    """``max_baseline_weight`` caps per-group rescaling; otherwise a very
    sparse cell-type could be rescaled by a huge factor."""
    cpm_matrix = np.array([[10.0, 0.1], [10.0, 0.1], [10.0, 0.1]])
    (weights, _means, _gini, _anchors, _fallback, _thr) = _fit_constitutive_weights(
        cpm_matrix,
        groups=["dominant", "sparse"],
        top_k_percent=1.0,
        peak_threshold=0.0,
        gini_std_threshold=10.0,
        min_anchor_peaks=1,
        allow_anchor_fallback=True,
        reference_strategy="max",
        max_baseline_weight=10.0,
    )
    # Uncapped would be 100x; cap should pull it to 10.
    assert weights[1] == pytest_approx(10.0)


# ---------------------------------------------------------------------------
# Argument parser smoke
# ---------------------------------------------------------------------------


def test_parse_args_requires_positional_dirs():
    import pytest
    with pytest.raises(SystemExit):
        _parse_args([])  # missing positional pseudobulk_dir + output_dir


def test_parse_args_defaults_round_trip(tmp_path: Path):
    """Round-trip with the minimum positional args; defaults match docs."""
    args = _parse_args([str(tmp_path / "pseudobulk"), str(tmp_path / "out")])
    assert args.input_scale == "raw"
    assert args.reference_strategy == "max"
    assert args.signal_stat == "sum"
    assert args.top_k_percent == 0.01
    assert args.gini_std_threshold == 1.0


# Tiny helper to avoid importing pytest at module top (some envs make pytest
# importorskip-style; we already import it in the test bodies that need it).
def pytest_approx(value, rel=1e-6):
    import pytest as _pytest
    return _pytest.approx(value, rel=rel)
