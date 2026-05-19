"""Tests for the shared scale-aware-pseudocount CLI helper.

Covers ``tools/_pseudocount_cli.py``: argparse registration, default
values, the legacy-pseudocount branch, and the reads-equivalent override
branch (including the CPM ``--total-reads`` requirement).
"""

from __future__ import annotations

import argparse

import pytest

from tools._pseudocount_cli import (
    add_pseudocount_cli_args,
    resolve_count_pseudocount_from_args,
)


def _parser(default_count_pseudocount: float = 1.0) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_pseudocount_cli_args(parser, default_count_pseudocount=default_count_pseudocount)
    return parser


# ---------------------------------------------------------------------------
# add_pseudocount_cli_args
# ---------------------------------------------------------------------------


def test_registers_all_five_flags_with_expected_defaults():
    ns = _parser(default_count_pseudocount=1.0).parse_args([])
    assert ns.count_pseudocount == 1.0
    assert ns.pseudocount_reads is None
    assert ns.read_length == 150
    assert ns.input_scale == "raw"
    assert ns.total_reads is None


def test_default_count_pseudocount_is_honored():
    """Each tool passes its own legacy default; the helper must respect it."""
    ns = _parser(default_count_pseudocount=150.0).parse_args([])
    assert ns.count_pseudocount == 150.0


def test_input_scale_choices_reject_unknown():
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--input-scale", "rpkm"])


# ---------------------------------------------------------------------------
# resolve_count_pseudocount_from_args -- legacy branch
# ---------------------------------------------------------------------------


def test_resolve_legacy_branch_applies_target_scale():
    """Default path: pc = --count-pseudocount * target_scale."""
    ns = _parser(default_count_pseudocount=42.0).parse_args([])
    result = resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=0.01)
    assert result == pytest.approx(0.42)


def test_resolve_legacy_branch_ignores_read_length_and_input_scale():
    """When --pseudocount-reads is unset, the override flags are inert."""
    ns = _parser(default_count_pseudocount=2.0).parse_args(
        ["--read-length", "999", "--input-scale", "cpm"],
    )
    # No --pseudocount-reads → input-scale='cpm' must NOT raise even without
    # --total-reads, because the override branch is not taken.
    result = resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=0.5)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# resolve_count_pseudocount_from_args -- override branch
# ---------------------------------------------------------------------------


def test_resolve_override_branch_raw_matches_helper_formula():
    """1 read of coverage at BPNet defaults: 1 × 150 / 1 × 1.0 = 150."""
    ns = _parser().parse_args(["--pseudocount-reads", "1.0"])
    result = resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=1.0)
    assert result == pytest.approx(150.0)


def test_resolve_override_branch_respects_atac_overrides():
    """1 read at ATAC defaults (100 bp / 4 bp bins): 1 × 100 / 4 = 25."""
    ns = _parser().parse_args(
        ["--pseudocount-reads", "1.0", "--read-length", "100"],
    )
    result = resolve_count_pseudocount_from_args(ns, bin_size=4, target_scale=1.0)
    assert result == pytest.approx(25.0)


def test_resolve_override_branch_cpm_requires_total_reads():
    """CPM without --total-reads must raise (delegated to the library helper)."""
    ns = _parser().parse_args(
        ["--pseudocount-reads", "1.0", "--input-scale", "cpm"],
    )
    with pytest.raises(ValueError, match="total_reads"):
        resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=1.0)


def test_resolve_override_branch_cpm_with_total_reads_divides_by_depth():
    """CPM raw at 1.0 reads / 50M depth = raw_pc × (1e6 / 50M) = 150 × 0.02 = 3.0."""
    ns = _parser().parse_args(
        [
            "--pseudocount-reads", "1.0",
            "--input-scale", "cpm",
            "--total-reads", "50000000",
        ],
    )
    result = resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=1.0)
    assert result == pytest.approx(3.0)


def test_resolve_override_branch_takes_precedence_over_count_pseudocount():
    """When both are set, --pseudocount-reads wins."""
    ns = _parser(default_count_pseudocount=999.0).parse_args(
        ["--pseudocount-reads", "1.0", "--count-pseudocount", "0.5"],
    )
    result = resolve_count_pseudocount_from_args(ns, bin_size=1, target_scale=1.0)
    # Override branch: 1 × 150 / 1 = 150.  Legacy branch would have given
    # 0.5 × 1.0 = 0.5; the result diverges from that explicitly.
    assert result == pytest.approx(150.0)
    assert result != pytest.approx(0.5)
