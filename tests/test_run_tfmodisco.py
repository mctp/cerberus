"""Tests for the TF-MoDISco runner's new descriptive-report wiring.

We don't actually run modisco motifs / report here (those need real seqlets
and a long h5 fixture); we only pin the CLI parser surface and the
ImportError pathway for the descriptive-report function.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.run_tfmodisco import _build_arg_parser, _run_descriptive_report  # noqa: E402


# ---------------------------------------------------------------------------
# CLI parser surface
# ---------------------------------------------------------------------------


def test_parser_exposes_descriptive_report_flags():
    parser = _build_arg_parser()
    args = parser.parse_args([])
    # Default off; opt-in via --run-descriptive-report.
    assert args.run_descriptive_report is False
    # Tomtom-lite defaults to True (avoids requiring system MEME Suite).
    assert args.descriptive_report_tomtom_lite is True
    # Defaults pinned to match the legacy modisco report knobs.
    assert args.descriptive_report_top_n_matches == 3
    assert args.descriptive_report_n_examples == 10
    assert args.descriptive_report_trim_threshold == 0.3


def test_parser_run_descriptive_report_enables_flag():
    parser = _build_arg_parser()
    args = parser.parse_args(["--run-descriptive-report"])
    assert args.run_descriptive_report is True


def test_parser_descriptive_report_no_tomtom_lite():
    parser = _build_arg_parser()
    args = parser.parse_args(["--no-descriptive-report-tomtom-lite"])
    assert args.descriptive_report_tomtom_lite is False


# ---------------------------------------------------------------------------
# _run_descriptive_report -- exercise either real run or import-error path.
# We don't have a real modisco h5 fixture, so we can only check the
# error-message path when modiscolite.descriptive_report is missing.
# ---------------------------------------------------------------------------


def test_run_descriptive_report_raises_actionable_error_without_modisco(
    monkeypatch, tmp_path: Path,
):
    """If ``modiscolite.descriptive_report`` is not importable, the helper
    raises a RuntimeError naming the right PyPI package (``modisco``, not
    the historical ``modisco-lite``)."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modiscolite" or name.startswith("modiscolite."):
            raise ImportError(f"forced for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        _run_descriptive_report(
            modisco_h5=tmp_path / "modisco.h5",
            report_dir=tmp_path / "report",
            meme_db=None,
            use_tomtom_lite=True,
            top_n_matches=3,
            n_examples=10,
            trim_threshold=0.3,
        )
    msg = str(exc_info.value)
    # Mentions the right PyPI name + version requirement.
    assert "modisco" in msg
    assert "2.5.2" in msg
    # Explicitly warns about the old modisco-lite package conflict.
    assert "modisco-lite" in msg


def test_run_descriptive_report_smoke_with_real_modisco(tmp_path: Path):
    """When modiscolite.descriptive_report IS available, ensure the helper
    delegates to it rather than re-raising.  We monkey-patch the heavy
    ``generate_descriptive_report`` call to a recorder; this both verifies
    the call wiring and avoids needing a real seqlet h5.
    """
    pytest.importorskip("modiscolite.descriptive_report")
    from modiscolite import descriptive_report

    h5_path = tmp_path / "modisco.h5"
    h5_path.write_bytes(b"fake h5")  # not read; only the path is passed through
    report_dir = tmp_path / "report"

    seen: dict[str, object] = {}

    def fake_generate(
        modisco_h5,
        out_dir,
        meme_motif_db=None,
        top_n_matches=3,
        ttl=True,
        n_examples=10,
        trim_threshold=0.3,
    ):
        seen.update(dict(
            modisco_h5=modisco_h5,
            out_dir=out_dir,
            meme_motif_db=meme_motif_db,
            top_n_matches=top_n_matches,
            ttl=ttl,
            n_examples=n_examples,
            trim_threshold=trim_threshold,
        ))

    original = descriptive_report.generate_descriptive_report
    descriptive_report.generate_descriptive_report = fake_generate
    try:
        _run_descriptive_report(
            modisco_h5=h5_path,
            report_dir=report_dir,
            meme_db=None,
            use_tomtom_lite=True,
            top_n_matches=4,
            n_examples=7,
            trim_threshold=0.25,
        )
    finally:
        descriptive_report.generate_descriptive_report = original

    assert seen["modisco_h5"] == str(h5_path)
    assert seen["out_dir"] == str(report_dir)
    assert seen["meme_motif_db"] is None
    assert seen["top_n_matches"] == 4
    assert seen["n_examples"] == 7
    assert seen["trim_threshold"] == 0.25
    assert seen["ttl"] is True
    # The helper creates the report dir before delegating.
    assert report_dir.exists()
