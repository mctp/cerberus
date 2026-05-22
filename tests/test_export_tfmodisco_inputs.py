"""Tests for the TF-MoDISco export tool's CLI + delta-target dispatch.

The full ``_export_arrays`` call needs a real ModelEnsemble, so we only
pin the CLI parser surface and the small _resolve_target_channels helper
here.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.export_tfmodisco_inputs import (  # noqa: E402
    _DELTA_TARGET_MODES,
    _build_arg_parser,
    _resolve_chrombpnet_accessibility_model,
    _resolve_target_channels,
)


def test_delta_target_modes_constant():
    assert _DELTA_TARGET_MODES == frozenset(
        {"delta_log_counts", "delta_profile_window_sum"}
    )


def test_resolve_target_channels_returns_int_for_absolute_modes():
    args = Namespace(
        target_mode="log_counts",
        target_channel=2,
        target_cond_a=0,
        target_cond_b=1,
    )
    assert _resolve_target_channels(args) == 2


def test_resolve_target_channels_returns_pair_for_delta_modes():
    args = Namespace(
        target_mode="delta_log_counts",
        target_channel=0,
        target_cond_a=3,
        target_cond_b=5,
    )
    assert _resolve_target_channels(args) == (3, 5)


def test_resolve_target_channels_for_delta_profile_window_sum():
    args = Namespace(
        target_mode="delta_profile_window_sum",
        target_channel=0,
        target_cond_a=1,
        target_cond_b=2,
    )
    assert _resolve_target_channels(args) == (1, 2)


def test_resolve_chrombpnet_accessibility_model_accepts_current_branch_name():
    class Model:
        accessibility_model = object()

    assert _resolve_chrombpnet_accessibility_model(Model()) is Model.accessibility_model


def test_resolve_chrombpnet_accessibility_model_prefers_legacy_branch_name():
    class Model:
        chrombpnet_wo_bias = object()
        accessibility_model = object()

    assert _resolve_chrombpnet_accessibility_model(Model()) is Model.chrombpnet_wo_bias


# ---------------------------------------------------------------------------
# CLI parser surface -- regression guards for the new flags
# ---------------------------------------------------------------------------


_REQUIRED = ["--checkpoint-dir", "x", "--output-dir", "y"]


def test_parser_exposes_intervals_as_simple_sampler():
    parser = _build_arg_parser()
    args = parser.parse_args(_REQUIRED + ["--intervals-as-simple-sampler"])
    assert args.intervals_as_simple_sampler is True


def test_parser_exposes_chrombpnet_accessibility_only_default_none():
    """Default is ``None`` so the export auto-detects whether the loaded
    checkpoint has a bias-stripped accessibility branch."""
    parser = _build_arg_parser()
    args = parser.parse_args(_REQUIRED)
    assert args.chrombpnet_accessibility_only is None


def test_parser_exposes_chrombpnet_accessibility_only_explicit():
    parser = _build_arg_parser()
    args = parser.parse_args(_REQUIRED + ["--chrombpnet-accessibility-only"])
    assert args.chrombpnet_accessibility_only is True
    args = parser.parse_args(_REQUIRED + ["--no-chrombpnet-accessibility-only"])
    assert args.chrombpnet_accessibility_only is False


def test_parser_exposes_target_cond_a_and_b():
    parser = _build_arg_parser()
    args = parser.parse_args(_REQUIRED)
    assert args.target_cond_a == 0
    assert args.target_cond_b == 1
    args = parser.parse_args(_REQUIRED + [
        "--target-cond-a", "7", "--target-cond-b", "9",
    ])
    assert args.target_cond_a == 7
    assert args.target_cond_b == 9


def test_parser_target_mode_choices_include_delta_modes():
    parser = _build_arg_parser()
    args = parser.parse_args(_REQUIRED + ["--target-mode", "delta_log_counts"])
    assert args.target_mode == "delta_log_counts"
    args = parser.parse_args(_REQUIRED + ["--target-mode", "delta_profile_window_sum"])
    assert args.target_mode == "delta_profile_window_sum"
