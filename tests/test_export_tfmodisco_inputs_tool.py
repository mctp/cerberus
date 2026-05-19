"""Regression guards for ``tools/export_tfmodisco_inputs.py`` CLI wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_tool_module():
    tool_path = (
        Path(__file__).resolve().parents[1] / "tools" / "export_tfmodisco_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("export_tfmodisco_inputs", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_delta_target_modes() -> None:
    tool = _load_tool_module()
    parser = tool._build_arg_parser()

    target_mode_action = next(
        action for action in parser._actions if action.dest == "target_mode"
    )
    assert "delta_log_counts" in target_mode_action.choices
    assert "delta_profile_window_sum" in target_mode_action.choices


def test_resolve_target_channels_uses_single_channel_for_absolute_modes() -> None:
    tool = _load_tool_module()
    parser = tool._build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint-dir",
            "model_dir",
            "--output-dir",
            "out_dir",
            "--target-mode",
            "log_counts",
            "--target-channel",
            "3",
            "--target-cond-a",
            "0",
            "--target-cond-b",
            "1",
        ]
    )

    assert tool._resolve_target_channels(args) == 3


def test_resolve_target_channels_uses_pair_for_delta_modes() -> None:
    tool = _load_tool_module()
    parser = tool._build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint-dir",
            "model_dir",
            "--output-dir",
            "out_dir",
            "--target-mode",
            "delta_log_counts",
            "--target-channel",
            "9",
            "--target-cond-a",
            "2",
            "--target-cond-b",
            "4",
        ]
    )

    assert tool._resolve_target_channels(args) == (2, 4)
