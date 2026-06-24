"""Tests for the from-scratch parallel differential ChromBPNet trainer."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.train_chrombpnet_multitask_differential_parallel import (  # noqa: E402
    _parse_devices,
    get_args,
)


def test_parse_devices_auto_passthrough():
    assert _parse_devices("auto") == "auto"


def test_parse_devices_int_string():
    assert _parse_devices("2") == 2


def test_parse_devices_passes_non_int_through():
    assert _parse_devices("0,1") == "0,1"


def test_parser_exposes_differential_loss_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_chrombpnet_multitask_differential_parallel.py",
            "--targets-json",
            "targets.json",
            "--output-dir",
            "out",
            "--pretrained-bias",
            "bias.pt",
            "--differential-cond-a",
            "2",
            "--differential-cond-b",
            "3",
            "--differential-pseudocount",
            "5.5",
        ],
    )

    args = get_args()

    assert args.differential_cond_a == 2
    assert args.differential_cond_b == 3
    assert args.differential_pseudocount == 5.5
