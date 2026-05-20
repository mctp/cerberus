"""Tests for the ChromBPNet multi-task differential phase-2 trainer.

Pins the small helpers (fold-dir resolution, model.pt discovery, DDP
strategy promotion, accessibility-only export).  The end-to-end
``train_single`` call is not exercised here -- it needs a real
DataModule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.train_chrombpnet_multitask_differential import (  # noqa: E402
    _export_accessibility_checkpoints,
    _find_model_pt,
    _find_phase1_fold_dir,
    _parse_devices,
    _select_phase2_strategy,
)


# ---------------------------------------------------------------------------
# _parse_devices
# ---------------------------------------------------------------------------


def test_parse_devices_auto_passthrough():
    assert _parse_devices("auto") == "auto"


def test_parse_devices_int_string():
    assert _parse_devices("2") == 2


def test_parse_devices_passes_non_int_through():
    """Lightning accepts comma-separated GPU lists like '0,1'; the parser
    leaves anything non-integer as-is so Lightning can validate."""
    assert _parse_devices("0,1") == "0,1"


# ---------------------------------------------------------------------------
# _find_phase1_fold_dir + _find_model_pt
# ---------------------------------------------------------------------------


def test_find_phase1_fold_dir_picks_fold_subdirectory(tmp_path: Path):
    (tmp_path / "fold_0").mkdir()
    assert _find_phase1_fold_dir(tmp_path, 0) == tmp_path / "fold_0"


def test_find_phase1_fold_dir_accepts_flat_checkpoint(tmp_path: Path):
    """When the checkpoint dir already contains model.pt directly, treat
    it as the fold dir (single-fold flat layout)."""
    (tmp_path / "model.pt").write_bytes(b"")
    assert _find_phase1_fold_dir(tmp_path, 0) == tmp_path


def test_find_phase1_fold_dir_raises_when_neither_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fold_0"):
        _find_phase1_fold_dir(tmp_path, 0)


def test_find_model_pt_prefers_top_level(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"top")
    (tmp_path / "lightning_logs" / "version_0").mkdir(parents=True)
    (tmp_path / "lightning_logs" / "version_0" / "model.pt").write_bytes(b"nested")
    assert _find_model_pt(tmp_path) == tmp_path / "model.pt"


def test_find_model_pt_falls_back_to_nested(tmp_path: Path):
    nested = tmp_path / "lightning_logs" / "version_0" / "model.pt"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    assert _find_model_pt(tmp_path) == nested


def test_find_model_pt_raises_when_absent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _find_model_pt(tmp_path)


# ---------------------------------------------------------------------------
# _select_phase2_strategy -- the DDP strategy promotion
# ---------------------------------------------------------------------------


def test_select_phase2_strategy_overrides_false_to_true():
    """``DifferentialCountLoss`` leaves the profile heads unused under DDP,
    which trips bucket rebuild errors unless find_unused_parameters=True."""
    out = _select_phase2_strategy({"strategy": "ddp_find_unused_parameters_false"})
    assert out["strategy"] == "ddp_find_unused_parameters_true"


def test_select_phase2_strategy_passthrough_for_other_strategies():
    """Single-GPU runs (strategy=auto) and explicitly-true configs
    pass through unchanged."""
    auto = {"strategy": "auto", "precision": "bf16-mixed"}
    assert _select_phase2_strategy(auto) == auto
    explicit_true = {"strategy": "ddp_find_unused_parameters_true"}
    assert _select_phase2_strategy(explicit_true) == explicit_true


def test_select_phase2_strategy_does_not_mutate_input():
    base = {"strategy": "ddp_find_unused_parameters_false"}
    _ = _select_phase2_strategy(base)
    assert base["strategy"] == "ddp_find_unused_parameters_false"


# ---------------------------------------------------------------------------
# _export_accessibility_checkpoints
# ---------------------------------------------------------------------------


def test_export_accessibility_checkpoints_writes_chrombpnet_wo_bias(tmp_path: Path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    full_state_dict = {
        "accessibility_model.x": torch.zeros(2),
        "bias_model.y": torch.ones(2),
        "bias_logcount_offset": torch.tensor(0.0),
    }
    torch.save(full_state_dict, fold_dir / "model.pt")

    _export_accessibility_checkpoints(tmp_path)

    out = fold_dir / "chrombpnet_wo_bias.pt"
    assert out.exists()
    acc_sd = torch.load(out, map_location="cpu", weights_only=True)
    assert set(acc_sd.keys()) == {"x"}
