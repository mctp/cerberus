"""Regression guards for tool-private helpers in
``tools/train_multitask_differential_bpnet.py``.

The tool is not importable as a package symbol, so it is loaded via
``importlib``. These tests cover invariants that would silently break
Phase 2 training if regressed — DDP strategy override and checkpoint
glob semantics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool_module():
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "train_multitask_differential_bpnet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_multitask_differential_bpnet", tool_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _select_phase2_strategy
# ---------------------------------------------------------------------------


def test_select_phase2_strategy_overrides_find_unused_false_to_true():
    """Phase 2 supervises only the count-head delta; the profile heads
    receive zero gradient. Under multi-GPU DDP this trips a bucket-rebuild
    error unless ``find_unused_parameters=True``.

    :func:`get_precision_kwargs` returns ``ddp_find_unused_parameters_false``
    for multi-GPU bf16 runs by default (the other trainers don't have
    unused parameters). Phase 2 overrides to ``_true``; regress if anyone
    drops or renames this override.
    """
    tool = _load_tool_module()
    base = {
        "precision": "bf16-mixed",
        "accelerator": "gpu",
        "devices": 4,
        "strategy": "ddp_find_unused_parameters_false",
    }
    result = tool._select_phase2_strategy(base)
    assert result["strategy"] == "ddp_find_unused_parameters_true"
    # Other keys must pass through unchanged.
    assert result["precision"] == "bf16-mixed"
    assert result["devices"] == 4


def test_select_phase2_strategy_leaves_auto_untouched():
    """Single-device runs keep ``strategy="auto"``."""
    tool = _load_tool_module()
    base = {
        "precision": "bf16-mixed",
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
    }
    result = tool._select_phase2_strategy(base)
    assert result["strategy"] == "auto"


def test_select_phase2_strategy_leaves_find_unused_true_untouched():
    """Idempotent: an already-correct strategy must pass through."""
    tool = _load_tool_module()
    base = {"strategy": "ddp_find_unused_parameters_true"}
    result = tool._select_phase2_strategy(base)
    assert result["strategy"] == "ddp_find_unused_parameters_true"


def test_select_phase2_strategy_does_not_mutate_input():
    """The helper must not mutate the caller's precision_kwargs dict."""
    tool = _load_tool_module()
    base = {"strategy": "ddp_find_unused_parameters_false", "precision": "bf16-mixed"}
    _ = tool._select_phase2_strategy(base)
    assert base["strategy"] == "ddp_find_unused_parameters_false"


# ---------------------------------------------------------------------------
# _find_phase1_checkpoint
# ---------------------------------------------------------------------------


def test_find_phase1_checkpoint_finds_nested_model_pt(tmp_path: Path):
    """``_find_phase1_checkpoint`` must locate ``model.pt`` inside a fold
    subdir (standard ``train_single`` / ``train_multi`` layout)."""
    tool = _load_tool_module()
    fold_dir = tmp_path / "fold_0" / "version_0"
    fold_dir.mkdir(parents=True)
    ckpt = fold_dir / "model.pt"
    ckpt.write_bytes(b"")
    result = tool._find_phase1_checkpoint(tmp_path)
    assert result == ckpt


def test_find_phase1_checkpoint_raises_if_missing(tmp_path: Path):
    """Clean error when no checkpoint exists under the given directory."""
    tool = _load_tool_module()
    with pytest.raises(FileNotFoundError, match="No model.pt"):
        tool._find_phase1_checkpoint(tmp_path)
