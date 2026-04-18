"""Tests for private helpers inside tools/train_multitask_differential_bpnet.py.

The tool is not importable as a package symbol, so we load it via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import Dataset


def _load_tool_module():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "train_multitask_differential_bpnet.py"
    spec = importlib.util.spec_from_file_location("train_multitask_differential_bpnet", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {"inputs": np.zeros(4), "targets": np.zeros(4)}


def test_diff_wrapper_accepts_matching_length():
    tool = _load_tool_module()
    dataset = _StubDataset(5)
    log2fc = np.arange(5, dtype=np.float32)
    wrapper = tool._DiffWrapper(dataset, log2fc)
    assert len(wrapper) == 5
    assert wrapper[2]["log2fc"].item() == pytest.approx(2.0)


def test_diff_wrapper_rejects_length_mismatch():
    """Silent index-misaligned supervision is the footgun this guard prevents."""
    tool = _load_tool_module()
    dataset = _StubDataset(5)
    log2fc = np.arange(4, dtype=np.float32)  # one short
    with pytest.raises(ValueError, match="must equal dataset length"):
        tool._DiffWrapper(dataset, log2fc)
