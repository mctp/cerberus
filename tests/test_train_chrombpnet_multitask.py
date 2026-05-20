"""Tests for the multi-task ChromBPNet stage-2 trainer.

Covers the targets.json loader, peak-merging helper, and the
accessibility-only checkpoint export.  The full ``train_single`` call
is not exercised here -- that requires a real DataModule.  Argv-style
guards (constructed subprocess args) are pinned in
``tests/test_chrombpnet_reporting.py``'s style for the single-task
trainer; the multi-task trainer does not run subprocesses post-train,
so it has no argv-construction surface to guard.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest
import torch

# Sibling-import shim: same pattern as test_chrombpnet_reporting.py so the
# tool's `from _pseudocount_cli import ...` resolves under pytest's
# importlib-style loading.
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.train_chrombpnet_multitask import (  # noqa: E402
    _export_accessibility_checkpoints,
    _load_targets_json,
    _merge_peaks,
    _sanitize_channel_name,
)


# ---------------------------------------------------------------------------
# _load_targets_json -- the JSON spec loader
# ---------------------------------------------------------------------------


def test_load_targets_json_sanitises_and_sorts(tmp_path: Path):
    """Spec keys sort + whitespace is replaced by underscores; peak paths
    follow the same order as the (sorted) target keys."""
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({
        "task two": {"bigwig": "two.bw", "peaks": "two.bed"},
        "task one": {"bigwig": "one.bw", "peaks": "one.bed"},
    }))

    targets, peaks = _load_targets_json(spec_path)

    assert list(targets) == ["task_one", "task_two"]
    assert targets == {"task_one": "one.bw", "task_two": "two.bw"}
    assert peaks == ["one.bed", "two.bed"]


def test_load_targets_json_accepts_shorthand_bigwig_string(tmp_path: Path):
    """Shorthand ``{name: "path.bw"}`` is accepted; no per-task peaks are
    emitted (caller must supply --peaks)."""
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({"a": "a.bw", "b": "b.bw"}))
    targets, peaks = _load_targets_json(spec_path)
    assert targets == {"a": "a.bw", "b": "b.bw"}
    assert peaks == []


def test_load_targets_json_requires_two_targets(tmp_path: Path):
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({"only": "only.bw"}))
    with pytest.raises(ValueError, match="at least two targets"):
        _load_targets_json(spec_path)


def test_load_targets_json_rejects_empty(tmp_path: Path):
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="non-empty object"):
        _load_targets_json(spec_path)


def test_load_targets_json_rejects_duplicate_sanitised_names(tmp_path: Path):
    """Two raw names that sanitise to the same channel name (e.g.
    ``"task one"`` and ``"task_one"``) raise rather than silently
    overwriting each other."""
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({
        "task one": {"bigwig": "a.bw"},
        "task_one": {"bigwig": "b.bw"},
    }))
    with pytest.raises(ValueError, match="Duplicate sanitised channel name"):
        _load_targets_json(spec_path)


def test_load_targets_json_rejects_entry_without_bigwig(tmp_path: Path):
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(json.dumps({
        "a": {"bigwig": "a.bw"},
        "b": {"peaks": "b.bed"},  # missing 'bigwig' key
    }))
    with pytest.raises(ValueError, match="'bigwig' key"):
        _load_targets_json(spec_path)


def test_sanitize_channel_name_replaces_spaces():
    assert _sanitize_channel_name("two words") == "two_words"
    assert _sanitize_channel_name("ok") == "ok"


# ---------------------------------------------------------------------------
# _merge_peaks -- concatenates BED / BED.gz files
# ---------------------------------------------------------------------------


def test_merge_peaks_concatenates_plain_and_gzip_inputs(tmp_path: Path):
    """Plain ``.bed`` and gzipped ``.bed.gz`` both read cleanly into
    the merged BED, preserving line order across input files."""
    peak_a = tmp_path / "a.bed.gz"
    peak_b = tmp_path / "b.bed"
    with gzip.open(peak_a, "wt") as handle:
        handle.write("chr1\t0\t10\n")
    peak_b.write_text("chr2\t20\t30\n")

    merged = Path(_merge_peaks([str(peak_a), str(peak_b)], str(tmp_path)))

    assert merged.read_text() == "chr1\t0\t10\nchr2\t20\t30\n"


def test_merge_peaks_raises_on_empty_input_list(tmp_path: Path):
    with pytest.raises(ValueError, match="No peak paths"):
        _merge_peaks([], str(tmp_path))


# ---------------------------------------------------------------------------
# _export_accessibility_checkpoints
# ---------------------------------------------------------------------------


def test_export_accessibility_checkpoints_strips_to_acc_only(tmp_path: Path):
    """``chrombpnet_wo_bias.pt`` contains only ``accessibility_model.*`` keys
    (with the prefix stripped); ``bias_model.*`` and
    ``bias_logcount_offset`` are dropped from the exported file."""
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    full_state_dict = {
        "accessibility_model.iconv.weight": torch.zeros(2, 4, 3),
        "accessibility_model.profile_conv.weight": torch.ones(1, 2, 3),
        "bias_model.iconv.weight": torch.zeros(2, 4, 3),
        "bias_logcount_offset": torch.tensor(0.42),
    }
    torch.save(full_state_dict, fold_dir / "model.pt")

    _export_accessibility_checkpoints(tmp_path)

    out_path = fold_dir / "chrombpnet_wo_bias.pt"
    assert out_path.exists()
    acc_state_dict = torch.load(out_path, map_location="cpu", weights_only=True)
    assert set(acc_state_dict.keys()) == {"iconv.weight", "profile_conv.weight"}
    assert torch.equal(acc_state_dict["iconv.weight"], torch.zeros(2, 4, 3))
    assert torch.equal(acc_state_dict["profile_conv.weight"], torch.ones(1, 2, 3))
