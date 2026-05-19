"""Tests for the ChromBPNet stage-2 trainer's reporting helpers.

Covers the CLI-construction logic in
``tools.train_chrombpnet._run_prediction_evaluation`` without actually
invoking ``export_predictions.py`` -- subprocess.run is patched so the
test asserts the constructed argv rather than running the subprocess.
"""

import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

# The trainer uses a sibling ``from _pseudocount_cli import ...`` that
# resolves naturally when the file is run as a script (sys.path[0] is
# ``tools/``).  When pytest imports it as ``tools.train_chrombpnet`` the
# sibling lookup fails, so explicitly add ``tools/`` to sys.path before
# the import.
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.train_chrombpnet import (  # noqa: E402
    _export_accessibility_checkpoints,
    _run_prediction_evaluation,
)


def _predict_eval_args(**overrides):
    args = Namespace(
        peaks="peaks.bed.gz",
        bigwig="signal.bw",
        batch_size=32,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_run_prediction_evaluation_defaults_include_background(tmp_path: Path):
    args = _predict_eval_args()

    with patch("tools.train_chrombpnet.subprocess.run") as run:
        _run_prediction_evaluation(args, tmp_path / "single-fold")

    cmd = run.call_args.args[0]
    assert run.call_args.kwargs == {"check": True}
    assert cmd[1].endswith("export_predictions.py")
    assert cmd[2] == str(tmp_path / "single-fold")
    assert "--include-background" in cmd
    assert "--background-ratio" not in cmd
    assert cmd[cmd.index("--batch_size") + 1] == "128"
    assert cmd[cmd.index("--eval-split") + 1] == "test"
    assert cmd[cmd.index("--use_folds") + 1] == "test"
    assert cmd[cmd.index("--seed") + 1] == "1234"
    assert Path(cmd[cmd.index("--output") + 1]).name == "predictions.tsv.gz"


def test_export_accessibility_checkpoints_strips_to_acc_only(tmp_path: Path):
    """``chrombpnet_wo_bias.pt`` contains only ``accessibility_model.*`` keys
    (with the prefix stripped) -- ``bias_model.*`` and
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
