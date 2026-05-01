from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from tools.train_chrombpnet import _run_prediction_evaluation


def _predict_eval_args(**overrides):
    args = Namespace(
        peaks="peaks.bed.gz",
        bigwig="signal.bw",
        batch_size=32,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_run_prediction_evaluation_defaults_include_background(tmp_path):
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
