import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cerberus.config import ModelConfig
from cerberus.loss import BPAITACPoissonNLLLoss
from cerberus.output import ProfileCountOutput, get_log_count_params

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def test_bpaitac_poisson_nll_matches_reconstructed_log_rates():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    log_counts = torch.randn(2, 3)
    targets = torch.rand(2, 3, 7) * 5.0
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)

    loss_fn = BPAITACPoissonNLLLoss()
    components = loss_fn.loss_components(outputs, targets)

    log_rates = F.log_softmax(logits, dim=-1) + log_counts.unsqueeze(-1)
    expected = F.poisson_nll_loss(
        log_rates,
        targets,
        log_input=True,
        full=False,
        eps=1e-8,
        reduction="mean",
    )

    assert set(components) == {"poisson_nll_loss"}
    assert torch.allclose(components["poisson_nll_loss"], expected)
    assert torch.allclose(loss_fn(outputs, targets), expected)


def test_bpaitac_poisson_nll_requires_per_channel_counts_for_multitask():
    logits = torch.randn(2, 3, 7)
    targets = torch.rand(2, 3, 7)
    outputs = ProfileCountOutput(logits=logits, log_counts=torch.zeros(2, 1))

    with pytest.raises(ValueError, match="per-channel log_counts"):
        BPAITACPoissonNLLLoss()(outputs, targets)


def test_bpaitac_poisson_nll_validates_profile_shape():
    outputs = ProfileCountOutput(
        logits=torch.randn(2, 3, 7),
        log_counts=torch.zeros(2, 3),
    )
    targets = torch.rand(2, 3, 6)

    with pytest.raises(ValueError, match="matching shape"):
        BPAITACPoissonNLLLoss()(outputs, targets)


def test_bpaitac_poisson_nll_uses_raw_log_counts_for_prediction():
    config = ModelConfig(
        name="m",
        model_cls="torch.nn.Linear",
        loss_cls="cerberus.loss.BPAITACPoissonNLLLoss",
        loss_args={},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={},
        pretrained=[],
        count_pseudocount=123.0,
    )

    includes_pseudocount, pseudocount = get_log_count_params(config)

    assert includes_pseudocount is False
    assert pseudocount == 0.0


def test_multitask_trainer_exposes_bpaitac_pnll_loss(monkeypatch):
    from tools.train_chrombpnet_multitask import get_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_chrombpnet_multitask.py",
            "--targets-json",
            "targets.json",
            "--output-dir",
            "out",
            "--pretrained-bias",
            "bias.pt",
            "--loss",
            "bpaitac-pnll",
        ],
    )

    args = get_args()

    assert args.loss == "bpaitac-pnll"
