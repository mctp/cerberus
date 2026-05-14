from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cerberus.config import PretrainedConfig
from cerberus.loss import MSEMultinomialLoss
from cerberus.models.bpnet import BPNet, MultitaskBPNetLoss
from cerberus.models.chrombpnet import ChromBPNet, MultitaskChromBPNet
from cerberus.output import ProfileCountOutput
from cerberus.pretrained import load_pretrained_weights
from tools.train_chrombpnet_multitask import _load_targets_json, _merge_peaks


def _small_bpnet_args(filters: int = 4) -> dict[str, object]:
    return {
        "filters": filters,
        "n_dilated_layers": 1,
        "conv_kernel_size": 5,
        "dil_kernel_size": 3,
        "profile_kernel_size": 5,
        "activation": "relu",
        "weight_norm": False,
        "residual_architecture": "residual_post-activation_conv",
    }


class _StaticBranch(nn.Module):
    def __init__(self, logits: torch.Tensor, log_counts: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)
        self.register_buffer("_log_counts", log_counts)

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        batch_size = x.shape[0]
        return ProfileCountOutput(
            logits=self._logits.expand(batch_size, -1, -1),
            log_counts=self._log_counts.expand(batch_size, -1),
        )


def test_chrombpnet_shared_bias_broadcasts_logits_and_counts():
    model = ChromBPNet(
        input_len=16,
        output_len=4,
        output_channels=["task_a", "task_b"],
        shared_bias=True,
        accessibility_args=_small_bpnet_args(),
        bias_args=_small_bpnet_args(),
    )
    acc_logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
    acc_log_counts = torch.log(torch.tensor([[2.0, 4.0]]))
    bias_logits = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    bias_log_counts = torch.log(torch.tensor([[8.0]]))
    model.accessibility_model = _StaticBranch(acc_logits, acc_log_counts)
    model.bias_model = _StaticBranch(bias_logits, bias_log_counts)

    out = model(torch.zeros(3, 4, 16))

    assert out.logits.shape == (3, 2, 4)
    assert out.log_counts.shape == (3, 2)
    assert torch.allclose(out.logits[0], acc_logits[0] + bias_logits[0])
    expected_counts = torch.log(torch.tensor([[10.0, 12.0]]))
    assert torch.allclose(out.log_counts[:1], expected_counts)


def test_multitask_chrombpnet_forward_shapes_and_branch_modes():
    model = MultitaskChromBPNet(
        input_len=64,
        output_len=32,
        output_channels=["task_a", "task_b", "task_c"],
        accessibility_args=_small_bpnet_args(filters=6),
        bias_args=_small_bpnet_args(filters=4),
    )

    out = model(torch.randn(2, 4, 64))

    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (2, 3, 32)
    assert out.log_counts.shape == (2, 3)
    assert model.shared_bias is True
    assert model.accessibility_model.n_output_channels == 3
    assert model.accessibility_model.predict_total_count is False
    assert model.bias_model.n_output_channels == 1
    assert model.bias_model.predict_total_count is True


def test_multitask_chrombpnet_requires_multiple_tasks():
    with pytest.raises(ValueError, match="at least two output_channels"):
        MultitaskChromBPNet(output_channels=["only_task"])


def test_multitask_chrombpnet_loss_is_per_channel_compatible():
    model = MultitaskChromBPNet(
        input_len=64,
        output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_small_bpnet_args(filters=5),
        bias_args=_small_bpnet_args(filters=4),
    )
    outputs = model(torch.randn(2, 4, 64))
    targets = torch.rand(2, 2, 32)

    loss = MultitaskBPNetLoss(alpha=0.1, beta=0.1)(outputs, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_single_channel_bias_checkpoint_loads_into_multitask_shared_bias(tmp_path: Path):
    bias_args = _small_bpnet_args(filters=4)
    single_bias = BPNet(
        input_len=64,
        output_len=32,
        output_channels=["signal"],
        predict_total_count=True,
        **bias_args,
    )
    ckpt_path = tmp_path / "bias.pt"
    torch.save(single_bias.state_dict(), ckpt_path)

    model = MultitaskChromBPNet(
        input_len=64,
        output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_small_bpnet_args(filters=5),
        bias_args=bias_args,
    )
    load_pretrained_weights(
        model,
        [
            PretrainedConfig(
                weights_path=str(ckpt_path),
                source=None,
                target="bias_model",
                freeze=True,
            )
        ],
    )

    assert all(not param.requires_grad for param in model.bias_model.parameters())


def test_non_shared_multitask_counts_fail_per_channel_loss():
    model = ChromBPNet(
        input_len=64,
        output_len=32,
        output_channels=["task_a", "task_b"],
        accessibility_args=_small_bpnet_args(filters=5),
        bias_args=_small_bpnet_args(filters=4),
    )
    outputs = model(torch.randn(2, 4, 64))
    targets = torch.rand(2, 2, 32)

    with pytest.raises(ValueError, match="per-channel log_counts"):
        MSEMultinomialLoss(count_per_channel=True)(outputs, targets)


def test_load_targets_json_sanitizes_and_sorts(tmp_path: Path):
    spec_path = tmp_path / "targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "task two": {"bigwig": "two.bw", "peaks": "two.bed"},
                "task one": {"bigwig": "one.bw", "peaks": "one.bed"},
            }
        )
    )

    targets, peaks = _load_targets_json(spec_path)

    assert list(targets) == ["task_one", "task_two"]
    assert targets == {"task_one": "one.bw", "task_two": "two.bw"}
    assert peaks == ["one.bed", "two.bed"]


def test_merge_peaks_reads_gzip_inputs(tmp_path: Path):
    peak_a = tmp_path / "a.bed.gz"
    peak_b = tmp_path / "b.bed"
    with gzip.open(peak_a, "wt") as handle:
        handle.write("chr1\t0\t10\n")
    peak_b.write_text("chr2\t20\t30\n")

    merged = Path(_merge_peaks([str(peak_a), str(peak_b)], str(tmp_path)))

    assert merged.read_text() == "chr1\t0\t10\nchr2\t20\t30\n"
