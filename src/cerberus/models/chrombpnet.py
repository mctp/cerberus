"""ChromBPNet: BPNet-based bias-factorized ATAC model.

This implementation mirrors the reference ``chrombpnet-pytorch`` design:

- a large accessibility ``BPNet`` branch
- a smaller bias ``BPNet`` branch
- profile combination by logit addition
- count combination by ``logaddexp``

Unlike :class:`cerberus.models.dalmatian.Dalmatian`, this model does not use
gradient routing or decomposed outputs.  It returns a plain
``ProfileCountOutput`` so it plugs into the standard Cerberus
loss/metric/prediction infrastructure.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from cerberus.output import ProfileCountOutput

from .bpnet import BPNet

logger = logging.getLogger(__name__)

_ACCESSIBILITY_DEFAULTS: dict[str, Any] = {
    "filters": 512,
    "n_dilated_layers": 8,
    "conv_kernel_size": 21,
    "dil_kernel_size": 3,
    "profile_kernel_size": 75,
    "predict_total_count": True,
    "activation": "relu",
    "weight_norm": False,
    "residual_architecture": "residual_post-activation_conv",
}

_BIAS_DEFAULTS: dict[str, Any] = {
    "filters": 128,
    "n_dilated_layers": 4,
    "conv_kernel_size": 21,
    "dil_kernel_size": 3,
    "profile_kernel_size": 75,
    "predict_total_count": True,
    "activation": "relu",
    "weight_norm": False,
    "residual_architecture": "residual_post-activation_conv",
}


def _resolve_branch_shapes(
    branch_name: str,
    branch_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    """Broadcast singleton channel outputs, otherwise require exact shape."""
    if branch_tensor.shape == target_tensor.shape:
        return branch_tensor

    if branch_tensor.ndim != target_tensor.ndim:
        raise ValueError(
            f"{branch_name} tensor rank {branch_tensor.ndim} does not match "
            f"target rank {target_tensor.ndim}"
        )

    # Allow a shared-bias style singleton channel to broadcast over channels.
    if branch_tensor.ndim >= 2 and branch_tensor.shape[1] == 1:
        expand_shape = list(target_tensor.shape)
        expand_shape[0] = branch_tensor.shape[0]
        return branch_tensor.expand(*expand_shape)

    raise ValueError(
        f"{branch_name} shape {tuple(branch_tensor.shape)} is incompatible with "
        f"target shape {tuple(target_tensor.shape)}"
    )


class ChromBPNet(nn.Module):
    """ChromBPNet-style ATAC model composed of two BPNet sub-networks.

    Args:
        input_len: Input sequence length. Default ``2114``.
        output_len: Output profile length. Default ``1000``.
        output_bin_size: Output bin size. Default ``1``.
        input_channels: Input channel names. Default DNA one-hot channels.
        output_channels: Output track names. Default ``["signal"]``.
        accessibility_args: Optional overrides for the accessibility BPNet.
        bias_args: Optional overrides for the bias BPNet.
        bias_logcount_offset: Constant additive offset applied to the bias
            branch log-count predictions before ``logaddexp`` combination.
            This mirrors ChromBPNet's optional post-hoc bias-count adjustment.
    """

    def __init__(
        self,
        input_len: int = 2114,
        output_len: int = 1000,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        accessibility_args: dict[str, Any] | None = None,
        bias_args: dict[str, Any] | None = None,
        bias_logcount_offset: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels is None:
            input_channels = ["A", "C", "G", "T"]
        if output_channels is None:
            output_channels = ["signal"]

        acc_kw: dict[str, Any] = dict(accessibility_args or {})
        bias_kw: dict[str, Any] = dict(bias_args or {})

        for key, value in _ACCESSIBILITY_DEFAULTS.items():
            acc_kw.setdefault(key, value)
        for key, value in _BIAS_DEFAULTS.items():
            bias_kw.setdefault(key, value)

        shared_kw = {
            "input_len": input_len,
            "output_len": output_len,
            "output_bin_size": output_bin_size,
            "input_channels": input_channels,
            "output_channels": output_channels,
        }
        acc_kw.update(shared_kw)
        bias_kw.update(shared_kw)

        self.accessibility_model = BPNet(**acc_kw)
        self.bias_model = BPNet(**bias_kw)
        self.register_buffer(
            "bias_logcount_offset",
            torch.tensor(float(bias_logcount_offset), dtype=torch.float32),
        )

        acc_params = sum(p.numel() for p in self.accessibility_model.parameters())
        bias_params = sum(p.numel() for p in self.bias_model.parameters())
        logger.info(
            "ChromBPNet initialized: accessibility_model=%s params, "
            "bias_model=%s params, total=%s params",
            f"{acc_params:,}",
            f"{bias_params:,}",
            f"{acc_params + bias_params:,}",
        )

    @property
    def chrombpnet_wo_bias(self) -> BPNet:
        """Reference-style alias for the accessibility branch."""
        return self.accessibility_model

    @property
    def bias(self) -> BPNet:
        """Reference-style alias for the bias branch."""
        return self.bias_model

    def set_bias_logcount_offset(self, value: float) -> None:
        """Update the non-trainable scalar offset on the bias count branch."""
        self.bias_logcount_offset.fill_(float(value))

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        acc_out = self.accessibility_model(x)
        bias_out = self.bias_model(x)

        bias_logits = _resolve_branch_shapes(
            "bias logits", bias_out.logits, acc_out.logits
        )
        bias_log_counts = _resolve_branch_shapes(
            "bias log_counts", bias_out.log_counts, acc_out.log_counts
        )
        bias_log_counts = bias_log_counts + self.bias_logcount_offset.to(
            dtype=bias_log_counts.dtype, device=bias_log_counts.device
        )

        combined_logits = acc_out.logits + bias_logits
        combined_log_counts = torch.logaddexp(acc_out.log_counts, bias_log_counts)
        return ProfileCountOutput(
            logits=combined_logits,
            log_counts=combined_log_counts,
        )


@torch.no_grad()
def estimate_bias_logcount_offset(
    bias_model: nn.Module,
    dataloader: Iterable[dict[str, object]],
    count_pseudocount: float = 1.0,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> float:
    """Estimate a scalar offset for the bias branch log-count head.

    The offset is the mean residual between observed log-counts and the bias
    model's predicted log-counts over the provided dataloader.  This mirrors the
    optional post-hoc adjustment used by ``chrombpnet-pytorch``.
    """

    try:
        first_param = next(bias_model.parameters())
    except StopIteration as exc:
        raise ValueError("bias_model has no parameters") from exc

    if device is None:
        device = first_param.device
    device = torch.device(device)

    was_training = bias_model.training
    bias_model.eval()
    bias_model.to(device)

    deltas: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs_obj = batch.get("inputs")
        targets_obj = batch.get("targets")
        if not isinstance(inputs_obj, torch.Tensor) or not isinstance(
            targets_obj, torch.Tensor
        ):
            raise TypeError(
                "estimate_bias_logcount_offset expects batch['inputs'] and "
                "batch['targets'] to be torch.Tensor instances"
            )

        inputs = inputs_obj.to(device)
        targets = targets_obj.to(device)
        pred_log_counts = bias_model(inputs).log_counts.float()
        if pred_log_counts.ndim != 2:
            raise ValueError(
                f"Expected bias_model log_counts to have shape (B, C), got "
                f"{tuple(pred_log_counts.shape)}"
            )

        if pred_log_counts.shape[1] == 1:
            pred_total_log_counts = pred_log_counts[:, 0]
        else:
            pred_total_log_counts = torch.logsumexp(pred_log_counts, dim=1)

        target_total_counts = targets.float().sum(dim=(1, 2))
        target_total_log_counts = torch.log(target_total_counts + count_pseudocount)
        deltas.append(target_total_log_counts - pred_total_log_counts)

    if was_training:
        bias_model.train()

    if not deltas:
        raise ValueError("No batches were available to estimate bias log-count offset")

    return torch.cat(deltas).mean().item()
