"""
ChromBPNet: BPNet-based bias-factorized ATAC model.

Composes two BPNet sub-networks (accessibility + bias).  Profiles
combine by logit addition; counts combine by ``logaddexp``.  Matches
the forward semantics of the reference chrombpnet-pytorch
implementation.
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from cerberus.models.bpnet import BPNet
from cerberus.output import ProfileCountOutput

logger = logging.getLogger(__name__)


# Module-level defaults are exposed (not just inlined) so users can inspect
# the architecture spec without instantiating the model, and so
# ``compute_shrinkage`` and similar helpers can be derived from the same
# constants the constructor consumes.  Mirrors the reference
# chrombpnet-pytorch architecture (filters=512/128, n_layers=8/4).
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
    """Broadcast a singleton-channel branch tensor over the target channels.

    Pass-through when shapes already match.  When ``branch_tensor`` has the
    same rank but a singleton channel dim (``shape[1] == 1``), expand it to
    match ``target_tensor``'s channel count.  Any other shape mismatch
    raises -- the caller has a structural bug, not a broadcastable one.

    Used by :meth:`ChromBPNet.forward` so a one-channel bias branch
    (``shared_bias=True``) can combine with a multi-task accessibility
    branch without mutating either sub-model's output contract.
    """
    if branch_tensor.shape == target_tensor.shape:
        return branch_tensor

    if branch_tensor.ndim == target_tensor.ndim and branch_tensor.shape[1] == 1:
        expand_shape = list(target_tensor.shape)
        expand_shape[0] = branch_tensor.shape[0]
        return branch_tensor.expand(*expand_shape)

    raise ValueError(
        f"{branch_name} shape {tuple(branch_tensor.shape)} is incompatible "
        f"with target shape {tuple(target_tensor.shape)}"
    )


class ChromBPNet(nn.Module):
    """
    ChromBPNet: Bias-factorized ATAC model with two BPNet sub-networks.

    Architecture (mirrors the reference chrombpnet-pytorch design):
    - Accessibility branch: large :class:`BPNet` (``filters=512``,
      ``n_dilated_layers=8``).  Learns regulatory grammar.
    - Bias branch: smaller :class:`BPNet` (``filters=128``,
      ``n_dilated_layers=4``).  Captures Tn5 enzymatic sequence
      preference; typically loaded pre-trained and frozen via
      :attr:`ModelConfig.freeze`.
    - Profile combination: raw logit addition
      (``acc.logits + bias.logits``).
    - Count combination: ``torch.logaddexp`` (numerically stable form
      of ``log(exp(acc) + exp(bias))``).

    Returns a plain :class:`ProfileCountOutput`, unlike
    :class:`cerberus.models.dalmatian.Dalmatian` which returns the
    decomposed factorized variant for gradient routing.

    The bias branch's log-count predictions are shifted by a
    non-trainable scalar ``bias_logcount_offset`` before combination,
    mirroring chrombpnet-pytorch's bias-count calibration step (see
    :func:`estimate_bias_logcount_offset`).  In the reference the
    offset is baked into ``bias.linear.bias`` directly; here it is a
    separate buffer so the loaded bias ``state_dict`` is preserved
    intact and the calibration round-trips through
    ``ChromBPNet.state_dict``.  Update with
    ``model.bias_logcount_offset.fill_(value)``.

    Reference-equivalent training: set
    ``ModelConfig.count_pseudocount=1.0`` and
    ``DataConfig.target_scale=1.0`` to reproduce chrombpnet-pytorch's
    ``log1p(counts)`` count-head target.  Both sub-BPNets apply
    Xavier/Glorot uniform initialisation (``BPNet._tf_style_reinit``)
    matching the reference's ``tf_style_reinit``.

    Args:
        input_len (int): Length of input sequence in bp. Default: ``2114``.
        output_len (int): Length of output profile in bp. Default: ``1000``.
        output_bin_size (int): Output resolution bin size. Default: ``1``.
        input_channels (list[str]): Input channel names. Default DNA
            one-hot (``["A", "C", "G", "T"]``).
        output_channels (list[str]): Output track names. Default
            ``["signal"]`` (single-channel ATAC).
        accessibility_args (dict[str, Any] | None): Optional overrides
            for the accessibility :class:`BPNet` constructor.  Shared
            kwargs (``input_len``, ``output_len``, ``output_bin_size``,
            ``input_channels``, ``output_channels``) are injected
            automatically and override any user-supplied duplicates.
        bias_args (dict[str, Any] | None): Optional overrides for the
            bias :class:`BPNet`.  Same semantics as
            ``accessibility_args``.
        bias_logcount_offset (float): Initial value for the additive
            offset applied to bias log-counts before ``logaddexp``
            combination. Default: ``0.0``.
        shared_bias (bool): If ``True``, the bias branch is built with
            ``output_channels=["bias"]`` (one channel) and broadcast
            across the accessibility branch's channels at forward time.
            Required to compose a single Tn5-bias model with a
            multi-task accessibility branch (see
            :class:`MultitaskChromBPNet`).  Default: ``False``.
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
        shared_bias: bool = False,
    ):
        super().__init__()
        if input_channels is None:
            input_channels = ["A", "C", "G", "T"]
        if output_channels is None:
            output_channels = ["signal"]

        # Build sub-model kwargs: start from user overrides, then apply
        # ChromBPNet defaults via setdefault (so user values win), then inject
        # shared params via update (so they override duplicates, matching the
        # Dalmatian pattern).
        acc_kw: dict[str, Any] = dict(accessibility_args) if accessibility_args else {}
        bias_kw: dict[str, Any] = dict(bias_args) if bias_args else {}
        for key, value in _ACCESSIBILITY_DEFAULTS.items():
            acc_kw.setdefault(key, value)
        for key, value in _BIAS_DEFAULTS.items():
            bias_kw.setdefault(key, value)

        shared_kw = {
            "input_len": input_len,
            "output_len": output_len,
            "output_bin_size": output_bin_size,
            "input_channels": input_channels,
        }
        acc_kw.update(shared_kw, output_channels=output_channels)
        # shared_bias=True collapses the bias branch to a single "bias" channel
        # that broadcasts across all accessibility channels at forward time --
        # so one Tn5-bias model can serve a multi-task accessibility branch.
        bias_kw.update(
            shared_kw,
            output_channels=["bias"] if shared_bias else output_channels,
        )

        self.shared_bias = shared_bias
        self.accessibility_model = BPNet(**acc_kw)
        self.bias_model = BPNet(**bias_kw)

        # register_buffer (not Parameter) keeps the offset non-trainable while
        # still moving with .to(device) and persisting in state_dict.  Mutate
        # via ``self.bias_logcount_offset.fill_(value)``.
        self.register_buffer(
            "bias_logcount_offset",
            torch.tensor(float(bias_logcount_offset), dtype=torch.float32),
        )

        acc_params = sum(p.numel() for p in self.accessibility_model.parameters())
        bias_params = sum(p.numel() for p in self.bias_model.parameters())
        logger.info(
            "ChromBPNet initialized: accessibility_model=%s params, "
            "bias_model=%s params, total=%s params, bias_logcount_offset=%.4f, "
            "shared_bias=%s",
            f"{acc_params:,}",
            f"{bias_params:,}",
            f"{acc_params + bias_params:,}",
            float(bias_logcount_offset),
            shared_bias,
        )

    def forward(self, x) -> ProfileCountOutput:
        """
        Forward pass.

        Args:
            x (Tensor): Input sequence (Batch, Channels, Input_Len).

        Returns:
            ProfileCountOutput: Contains ``logits`` (Batch, Out_Channels,
                Output_Len) from raw logit addition and ``log_counts``
                (Batch, Out_Channels) from ``logaddexp`` combination.
        """
        acc_out = self.accessibility_model(x)
        bias_out = self.bias_model(x)

        # Broadcast a singleton-channel bias output over the accessibility
        # channels when shared_bias=True; pass-through otherwise.
        bias_logits = _resolve_branch_shapes(
            "bias logits", bias_out.logits, acc_out.logits,
        )
        bias_log_counts_raw = _resolve_branch_shapes(
            "bias log_counts", bias_out.log_counts, acc_out.log_counts,
        )

        # Deviation vs chrombpnet-pytorch: reference bakes this offset into
        # bias_model.linear.bias before training; we apply it at forward time
        # from a wrapper-level buffer.  Forward math is identical.
        offset: torch.Tensor = self.bias_logcount_offset.to(  # type: ignore[union-attr]
            dtype=bias_log_counts_raw.dtype, device=bias_log_counts_raw.device,
        )
        bias_log_counts = bias_log_counts_raw + offset

        combined_logits = acc_out.logits + bias_logits
        combined_log_counts = torch.logaddexp(acc_out.log_counts, bias_log_counts)
        return ProfileCountOutput(
            logits=combined_logits,
            log_counts=combined_log_counts,
        )


class MultitaskChromBPNet(ChromBPNet):
    """Multi-task ChromBPNet with a reusable single-channel bias branch.

    The accessibility branch predicts one profile + one count scalar
    per task (``predict_total_count=False`` — same per-channel count
    requirement as :class:`cerberus.models.bpnet.MultitaskBPNet`).
    The bias branch stays one channel and broadcasts across tasks at
    forward time, so a stage-1 ChromBPNet bias BPNet trained on one
    normalised sample can be frozen and reused for every other task
    from the same assay (mirrors the chrombpnet-pytorch convention
    that the Tn5 bias model is dataset-wide, not per-condition).

    Pairs with :class:`cerberus.models.bpnet.MultitaskBPNetLoss` for
    absolute-counts training (each task gets independent per-channel
    profile + count losses) and with
    :class:`cerberus.loss.DifferentialCountLoss` for differential
    fine-tuning across two of the task channels (see
    ``tools/train_chrombpnet_multitask.py`` and
    ``tools/train_chrombpnet_multitask_differential.py``).

    Args:
        input_len, output_len, output_bin_size, input_channels,
        accessibility_args, bias_args, bias_logcount_offset: forwarded
            to :class:`ChromBPNet`.
        output_channels (list[str]): Task names; must contain at least
            two entries.

    Raises:
        ValueError: If fewer than two output channels are supplied.
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
    ):
        if output_channels is None or len(output_channels) < 2:
            raise ValueError(
                "MultitaskChromBPNet requires at least two output_channels; "
                f"got {output_channels!r}"
            )

        acc_kw: dict[str, Any] = dict(accessibility_args or {})
        bias_kw: dict[str, Any] = dict(bias_args or {})

        # Pin the two count-mode flags that define this architecture and
        # warn (rather than silently overwrite) when the caller disagrees,
        # mirroring the MultitaskBPNetLoss `_fixed` pattern.
        caller_acc_count_mode = acc_kw.pop("predict_total_count", None)
        if caller_acc_count_mode is not None and caller_acc_count_mode is not False:
            logger.warning(
                "MultitaskChromBPNet ignores accessibility_args['predict_total_count']=%r "
                "and uses False so each task has its own count head.",
                caller_acc_count_mode,
            )
        acc_kw["predict_total_count"] = False

        # A one-channel bias model has identical state-dict shapes with either
        # count mode, but predict_total_count=True documents the stage-1 bias
        # workflow and matches tools/train_chrombpnet_bias.py.
        caller_bias_count_mode = bias_kw.pop("predict_total_count", None)
        if caller_bias_count_mode is not None and caller_bias_count_mode is not True:
            logger.warning(
                "MultitaskChromBPNet ignores bias_args['predict_total_count']=%r "
                "and uses True for the reusable one-channel bias branch.",
                caller_bias_count_mode,
            )
        bias_kw["predict_total_count"] = True

        super().__init__(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=output_channels,
            accessibility_args=acc_kw,
            bias_args=bias_kw,
            bias_logcount_offset=bias_logcount_offset,
            shared_bias=True,
        )


@torch.no_grad()
def estimate_bias_logcount_offset(
    bias_model: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    count_pseudocount: float = 1.0,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> float:
    """Estimate the scalar bias-branch log-count calibration offset.

    Iterates ``dataloader`` and returns the mean residual between the
    observed log-counts and the bias model's predicted log-counts.
    Mirrors chrombpnet-pytorch's ``adjust_bias_model_logcounts``
    step.

    In the reference the offset is applied by mutating
    ``bias.linear.bias`` directly.  Here the returned scalar is
    intended for :attr:`ChromBPNet.bias_logcount_offset` (set via
    ``chrombpnet.bias_logcount_offset.fill_(value)``) so the bias
    model's loaded weights are preserved intact.

    Args:
        bias_model: A :class:`BPNet` (or compatible) module whose
            ``log_counts`` head produces shape ``(B, C)``.  Mode is
            saved and restored.
        dataloader: Iterable of batch dicts with ``"inputs"`` (Tensor,
            ``(B, C_in, L)``) and ``"targets"`` (Tensor,
            ``(B, C_target, L)``).
        count_pseudocount: Additive offset before
            ``log(total_counts + pc)`` on the observed counts. Default
            ``1.0`` matches the reference's ``log1p`` target.
        device: Optional override for the device on which to run the
            forward passes.  Defaults to ``next(bias_model.parameters()
            ).device``.
        max_batches: Optional cap on the number of batches consumed.

    Returns:
        Mean residual ``log(observed) - log(predicted)`` across the
        consumed batches, suitable to assign to a ChromBPNet's
        ``bias_logcount_offset`` buffer.

    Raises:
        ValueError: If the bias model has no parameters, no batches
            were available, or the bias model's ``log_counts`` shape
            is not 2D.
        TypeError: If a batch's ``"inputs"`` or ``"targets"`` entry is
            not a :class:`torch.Tensor`.
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

        # Sum predicted log-counts across channels in log-space (logsumexp) so
        # a multi-channel bias model maps to a single total-count scalar; for
        # the standard single-channel ATAC bias this collapses to the
        # identity.
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

    # Deviation vs chrombpnet-pytorch: reference mutates
    # ``bias_model.linear.bias += delta`` here; we return the scalar so the
    # caller can assign it to ``chrombpnet.bias_logcount_offset``.
    return torch.cat(deltas).mean().item()
