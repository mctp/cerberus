"""
Dalmatian: End-to-end bias-factorized sequence-to-function model.

Composes two sub-networks:
- BiasNet: lightweight Conv1d+ReLU stack, limited RF (~105bp), captures Tn5 sequence bias.
  Fully DeepLIFT/DeepSHAP compatible (Conv1d + ReLU + residual add only).
- SignalNet: large Pomeranian, full RF (~1089bp), captures regulatory grammar.

Their outputs are combined (profile addition in logit space, count
addition in log space via logsumexp) and returned as FactorizedProfileCountOutput.

Gradient separation: bias outputs are detached before combining, so the
combined reconstruction loss (L_recon) trains only SignalNet. BiasNet
receives gradients exclusively from the bias-only loss term (L_bias on
background regions), preventing it from learning TF footprints.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from cerberus.models.biasnet import BiasNet
from cerberus.models.pomeranian import Pomeranian
from cerberus.output import FactorizedProfileCountOutput

logger = logging.getLogger(__name__)

# Preset configurations for SignalNet: (filters, expansion)
_SIGNAL_PRESETS: dict[str, tuple[int, int]] = {
    "standard": (64, 1),  # ~150K params (matches Pomeranian K9)
    "large": (256, 2),  # ~3.9M params
}


class Dalmatian(nn.Module):
    """End-to-end bias-factorized sequence-to-function model.

    Composes a BiasNet (Conv1d+ReLU, short RF) and a SignalNet (Pomeranian,
    long RF) whose outputs are combined via logit addition (profile) and
    logsumexp (counts).

    Gradient separation: bias outputs are ``.detach()``-ed before combining,
    so L_recon gradients train only SignalNet. BiasNet learns exclusively
    from L_bias (background reconstruction), replicating ChromBPNet's
    freeze-bias design without requiring a two-stage training procedure.

    The BiasNet receives a center-cropped input (sized to its receptive field
    needs) so that its natural output length matches output_len exactly with
    no excess cropping.

    Args:
        input_len: Length of input sequence in bp (SignalNet uses full input).
        output_len: Length of output profile in bp/bins.
        output_bin_size: Output resolution bin size.
        input_channels: List of input channel names.
        output_channels: List of output channel names.
        bias_args: Dict of BiasNet constructor kwargs (native parameter names,
            e.g. ``{"filters": 12, "dropout": 0.1}``). Shared params
            (``input_len``, ``output_len``, etc.) are injected automatically.
        signal_args: Dict of Pomeranian constructor kwargs (native parameter
            names, e.g. ``{"dropout": 0.1}``). Shared params are injected
            automatically.
        signal_preset: SignalNet preset. ``"standard"`` (default): f=64,
            expansion=1, ~150K params — matches standalone Pomeranian K9.
            ``"large"``: f=256, expansion=2, ~3.9M params.
            Individual ``signal_args`` entries override the preset.
        shared_bias: If True, BiasNet uses a single output channel
            (``["bias"]``) while SignalNet uses the full ``output_channels``
            list. This enables multi-task training where Tn5 insertion bias
            is shared across cell types. Default: False.
    """

    def __init__(
        self,
        input_len: int = 2112,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        bias_args: dict[str, Any] | None = None,
        signal_args: dict[str, Any] | None = None,
        signal_preset: str = "standard",
        shared_bias: bool = False,
    ):
        super().__init__()
        self.shared_bias = shared_bias

        # --- Resolve signal preset defaults ---
        if signal_preset not in _SIGNAL_PRESETS:
            raise ValueError(
                f"Unknown signal_preset={signal_preset!r}. "
                f"Choose from {list(_SIGNAL_PRESETS.keys())}."
            )
        _default_filters, _default_expansion = _SIGNAL_PRESETS[signal_preset]

        # Build sub-model kwargs: start from user overrides, inject shared
        # params + preset defaults.
        bias_kw: dict[str, Any] = dict(bias_args) if bias_args else {}
        signal_kw: dict[str, Any] = dict(signal_args) if signal_args else {}

        # Apply signal preset defaults (user overrides take precedence)
        signal_kw.setdefault("filters", _default_filters)
        signal_kw.setdefault("expansion", _default_expansion)

        # Resolve bias architecture params for shrinkage computation
        bias_conv_kernel_size = bias_kw.get("conv_kernel_size", [11, 11])
        bias_n_layers = bias_kw.get("n_layers", 5)
        bias_dilations = bias_kw.get("dilations")
        bias_dil_kernel_size = bias_kw.get("dil_kernel_size", 9)
        bias_profile_kernel_size = bias_kw.get("profile_kernel_size", 45)

        bias_shrinkage = BiasNet.compute_shrinkage(
            conv_kernel_size=bias_conv_kernel_size,
            n_layers=bias_n_layers,
            dilations=bias_dilations,
            dil_kernel_size=bias_dil_kernel_size,
            profile_kernel_size=bias_profile_kernel_size,
        )
        self.bias_input_len = output_len + bias_shrinkage

        if self.bias_input_len > input_len:
            raise ValueError(
                f"BiasNet requires input_len >= {self.bias_input_len} "
                f"(output_len={output_len} + shrinkage={bias_shrinkage}), "
                f"but got input_len={input_len}"
            )

        # Resolve signal architecture params for shrinkage validation
        signal_conv_kernel_size = signal_kw.get("conv_kernel_size", [11, 11])
        signal_dilations = signal_kw.get("dilations")
        signal_n_layers = signal_kw.get("n_dilated_layers", 8)
        signal_dil_kernel_size = signal_kw.get("dil_kernel_size", 9)
        signal_profile_kernel_size = signal_kw.get("profile_kernel_size", 45)

        signal_shrinkage = Pomeranian.compute_shrinkage(
            conv_kernel_size=signal_conv_kernel_size,
            n_dilated_layers=signal_n_layers,
            dilations=signal_dilations,
            dil_kernel_size=signal_dil_kernel_size,
            profile_kernel_size=signal_profile_kernel_size,
        )
        signal_natural_output = input_len - signal_shrinkage
        if signal_natural_output != output_len:
            raise ValueError(
                f"SignalNet shrinkage={signal_shrinkage} gives natural output "
                f"{input_len}-{signal_shrinkage}={signal_natural_output}, "
                f"but output_len={output_len}. Adjust signal model parameters."
            )

        self.input_len = input_len
        self.output_len = output_len

        # Determine output channels for each sub-model
        bias_output_channels = ["bias"] if shared_bias else output_channels
        signal_output_channels = output_channels

        # Inject shared params (override any user-supplied duplicates)
        bias_kw.update(
            input_len=self.bias_input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=bias_output_channels,
            predict_total_count=False,
        )
        signal_kw.update(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=signal_output_channels,
            predict_total_count=False,
        )

        self.bias_model = BiasNet(**bias_kw)
        self.signal_model = Pomeranian(**signal_kw)

        bias_params = sum(p.numel() for p in self.bias_model.parameters())
        signal_params = sum(p.numel() for p in self.signal_model.parameters())
        logger.info(
            f"Dalmatian initialized: bias_model={bias_params:,} params (RF={bias_shrinkage + 1}bp), "
            f"signal_model={signal_params:,} params (RF={signal_shrinkage + 1}bp), "
            f"total={bias_params + signal_params:,} params, shared_bias={shared_bias}"
        )

    def forward(self, x: torch.Tensor) -> FactorizedProfileCountOutput:
        # Both sub-models center-crop to their own input_len automatically
        bias_out = self.bias_model(x)
        signal_out = self.signal_model(x)

        # Detach bias outputs before combining so L_recon gradients flow
        # only to signal_model.  The raw (non-detached) bias outputs are
        # returned in the FactorizedProfileCountOutput for L_bias to train
        # the bias model independently on background regions.
        bias_logits_detached = bias_out.logits.detach()
        bias_log_counts_detached = bias_out.log_counts.detach()

        # (B,1,L) + (B,N,L) broadcasts automatically for profile logits
        combined_logits = bias_logits_detached + signal_out.logits

        # For logsumexp, expand bias counts to match signal shape
        if self.shared_bias:
            bias_log_counts_detached = bias_log_counts_detached.expand_as(
                signal_out.log_counts
            )
        combined_log_counts = torch.logsumexp(
            torch.stack([bias_log_counts_detached, signal_out.log_counts], dim=-1),
            dim=-1,
        )

        return FactorizedProfileCountOutput(
            logits=combined_logits,
            log_counts=combined_log_counts,
            bias_logits=bias_out.logits,
            bias_log_counts=bias_out.log_counts,
            signal_logits=signal_out.logits,
            signal_log_counts=signal_out.log_counts,
        )
