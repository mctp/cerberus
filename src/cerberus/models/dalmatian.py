"""
Dalmatian: End-to-end bias-factorized sequence-to-function model.

Composes two Pomeranian sub-networks:
- BiasNet: small, limited receptive field (~147bp) -- learns Tn5 sequence bias
- SignalNet: large, full receptive field (~1089bp) -- learns regulatory grammar

Their outputs are combined (profile addition in logit space, count
addition in log space via logsumexp) and returned as FactorizedProfileCountOutput.
"""

import logging

import torch
import torch.nn as nn

from cerberus.models.pomeranian import Pomeranian
from cerberus.output import FactorizedProfileCountOutput

logger = logging.getLogger(__name__)


def _compute_shrinkage(
    conv_kernel_size: int | list[int],
    dilations: list[int],
    dil_kernel_size: int,
    profile_kernel_size: int,
) -> int:
    """Compute total shrinkage (in bp) for a Pomeranian with valid padding.

    Shrinkage = stem + tower + profile_head, where each valid-padding layer
    shrinks by dilation * (kernel_size - 1).
    """
    if isinstance(conv_kernel_size, int):
        stem = conv_kernel_size - 1
    else:
        stem = sum(k - 1 for k in conv_kernel_size)
    tower = sum(d * (dil_kernel_size - 1) for d in dilations)
    head = profile_kernel_size - 1
    return stem + tower + head


class Dalmatian(nn.Module):
    """
    Dalmatian: End-to-end bias-factorized sequence-to-function model.

    Composes two Pomeranian sub-networks (BiasNet and SignalNet) whose outputs
    are combined via logit addition (profile) and logsumexp (counts). Signal
    output layers are zero-initialized so that at initialization, the combined
    output equals the bias-only output.

    The BiasNet receives a center-cropped input (sized to its receptive field
    needs) so that its natural output length matches output_len exactly with
    no excess cropping.

    Args:
        input_len: Length of input sequence in bp (SignalNet uses full input).
        output_len: Length of output profile in bp/bins.
        output_bin_size: Output resolution bin size.
        input_channels: List of input channel names.
        output_channels: List of output channel names.
        bias_filters: BiasNet model dimension.
        bias_n_layers: Number of dilated layers in BiasNet.
        bias_dilations: Dilation schedule for BiasNet. Limited RF (~147bp).
        bias_dil_kernel_size: Dilated conv kernel size for BiasNet.
        bias_conv_kernel_size: Stem kernel size(s) for BiasNet.
        bias_profile_kernel_size: Profile head kernel size for BiasNet.
        bias_expansion: PGC expansion factor for BiasNet.
        bias_stem_expansion: Stem expansion factor for BiasNet.
        bias_dropout: Dropout rate for BiasNet.
        signal_filters: SignalNet model dimension.
        signal_n_layers: Number of dilated layers in SignalNet.
        signal_dilations: Dilation schedule for SignalNet. Full RF (~1089bp).
        signal_dil_kernel_size: Dilated conv kernel size for SignalNet.
        signal_conv_kernel_size: Stem kernel size(s) for SignalNet.
        signal_profile_kernel_size: Profile head kernel size for SignalNet.
        signal_expansion: PGC expansion factor for SignalNet.
        signal_stem_expansion: Stem expansion factor for SignalNet.
        signal_dropout: Dropout rate for SignalNet.
    """

    def __init__(
        self,
        input_len: int = 2112,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        # --- BiasNet configuration (RF=147bp, ~72K params) ---
        bias_filters: int = 64,
        bias_n_layers: int = 4,
        bias_dilations: list[int] | None = None,
        bias_dil_kernel_size: int = 9,
        bias_conv_kernel_size: int | list[int] = 11,
        bias_profile_kernel_size: int = 17,
        bias_expansion: int = 1,
        bias_stem_expansion: int = 1,
        bias_dropout: float = 0.1,
        # --- SignalNet configuration (RF=1089bp, ~2-3M params) ---
        signal_filters: int = 256,
        signal_n_layers: int = 8,
        signal_dilations: list[int] | None = None,
        signal_dil_kernel_size: int = 9,
        signal_conv_kernel_size: int | list[int] | None = None,
        signal_profile_kernel_size: int = 45,
        signal_expansion: int = 2,
        signal_stem_expansion: int = 2,
        signal_dropout: float = 0.1,
    ):
        super().__init__()

        if bias_dilations is None:
            bias_dilations = [1, 2, 4, 8]
        if signal_dilations is None:
            signal_dilations = [1, 1, 2, 4, 8, 16, 32, 64]
        if signal_conv_kernel_size is None:
            signal_conv_kernel_size = [11, 11]

        # Compute BiasNet shrinkage and derive its input length
        bias_shrinkage = _compute_shrinkage(
            bias_conv_kernel_size, bias_dilations,
            bias_dil_kernel_size, bias_profile_kernel_size,
        )
        self.bias_input_len = output_len + bias_shrinkage

        if self.bias_input_len > input_len:
            raise ValueError(
                f"BiasNet requires input_len >= {self.bias_input_len} "
                f"(output_len={output_len} + shrinkage={bias_shrinkage}), "
                f"but got input_len={input_len}"
            )

        # Verify SignalNet shrinkage matches input_len -> output_len exactly
        signal_shrinkage = _compute_shrinkage(
            signal_conv_kernel_size, signal_dilations,
            signal_dil_kernel_size, signal_profile_kernel_size,
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

        self.bias_model = Pomeranian(
            input_len=self.bias_input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=output_channels,
            filters=bias_filters,
            n_dilated_layers=bias_n_layers,
            dilations=bias_dilations,
            dil_kernel_size=bias_dil_kernel_size,
            conv_kernel_size=bias_conv_kernel_size,
            profile_kernel_size=bias_profile_kernel_size,
            expansion=bias_expansion,
            stem_expansion=bias_stem_expansion,
            dropout=bias_dropout,
            predict_total_count=False,
        )

        self.signal_model = Pomeranian(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=output_channels,
            filters=signal_filters,
            n_dilated_layers=signal_n_layers,
            dilations=signal_dilations,
            dil_kernel_size=signal_dil_kernel_size,
            conv_kernel_size=signal_conv_kernel_size,
            profile_kernel_size=signal_profile_kernel_size,
            expansion=signal_expansion,
            stem_expansion=signal_stem_expansion,
            dropout=signal_dropout,
            predict_total_count=False,
        )

        self._zero_init_signal_outputs()

        bias_params = sum(p.numel() for p in self.bias_model.parameters())
        signal_params = sum(p.numel() for p in self.signal_model.parameters())
        logger.info(
            f"Dalmatian initialized: bias_model={bias_params:,} params (RF={bias_shrinkage + 1}bp), "
            f"signal_model={signal_params:,} params (RF={signal_shrinkage + 1}bp), "
            f"total={bias_params + signal_params:,} params"
        )

    def _zero_init_signal_outputs(self) -> None:
        """Zero-initialize signal model output layers for identity-element start.

        Profile head: weight=0, bias=0 -> signal_logits = 0 (identity for addition).
        Count head final layer: weight=0, bias=-10 -> signal_log_counts ~ -10
            (identity for logsumexp since exp(-10) ~ 0.000045 phantom reads).
        """
        profile_pw = self.signal_model.profile_pointwise
        profile_sp = self.signal_model.profile_spatial
        nn.init.zeros_(profile_pw.weight)
        nn.init.zeros_(profile_pw.bias)  # type: ignore[arg-type]
        nn.init.zeros_(profile_sp.weight)
        nn.init.zeros_(profile_sp.bias)  # type: ignore[arg-type]

        final_linear: nn.Linear = self.signal_model.count_mlp[-1]  # type: ignore[assignment]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, -10.0)

    def forward(self, x: torch.Tensor) -> FactorizedProfileCountOutput:
        # Both sub-models center-crop to their own input_len automatically
        bias_out = self.bias_model(x)
        signal_out = self.signal_model(x)

        combined_logits = bias_out.logits + signal_out.logits
        combined_log_counts = torch.logsumexp(
            torch.stack([bias_out.log_counts, signal_out.log_counts], dim=-1),
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
