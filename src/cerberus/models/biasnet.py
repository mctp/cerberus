"""
BiasNet: A lightweight bias model for Tn5 enzymatic sequence preference.

A plain Conv1d + ReLU stack with residual connections and valid padding.
Designed for ChromBPNet-style bias factorization in the Dalmatian model.
All operations are nn.Module-based (no F.relu) for full DeepLIFT/DeepSHAP
compatibility via captum.

Default configuration:
  - Filters: 12
  - Stem: 2-layer [11, 11] Conv1d + ReLU (valid padding)
  - Body: 5 × SimpleResidualBlock (k=9, dilation=1, residual)
  - Head: Conv1d(12, 1, 45) linear spatial conv (valid padding)
  - Count: GlobalAvgPool → Linear → ReLU → Linear
  - RF: 105bp, ~9.3K params

Architecture:
  Stem:  [Conv1d(4→f, k) → ReLU] × n_stem_layers             [valid padding]
  Body:  N × [Conv1d(f→f, k, d) → ReLU → Dropout + Residual]  [valid padding]
  Head:  Conv1d(f, 1, profile_k)                               [valid, linear]
     or: Conv1d(f, f, 1) → ReLU → Conv1d(f, 1, profile_k)     [valid, nonlinear]
  Count: GlobalAvgPool → Linear(f, f//2) → ReLU → Linear(f//2, n_out)
"""

import logging

import torch
import torch.nn as nn

from cerberus.layers import SimpleResidualBlock
from cerberus.output import ProfileCountOutput

logger = logging.getLogger(__name__)


class BiasNet(nn.Module):
    """
    BiasNet: Lightweight bias model for Tn5 sequence preference.

    A plain Conv1d + ReLU stack with residual connections designed to capture
    local enzymatic bias (~105bp receptive field) while remaining fully
    compatible with DeepLIFT/DeepSHAP attribution methods.

    Uses 'valid' padding throughout. Center-crops oversized input to
    ``input_len`` before the stem (same convention as Pomeranian and BPNet).

    Args:
        input_len (int): Length of input sequence. Default: 1128.
        output_len (int): Length of output sequence. Default: 1024.
        output_bin_size (int): Output resolution bin size. Default: 1.
        input_channels (list[str]): List of input channel names. Default: ACGT.
        output_channels (list[str]): List of output channel names. Default: ["signal"].
        filters (int): Number of conv filters (model dimension). Default: 12.
        n_layers (int): Number of residual tower blocks. Default: 5.
        conv_kernel_size (int | list[int]): Kernel size(s) for stem convolutions. Default: [11, 11].
        dil_kernel_size (int): Kernel size for tower convolutions. Default: 9.
        profile_kernel_size (int): Kernel size for profile head spatial conv. Default: 45.
        dilations (list[int] | None): Dilation schedule for tower layers. Default: all ones (no dilation).
        dropout (float): Dropout rate. Default: 0.1.
        predict_total_count (bool): If True, predicts a single total count scalar. Default: True.
        residual (bool): Whether to use residual connections in tower. Default: True.
        linear_head (bool): If True, profile head is a single linear spatial conv
            (no pointwise + ReLU). More interpretable for attribution. Default: True.
    """

    def __init__(
        self,
        input_len: int = 1128,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        filters: int = 12,
        n_layers: int = 5,
        conv_kernel_size: int | list[int] | None = None,
        dil_kernel_size: int = 9,
        profile_kernel_size: int = 45,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        predict_total_count: bool = True,
        residual: bool = True,
        linear_head: bool = True,
    ):
        super().__init__()
        if input_channels is None:
            input_channels = ["A", "C", "G", "T"]
        if output_channels is None:
            output_channels = ["signal"]
        if conv_kernel_size is None:
            conv_kernel_size = [11, 11]
        if isinstance(conv_kernel_size, int):
            conv_kernel_size = [conv_kernel_size]
        if dilations is None:
            dilations = [1] * n_layers

        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.n_input_channels = len(input_channels)
        self.n_output_channels = len(output_channels)
        self.predict_total_count = predict_total_count
        self.linear_head = linear_head

        # 1. Stem: plain Conv1d + ReLU (valid padding)
        stem_layers: list[nn.Module] = []
        for i, k in enumerate(conv_kernel_size):
            c_in = self.n_input_channels if i == 0 else filters
            stem_layers.append(nn.Conv1d(c_in, filters, k, padding="valid"))
            stem_layers.append(nn.ReLU())
        self.stem = nn.Sequential(*stem_layers)

        # 2. Residual Tower: SimpleResidualBlock stack
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                SimpleResidualBlock(
                    filters,
                    dil_kernel_size,
                    dilation=d,
                    dropout=dropout,
                    residual=residual,
                )
            )

        # 3. Profile Head
        if linear_head:
            # Single linear spatial conv — most interpretable for attribution
            self.profile_spatial = nn.Conv1d(
                filters,
                self.n_output_channels,
                kernel_size=profile_kernel_size,
                padding="valid",
            )
        else:
            # Pointwise → ReLU → spatial conv
            self.profile_pointwise = nn.Conv1d(filters, filters, kernel_size=1)
            self.profile_act = nn.ReLU()
            self.profile_spatial = nn.Conv1d(
                filters,
                self.n_output_channels,
                kernel_size=profile_kernel_size,
                padding="valid",
            )

        # 4. Count Head (MLP)
        num_count_outputs = 1 if predict_total_count else self.n_output_channels
        hidden_dim = max(filters // 2, 4)
        self.count_mlp = nn.Sequential(
            nn.Linear(filters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_count_outputs),
        )

        params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"BiasNet initialized: filters={filters}, n_layers={n_layers}, "
            f"linear_head={linear_head}, residual={residual}, "
            f"params={params:,}"
        )

    @staticmethod
    def compute_shrinkage(
        conv_kernel_size: int | list[int] = [11, 11],  # noqa: B006
        n_layers: int = 5,
        dilations: list[int] | None = None,
        dil_kernel_size: int = 9,
        profile_kernel_size: int = 45,
    ) -> int:
        """Compute total shrinkage (in bp) for BiasNet's valid-padding conv stack.

        Shrinkage = stem + tower + profile_head, where each valid-padding layer
        shrinks by ``dilation * (kernel_size - 1)``.

        Args:
            conv_kernel_size: Stem kernel size(s). Default: ``(11, 11)``.
            n_layers: Number of residual tower layers. Default: ``5``.
            dilations: Dilation schedule. Default: all ones (no dilation).
            dil_kernel_size: Tower conv kernel size. Default: ``9``.
            profile_kernel_size: Profile head kernel size. Default: ``45``.

        Returns:
            Total input-to-output shrinkage in bp.
        """
        if dilations is None:
            dilations = [1] * n_layers
        if isinstance(conv_kernel_size, int):
            stem = conv_kernel_size - 1
        else:
            stem = sum(k - 1 for k in conv_kernel_size)
        tower = sum(d * (dil_kernel_size - 1) for d in dilations)
        head = profile_kernel_size - 1
        return stem + tower + head

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        # Center-crop or reject input based on expected input_len
        if x.shape[-1] > self.input_len:
            crop = (x.shape[-1] - self.input_len) // 2
            x = x[..., crop : crop + self.input_len]
        elif x.shape[-1] < self.input_len:
            raise ValueError(
                f"Input length {x.shape[-1]} is shorter than required {self.input_len}"
            )

        # 1. Stem
        x = self.stem(x)

        # 2. Residual Tower
        for layer in self.layers:
            x = layer(x)

        # 3. Profile Head
        if self.linear_head:
            profile_logits = self.profile_spatial(x)
        else:
            profile_logits = self.profile_pointwise(x)
            profile_logits = self.profile_act(profile_logits)
            profile_logits = self.profile_spatial(profile_logits)

        # 4. Count Head
        x_pooled = x.mean(dim=-1)  # global average pool: (B, F)
        log_counts = self.count_mlp(x_pooled)  # (B, n_out)

        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)
