import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as _apply_weight_norm

class GRN1d(nn.Module):
    """ 
    ConvNeXt v2 GRN (Global Response Normalization) layer, adapted for 1d.
    
    This layer computes the L2 norm across the temporal dimension and divides
    by the mean norm to normalize channel responses globally.
    
    Reference: facebookresearch/ConvNeXt-V2
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 Block adapted for 1D genomic data.
    
    Structure:
    - Depthwise Conv1d (Spatial mixing)
    - LayerNorm
    - Pointwise Conv1d (Channel expansion)
    - GELU
    - GRN (Global Response Normalization)
    - Pointwise Conv1d (Channel compression)
    - Residual connection
    """
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        groups=False,
        inv_bottleneckscale=4,
        grn=True,
        dilation_rate=1,
        padding="same",
    ):
        super().__init__()
        self.res_early = channels_in == channels_out
        self.inv_bottleneckwidth = int(inv_bottleneckscale * channels_out)
        self.dwconv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels_in if groups else 1,
            dilation=dilation_rate,
        )  # depthwise conv
        ## shift from original LayerNorm to more recent RMSNorm
        self.norm = nn.RMSNorm(channels_out, eps=1e-6)
        self.pwconv1 = nn.Linear(
            channels_out, self.inv_bottleneckwidth
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if grn:
            self.grn = GRN1d(self.inv_bottleneckwidth)
        else:
            self.grn = nn.Identity()
        self.pwconv2 = nn.Linear(self.inv_bottleneckwidth, channels_out)

    def forward(self, x: torch.Tensor):
        if self.res_early:
            x_ = x
            x = self.dwconv(x)
        else:
            x = self.dwconv(x)
            x_ = x

        x = x.permute(0, 2, 1)
        x = self.norm(x.float()).type_as(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 1)
        
        # If residual is active but shapes differ (due to valid padding), crop the residual
        if self.res_early and x_.shape[-1] != x.shape[-1]:
            diff = x_.shape[-1] - x.shape[-1]
            if diff > 0:
                crop_l = diff // 2
                crop_r = diff - crop_l
                x_ = x_[..., crop_l:-crop_r]
            elif diff < 0:
                raise ValueError(f"Output larger than input in Block? In: {x_.shape}, Out: {x.shape}")

        x = x_ + x
        return x

class PGCBlock(nn.Module):
    """
    Projected Gated Convolution Block.

    Full mode (expansion >= 1):
    1. Projection (Pointwise) -> Expansion
    2. RMSNorm
    3. Split into X, V
    4. Dilated Depthwise Conv on X
    5. Gating: G = X_conv * V
    6. Projection (Pointwise) -> Compression
    7. RMSNorm
    8. Dropout
    9. Residual Connection

    Depthwise-only mode (expansion = 0):
    1. RMSNorm
    2. Dilated Depthwise Conv
    3. Dropout
    4. Residual Connection

    No pointwise projections or gating are used in depthwise-only mode.
    The tower features remain independent per-channel — useful for ablation
    studies testing whether inter-channel mixing matters.

    Args:
        dim (int): Input/Output dimension.
        kernel_size (int): Kernel size for depthwise convolution.
        dilation (int): Dilation rate.
        expansion (int): Expansion factor. Internal dimension = dim * expansion.
            When set to 0, the block operates in depthwise-only mode.
        dropout (float): Dropout rate.
        padding (str): Padding mode. Default: 'same'.
    """
    def __init__(self, dim: int, kernel_size: int, dilation: int, expansion: int = 2, dropout: float = 0.1, padding: str = 'same'):
        super().__init__()

        self.dim = dim
        self.depthwise_only = (expansion == 0)

        if self.depthwise_only:
            # Depthwise-only: norm -> conv -> dropout -> residual
            self.norm = nn.RMSNorm(dim)
            self.conv = nn.Conv1d(
                dim, dim,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=dim,
                padding=padding
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.hidden_dim = dim * expansion

            # 1. Input Projection (Expansion)
            # We project to 2 * hidden_dim to split into X and V
            self.in_proj = nn.Conv1d(dim, 2 * self.hidden_dim, kernel_size=1)

            # 2. Norm after projection
            self.norm1 = nn.RMSNorm(2 * self.hidden_dim)

            # 3. Depthwise Conv
            # Applied to X part only (first half)
            self.conv = nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=self.hidden_dim, # Depthwise
                padding=padding
            )

            # 4. Output Projection
            self.out_proj = nn.Conv1d(self.hidden_dim, dim, kernel_size=1)

            # 5. Norm & Dropout
            self.norm2 = nn.RMSNorm(dim)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        residual = x

        if self.depthwise_only:
            # Pre-norm -> depthwise conv -> dropout
            x = x.transpose(1, 2)
            x = self.norm(x.float()).type_as(x)
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = self.dropout(x)
        else:
            # 1. Project
            x = self.in_proj(x)

            # 2. Norm (requires B, L, C)
            x = x.transpose(1, 2) # (B, L, C)
            x = self.norm1(x.float()).type_as(x)
            x = x.transpose(1, 2) # (B, C, L)

            # 3. Split
            x, v = torch.chunk(x, 2, dim=1)

            # 4. Depthwise Conv on X
            x = self.conv(x)

            # If valid padding reduced x size, crop v to match
            if x.shape[-1] != v.shape[-1]:
                diff = v.shape[-1] - x.shape[-1]
                if diff > 0:
                    crop_l = diff // 2
                    crop_r = diff - crop_l
                    v = v[..., crop_l:-crop_r]
                elif diff < 0:
                     raise ValueError(f"Conv output larger than input? {x.shape} vs {v.shape}")

            # 5. Gating
            x = x * v

            # 6. Output Project
            x = self.out_proj(x)

            # 7. Norm & Dropout (requires B, L, C)
            x = x.transpose(1, 2)
            x = self.norm2(x.float()).type_as(x)
            x = x.transpose(1, 2)

            x = self.dropout(x)

        # Residual connection
        # If valid padding reduced x size, crop residual to match
        if x.shape[-1] != residual.shape[-1]:
            diff = residual.shape[-1] - x.shape[-1]
            if diff > 0:
                crop_l = diff // 2
                crop_r = diff - crop_l
                residual = residual[..., crop_l:-crop_r]
            elif diff < 0:
                 raise ValueError(f"Output larger than residual? {x.shape} vs {residual.shape}")

        return residual + x

class DilatedResidualBlock(nn.Module):
    """
    Dilated Residual Block for BPNet.

    Structure: Input -> Dilated Conv -> Activation -> Add to cropped Input

    Args:
        filters (int): Number of input and output channels.
        kernel_size (int): Kernel size for the dilated convolution.
        dilation (int): Dilation rate.
        activation (str): Activation function. One of ``"relu"`` or ``"gelu"``.
            Default: ``"relu"``.
        weight_norm (bool): If True, applies :func:`torch.nn.utils.weight_norm`
            to the convolution. Decouples weight magnitude from direction, which
            stabilizes gradient norms across deep dilated stacks and enables
            effective AdamW weight decay. Safe for DeepLIFT/DeepSHAP: weight
            normalization is a weight reparameterization, not an activation
            nonlinearity, so the Conv1d linear passthrough rule is unchanged.
            Default: ``False``.
    """
    def __init__(self, filters: int, kernel_size: int, dilation: int,
                 activation: str = "relu", weight_norm: bool = False):
        super().__init__()
        conv = nn.Conv1d(
            filters, filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding='valid'
        )
        if weight_norm:
            _apply_weight_norm(conv)
        self.conv = conv
        if activation == "relu":
            self.act: nn.Module = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"DilatedResidualBlock: unsupported activation {activation!r}. "
                "Must be 'relu' or 'gelu'."
            )

    def forward(self, x):
        out = self.act(self.conv(x))
        # Center crop x to match out due to 'valid' padding shrinking the sequence.
        diff = x.shape[-1] - out.shape[-1]
        if diff > 0:
            crop_l = diff // 2
            crop_r = diff - crop_l
            x = x[..., crop_l:-crop_r]
        return x + out


class SimpleResidualBlock(nn.Module):
    """Conv1d + ReLU residual block with valid padding.

    Structure: Conv1d(dim, dim, k, dilation=d) → ReLU → Dropout [+ Residual]

    With residual=True (default), the residual input is center-cropped to match
    the valid-padded output and added back. All operations are nn.Module-based
    (no ``F.relu``) for full DeepLIFT/DeepSHAP compatibility via captum.

    Args:
        dim: Number of input/output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation rate. Default: 1.
        dropout: Dropout rate. Default: 0.1.
        residual: Whether to add a residual connection. Default: True.
    """
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.shrinkage = dilation * (kernel_size - 1)
        self.residual = residual
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=0)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv(x))
        out = self.dropout(out)
        if self.residual:
            # Center-crop residual to match valid-padded output
            if self.shrinkage > 0:
                crop = self.shrinkage // 2
                x = x[..., crop: crop + out.shape[-1]]
            return out + x
        return out
