import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ):
        super().__init__()
        self.res_early = channels_in == channels_out
        self.inv_bottleneckwidth = int(inv_bottleneckscale * channels_out)
        self.dwconv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            padding="same",
            groups=channels_in if groups else 1,
            dilation=dilation_rate,
        )  # depthwise conv
        self.norm = nn.LayerNorm(channels_out, eps=1e-6)
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
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 1)
        x = x_ + x
        return x

class PGCBlock(nn.Module):
    """
    Projected Gated Convolution Block.
    
    Structure:
    1. Projection (Pointwise) -> Expansion
    2. RMSNorm
    3. Split into X, V
    4. Dilated Depthwise Conv on X
    5. Gating: G = X_conv * V
    6. Projection (Pointwise) -> Compression
    7. RMSNorm
    8. Dropout
    9. Residual Connection
    
    Args:
        dim (int): Input/Output dimension.
        kernel_size (int): Kernel size for depthwise convolution.
        dilation (int): Dilation rate.
        expansion (int): Expansion factor. Internal dimension = dim * expansion.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim: int, kernel_size: int, dilation: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
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
            padding='same'
        )
        
        # 4. Output Projection
        self.out_proj = nn.Conv1d(self.hidden_dim, dim, kernel_size=1)
        
        # 5. Norm & Dropout
        self.norm2 = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, L)
        residual = x
        
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
        
        # 5. Gating
        x = x * v
        
        # 6. Output Project
        x = self.out_proj(x)
        
        # 7. Norm & Dropout (requires B, L, C)
        x = x.transpose(1, 2)
        x = self.norm2(x.float()).type_as(x)
        x = x.transpose(1, 2)
        
        x = self.dropout(x)
        
        # 8. Residual
        return residual + x
