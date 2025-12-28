
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from cerberus.models.asap import ConvNeXtDCNN as CerberusConvNeXtDCNN

# ==========================================
# Original Implementation (Copied Verbatim)
# ==========================================

class OriginalGRN1d(nn.Module):
    """ ConvNeXt v2 GRN (Global Response Normalization) layer, adapted for 1d

    via facebookresearch/ConvNeXt-V2
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class OriginalConvNeXtV2Block(nn.Module):
    """Adapted from ConvNeXt"""
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
            self.grn = OriginalGRN1d(self.inv_bottleneckwidth)
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

class OriginalUnmapPredictor(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.out = nn.Linear(channels_in, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.activation(self.out(x))

DEFAULT = {
    'window_size': 2048,
    'bin_size': 4,
    'residual_blocks': 11,
    'dilation_mult': 1.5,
    'filters0': 256,
    'filters1': 128,
    'filters3': 2048,
    'kernel0': 15,
    'kernel1': 3,
    'kernel2': 1,
    'dropout': 0.3,
    'final_dropout': 0.05,
    'use_map': False
}

class OriginalConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out:int=1 , kernel_size:int=1 , dilation_rate:int=1, bn_gamma=None) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size,
            bias=False, # no need if batchnorm after conv layer
            dilation=dilation_rate,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(channels_out, momentum=0.1)
        if bn_gamma is not None:
            if bn_gamma == 'zeros':
                self.bn.weight = nn.Parameter(torch.zeros_like(self.bn.weight))
            elif bn_gamma == 'ones':
                # default of BatchNorm1d
                # but let's be explicit
                self.bn.weight = nn.Parameter(torch.ones_like(self.bn.weight))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class OriginalBasenjiCoreBlock(nn.Module):
    def __init__(self, nr_tracks: int, window: int, filters_in,
                  nr_res_blocks: int = 11, rate_mult: float = 1.5, bin_size: int = 100, filters1: int = 128,
                  filters3: Optional[int] = None, kernel1: int = 3, kernel2: int = 1, dropout: float = 0.3,
                  final_dropout: float = 0.05):
        super().__init__()
        if not filters3:
            filters3 = window
        dconv_blocks = []
        conv_blocks = []
        dilation_rate = 1.0
        self.nr_res_blocks = nr_res_blocks
        self.dropout = nn.Dropout(p=dropout)
        for _ in range(self.nr_res_blocks):
            d_conv_block = OriginalConvBlock(filters_in, filters1, kernel_size=kernel1, dilation_rate=int(np.round(dilation_rate)))
            dconv_blocks.append(d_conv_block)
            conv_block = OriginalConvBlock(filters1, filters_in,  kernel_size=kernel2, bn_gamma='zeros')
            conv_blocks.append(conv_block)
            dilation_rate *= rate_mult

        self.dconv_blocks = nn.ModuleList(dconv_blocks)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.final_conv = OriginalConvBlock(filters_in, channels_out=filters3)
        self.final_dropout = nn.Dropout(p=final_dropout)
        pool_size = bin_size // 2
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

        self.linear_out = nn.Linear(
            in_features=filters3,
            out_features=nr_tracks,
        )
        self.activation = nn.Softplus()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for i in range(self.nr_res_blocks):
            last_block_x = x
            # dilated conv
            x = self.dconv_blocks[i](x)

            # normal conv
            x = self.conv_blocks[i](x)
            x = self.dropout(x)

            # add residual
            x = x + last_block_x
        x = self.final_conv(x)
        x = self.final_dropout(x)
        x = self.pool(x)
        x = torch.transpose(x, -2, -1)
        x = self.linear_out(x)
        x = self.activation(x)
        return x

class OriginalConvNeXtDCNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        config = DEFAULT.copy()
        config.update(kwargs)

        window = config['window_size']
        self.use_map = config['use_map']
        self.init_conv = OriginalConvNeXtV2Block(channels_in=4, channels_out=config['filters0'],
                                   kernel_size=config['kernel0'])
        self.init_pool = nn.MaxPool1d(kernel_size=2)

        if self.use_map:
            self.unmap_predictor = OriginalUnmapPredictor(channels_in=config['filters0'])

        self.core = OriginalBasenjiCoreBlock(nr_tracks=1, window=window, filters_in=config['filters0'],
                      nr_res_blocks=config['residual_blocks'],
                      rate_mult=config['dilation_mult'],
                      bin_size=config['bin_size'],
                      filters1=config['filters1'],
                      filters3=config['filters3'],
                      kernel1=config['kernel1'],
                      kernel2=config['kernel2'],
                      dropout=config['dropout'],
                      final_dropout=config['final_dropout'])
    
    def forward(self, x:torch.Tensor, return_unmap=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.transpose(x, dim0=-1, dim1=-2)
        x = self.init_conv(x)
        x = F.pad(x, (1, 0))
        x = self.init_pool(x)
        if return_unmap:
            u = F.max_pool1d(x, 2)
            u = self.unmap_predictor(torch.transpose(u, dim0=-1, dim1=-2))
        x = self.core(x)
        if return_unmap:
            return x, u
        return x

# ==========================================
# Tests
# ==========================================

def test_convnext_dcnn_equivalence():
    # Configuration
    input_len = 2048
    output_bin_size = 4
    input_channels = ["A", "C", "G", "T"]
    output_channels = ["signal"] # 1 output track
    
    # Initialize Original Model
    # Original config: window_size=2048, bin_size=4, etc.
    # We use default params matching what we put in Cerberus model
    original_model = OriginalConvNeXtDCNN(
        window_size=input_len,
        bin_size=output_bin_size,
        filters3=2048, 
        use_map=False
    )
    original_model.eval()

    # Initialize Cerberus Model
    cerberus_model = CerberusConvNeXtDCNN(
        input_len=input_len,
        output_len=input_len // output_bin_size, # 512
        output_bin_size=output_bin_size,
        input_channels=input_channels,
        output_channels=output_channels,
        filters3=2048 
    )
    cerberus_model.eval()

    # Transfer weights
    original_state = original_model.state_dict()
    cerberus_state = cerberus_model.state_dict()
    
    new_state_dict = {}
    for k, v in original_state.items():
        if k in cerberus_state:
            new_state_dict[k] = v
    
    # Verify we are not missing important keys
    for k in cerberus_state:
        assert k in original_state, f"Key {k} in cerberus model not found in original"
    
    # Handle weight mismatch for linear_out (Linear vs Conv1d)
    # Cerberus uses Conv1d(kernel_size=1) to avoid transposes, Original uses Linear
    if 'core.linear_out.weight' in new_state_dict:
        w = new_state_dict['core.linear_out.weight']
        if w.ndim == 2:
             new_state_dict['core.linear_out.weight'] = w.unsqueeze(-1)
            
    cerberus_model.load_state_dict(new_state_dict, strict=True)
    
    # Input
    batch_size = 2
    # Cerberus input: (B, C, L)
    x_cerberus = torch.randn(batch_size, len(input_channels), input_len)
    
    # Original input: (B, L, C) - deduced from forward pass analysis
    x_original = x_cerberus.transpose(1, 2)
    
    # Forward pass
    with torch.no_grad():
        y_cerberus = cerberus_model(x_cerberus)
        y_original = original_model(x_original, return_unmap=False)
        
    # Compare outputs
    # Cerberus output: (B, Tracks, L_out)
    # Original output: (B, L_out, Tracks)
    
    y_original_transposed = y_original.transpose(1, 2)
    
    print(f"Cerberus Output Shape: {y_cerberus.shape}")
    print(f"Original Output Shape: {y_original.shape}")
    
    # Check shapes
    assert y_cerberus.shape == y_original_transposed.shape, f"Shape mismatch: {y_cerberus.shape} vs {y_original_transposed.shape}"
    
    # Check values
    diff = (y_cerberus - y_original_transposed).abs().max().item()
    print(f"Max difference: {diff}")
    
    assert np.isclose(diff, 0, atol=1e-5), f"Outputs do not match. Max diff: {diff}"

if __name__ == "__main__":
    test_convnext_dcnn_equivalence()
