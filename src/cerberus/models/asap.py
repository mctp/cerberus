from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _GRN1d(nn.Module):
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

class _ConvNeXtV2Block(nn.Module):
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
            self.grn = _GRN1d(self.inv_bottleneckwidth)
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

class _ConvBlock(nn.Module):
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

class _BasenjiCoreBlock(nn.Module):
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
            d_conv_block = _ConvBlock(filters_in, filters1, kernel_size=kernel1, dilation_rate=int(np.round(dilation_rate)))
            dconv_blocks.append(d_conv_block)
            conv_block = _ConvBlock(filters1, filters_in,  kernel_size=kernel2, bn_gamma='zeros')
            conv_blocks.append(conv_block)
            dilation_rate *= rate_mult

        self.dconv_blocks = nn.ModuleList(dconv_blocks)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.final_conv = _ConvBlock(filters_in, channels_out=filters3)
        self.final_dropout = nn.Dropout(p=final_dropout)
        pool_size = bin_size // 2
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

        self.linear_out = nn.Conv1d(
            in_channels=filters3,
            out_channels=nr_tracks,
            kernel_size=1
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
        # x = torch.transpose(x, -2, -1)  <-- Removed transpose
        x = self.linear_out(x)
        x = self.activation(x)
        return x

class ConvNeXtDCNN(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 4,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        **kwargs
    ):
        super().__init__()
        
        # Default config from original model
        config = {
            'window_size': input_len,
            'bin_size': output_bin_size,
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
        }
        config.update(kwargs)
        
        # Determine number of input channels
        num_input_channels = len(input_channels)
        num_output_channels = len(output_channels)
        
        self.init_conv = _ConvNeXtV2Block(
            channels_in=num_input_channels, 
            channels_out=config['filters0'],
            kernel_size=config['kernel0']
        )
        self.init_pool = nn.MaxPool1d(kernel_size=2)

        self.core = _BasenjiCoreBlock(
            nr_tracks=num_output_channels, 
            window=config['window_size'], 
            filters_in=config['filters0'],
            nr_res_blocks=config['residual_blocks'],
            rate_mult=config['dilation_mult'],
            bin_size=config['bin_size'],
            filters1=config['filters1'],
            filters3=config['filters3'],
            kernel1=config['kernel1'],
            kernel2=config['kernel2'],
            dropout=config['dropout'],
            final_dropout=config['final_dropout']
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # Input x: (Batch, Channels, Length)
        
        # Original model did transpose here because it expected (Batch, Length, Channels)
        # But cerberus provides (Batch, Channels, Length).
        # ConvNeXtV2Block (init_conv) expects (Batch, Channels, Length).
        # So we do NOT transpose here.
        
        x = self.init_conv(x)
        x = F.pad(x, (1, 0))
        x = self.init_pool(x)
        
        x = self.core(x)
        
        # _BasenjiCoreBlock now returns (Batch, Tracks, Length)
        # No transpose needed
        
        return (x,)
