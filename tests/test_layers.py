
import torch
import torch.nn as nn
import pytest
from cerberus.layers import GRN1d, ConvNeXtV2Block, PGCBlock

def test_grn1d():
    dim = 32
    layer = GRN1d(dim)
    # GRN1d expects (Batch, Length, Channels) because it is used inside ConvNeXt block after permutation
    x = torch.randn(2, 100, dim)
    out = layer(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_convnextv2_block_shape_same_channels():
    dim = 32
    # channels_in = channels_out
    block = ConvNeXtV2Block(channels_in=dim, channels_out=dim, kernel_size=7)
    x = torch.randn(2, dim, 50)
    out = block(x)
    assert out.shape == x.shape
    
def test_convnextv2_block_channel_change():
    c_in = 16
    c_out = 32
    block = ConvNeXtV2Block(channels_in=c_in, channels_out=c_out, kernel_size=7)
    x = torch.randn(2, c_in, 50)
    out = block(x)
    assert out.shape == (2, c_out, 50)

def test_pgc_block_shape():
    dim = 16
    expansion = 2
    block = PGCBlock(dim, kernel_size=3, dilation=1, expansion=expansion)
    length = 20
    x = torch.randn(1, dim, length)
    out = block(x)
    assert out.shape == (1, dim, length)
