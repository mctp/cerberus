import torch

from cerberus.layers import ConvNeXtV2Block, DilatedResidualBlock, GRN1d, PGCBlock


def test_dilated_residual_block_shape():
    filters = 16
    kernel_size = 3
    dilation = 2
    block = DilatedResidualBlock(filters, kernel_size, dilation)
    length = 20
    x = torch.randn(1, filters, length)
    out = block(x)
    # Output length: L_in - dilation * (kernel_size - 1)
    # 20 - 2 * (3-1) = 20 - 4 = 16
    assert out.shape == (1, filters, 16)


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
