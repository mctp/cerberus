"""Coverage tests for cerberus.layers — untested code paths."""
import pytest
import torch
from cerberus.layers import (
    GRN1d,
    ConvNeXtV2Block,
    PGCBlock,
    DilatedResidualBlock,
)


# ---------------------------------------------------------------------------
# GRN1d
# ---------------------------------------------------------------------------

class TestGRN1d:

    def test_forward_shape(self):
        torch.manual_seed(80)
        grn = GRN1d(dim=32)
        # GRN1d expects (B, L, C) based on the ConvNeXtV2Block usage
        x = torch.randn(2, 10, 32)
        out = grn(x)
        assert out.shape == (2, 10, 32)

    def test_gradients_flow(self):
        grn = GRN1d(dim=16)
        x = torch.randn(1, 5, 16, requires_grad=True)
        out = grn(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# ConvNeXtV2Block
# ---------------------------------------------------------------------------

class TestConvNeXtV2Block:

    def test_same_channels_residual(self):
        """channels_in == channels_out uses res_early=True."""
        torch.manual_seed(81)
        block = ConvNeXtV2Block(channels_in=32, channels_out=32, kernel_size=7)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_different_channels_no_res_early(self):
        """channels_in != channels_out uses res_early=False."""
        torch.manual_seed(82)
        block = ConvNeXtV2Block(channels_in=16, channels_out=32, kernel_size=7)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_grn_false(self):
        """grn=False replaces GRN with Identity."""
        torch.manual_seed(83)
        block = ConvNeXtV2Block(channels_in=32, channels_out=32, kernel_size=7, grn=False)
        assert isinstance(block.grn, torch.nn.Identity)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_with_dilation(self):
        block = ConvNeXtV2Block(channels_in=32, channels_out=32, kernel_size=7, dilation_rate=2)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)


# ---------------------------------------------------------------------------
# PGCBlock
# ---------------------------------------------------------------------------

class TestPGCBlock:

    def test_forward_shape(self):
        torch.manual_seed(84)
        block = PGCBlock(dim=32, kernel_size=7, dilation=1)
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_different_expansion(self):
        block = PGCBlock(dim=16, kernel_size=5, dilation=2, expansion=4)
        x = torch.randn(1, 16, 32)
        out = block(x)
        assert out.shape == (1, 16, 32)

    def test_gradients_flow(self):
        block = PGCBlock(dim=16, kernel_size=5, dilation=1, dropout=0.0)
        block.eval()
        x = torch.randn(1, 16, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# DilatedResidualBlock
# ---------------------------------------------------------------------------

class TestDilatedResidualBlock:

    def test_forward_shape_and_residual_cropping(self):
        """Valid padding shrinks output; residual is center-cropped to match."""
        torch.manual_seed(85)
        block = DilatedResidualBlock(filters=32, kernel_size=7, dilation=1)
        x = torch.randn(2, 32, 64)
        out = block(x)
        # Valid conv with kernel_size=7, dilation=1: output_len = 64 - 6 = 58
        assert out.shape == (2, 32, 58)

    def test_with_dilation(self):
        block = DilatedResidualBlock(filters=16, kernel_size=3, dilation=4)
        x = torch.randn(1, 16, 100)
        out = block(x)
        # Valid conv: kernel_size=3, dilation=4: effective_kernel = 3 + (3-1)*(4-1) = 9
        # output_len = 100 - 8 = 92
        assert out.shape == (1, 16, 92)

    def test_gradients_flow(self):
        block = DilatedResidualBlock(filters=16, kernel_size=3, dilation=1)
        x = torch.randn(1, 16, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
