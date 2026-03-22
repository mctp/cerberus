"""Regression tests: all models center-crop oversized inputs and reject undersized inputs."""

import pytest
import torch

from cerberus.models.asap import ConvNeXtDCNN
from cerberus.models.bpnet import BPNet
from cerberus.models.gopher import GlobalProfileCNN
from cerberus.models.pomeranian import Pomeranian

# --- Pomeranian ---


class TestPomeranianInputCrop:
    def test_exact_input(self):
        model = Pomeranian(input_len=2112, output_len=1024)
        model.eval()
        x = torch.randn(1, 4, 2112)
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (1, 1, 1024)

    def test_oversized_input_cropped(self):
        model = Pomeranian(input_len=2112, output_len=1024)
        model.eval()
        x_exact = torch.randn(1, 4, 2112)
        x_over = torch.zeros(1, 4, 2212)
        # Embed exact input centered in the oversized tensor
        x_over[..., 50 : 50 + 2112] = x_exact
        with torch.no_grad():
            out_exact = model(x_exact)
            out_over = model(x_over)
        assert out_over.logits.shape == out_exact.logits.shape
        assert torch.allclose(out_exact.logits, out_over.logits, atol=1e-5)
        assert torch.allclose(out_exact.log_counts, out_over.log_counts, atol=1e-5)

    def test_undersized_input_raises(self):
        model = Pomeranian(input_len=2112, output_len=1024)
        x = torch.randn(1, 4, 2000)
        with pytest.raises(ValueError, match="shorter than required"):
            model(x)


# --- BPNet ---


class TestBPNetInputCrop:
    def test_exact_input(self):
        model = BPNet(input_len=2114, output_len=1000)
        model.eval()
        x = torch.randn(1, 4, 2114)
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (1, 1, 1000)

    def test_oversized_input_cropped(self):
        model = BPNet(input_len=2114, output_len=1000)
        model.eval()
        x_exact = torch.randn(1, 4, 2114)
        x_over = torch.zeros(1, 4, 2214)
        x_over[..., 50 : 50 + 2114] = x_exact
        with torch.no_grad():
            out_exact = model(x_exact)
            out_over = model(x_over)
        assert out_over.logits.shape == out_exact.logits.shape
        assert torch.allclose(out_exact.logits, out_over.logits, atol=1e-5)
        assert torch.allclose(out_exact.log_counts, out_over.log_counts, atol=1e-5)

    def test_undersized_input_raises(self):
        model = BPNet(input_len=2114, output_len=1000)
        x = torch.randn(1, 4, 2000)
        with pytest.raises(ValueError, match="shorter than required"):
            model(x)


# --- GlobalProfileCNN (Gopher) ---


class TestGopherInputCrop:
    def test_exact_input(self):
        model = GlobalProfileCNN(input_len=2048, output_len=1024, output_bin_size=4)
        model.eval()
        x = torch.randn(1, 4, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.log_rates.shape[-1] == 1024 // 4

    def test_oversized_input_cropped(self):
        model = GlobalProfileCNN(input_len=2048, output_len=1024, output_bin_size=4)
        model.eval()
        x_exact = torch.randn(1, 4, 2048)
        x_over = torch.zeros(1, 4, 2176)
        # Center the exact input: (2176 - 2048) // 2 = 64
        x_over[..., 64 : 64 + 2048] = x_exact
        with torch.no_grad():
            out_exact = model(x_exact)
            out_over = model(x_over)
        assert out_over.log_rates.shape == out_exact.log_rates.shape
        assert torch.allclose(out_exact.log_rates, out_over.log_rates, atol=1e-5)

    def test_undersized_input_raises(self):
        model = GlobalProfileCNN(input_len=2048, output_len=1024, output_bin_size=4)
        x = torch.randn(1, 4, 1900)
        with pytest.raises(ValueError, match="shorter than required"):
            model(x)


# --- ConvNeXtDCNN (ASAP) ---


class TestASAPInputCrop:
    def test_exact_input(self):
        model = ConvNeXtDCNN(input_len=2048, output_len=1024, output_bin_size=4)
        model.eval()
        x = torch.randn(1, 4, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.log_rates.shape[0] == 1

    def test_oversized_input_cropped(self):
        model = ConvNeXtDCNN(input_len=2048, output_len=1024, output_bin_size=4)
        model.eval()
        x_exact = torch.randn(1, 4, 2048)
        x_over = torch.zeros(1, 4, 2176)
        x_over[..., 64 : 64 + 2048] = x_exact
        with torch.no_grad():
            out_exact = model(x_exact)
            out_over = model(x_over)
        assert out_over.log_rates.shape == out_exact.log_rates.shape
        assert torch.allclose(out_exact.log_rates, out_over.log_rates, atol=1e-5)

    def test_undersized_input_raises(self):
        model = ConvNeXtDCNN(input_len=2048, output_len=1024, output_bin_size=4)
        x = torch.randn(1, 4, 1900)
        with pytest.raises(ValueError, match="shorter than required"):
            model(x)
