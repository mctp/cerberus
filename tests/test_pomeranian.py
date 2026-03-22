import pytest
import torch

from cerberus.layers import PGCBlock
from cerberus.models.pomeranian import Pomeranian, PomeranianK5


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_pomeranian_default_initialization():
    # Test Default (K9 Config)
    model = Pomeranian()
    # Check dimensions
    assert model.input_len == 2112
    assert model.output_len == 1024
    
    # Check Stem (Factorized [11, 11] - Matching K5)
    assert isinstance(model.stem, torch.nn.Sequential)
    assert len(model.stem) == 2
    assert model.stem[0].dwconv.kernel_size == (11,) # type: ignore
    assert model.stem[1].dwconv.kernel_size == (11,) # type: ignore
    assert model.stem[1].dwconv.groups == 64 # Depthwise # type: ignore
    
    # Check Body (K=9, 8 Layers)
    assert len(model.layers) == 8
    for layer in model.layers:
        assert layer.conv.kernel_size == (9,) # type: ignore
        
    # Check Head (K=45)
    assert model.profile_spatial.kernel_size == (45,) # type: ignore

def test_pomeranian_k5_initialization():
    model = PomeranianK5()
    # Check dimensions
    assert model.input_len == 2112
    assert model.output_len == 1024
    
    # Check Stem (Factorized [11, 11])
    # The stem is an nn.Sequential
    assert isinstance(model.stem, torch.nn.Sequential)
    assert len(model.stem) == 2
    # Layer 0: ConvNeXtV2Block (K=11, Dense)
    assert model.stem[0].dwconv.kernel_size == (11,) # type: ignore
    assert model.stem[0].dwconv.groups == 1 # Dense (First layer) # type: ignore
    # Layer 1: ConvNeXtV2Block (K=11, Depthwise)
    assert model.stem[1].dwconv.kernel_size == (11,) # type: ignore
    assert model.stem[1].dwconv.groups == 64 # Depthwise (Second layer) # type: ignore
    
    # Check Body (K=5, 8 Layers)
    assert len(model.layers) == 8
    for layer in model.layers:
        assert layer.conv.kernel_size == (5,) # type: ignore
        
    # Check Head (K=49)
    assert model.profile_spatial.kernel_size == (49,) # type: ignore

def test_pomeranian_default_shape():
    model = Pomeranian()
    x = torch.randn(2, 4, 2112)
    output = model(x)
    assert output.logits.shape == (2, 1, 1024)
    assert output.log_counts.shape == (2, 1)

def test_pomeranian_k5_shape():
    model = PomeranianK5()
    x = torch.randn(2, 4, 2112)
    output = model(x)
    assert output.logits.shape == (2, 1, 1024)
    assert output.log_counts.shape == (2, 1)

def test_parameter_counts():
    model_k5 = PomeranianK5()
    model_default = Pomeranian()
    
    params_k5 = count_params(model_k5)
    params_default = count_params(model_default)
    
    print(f"K5 Params: {params_k5}")
    print(f"Default (K9) Params: {params_default}")
    
    # K5 should be around 151k
    assert 145000 < params_k5 < 160000
    
    # Default (K9) should be similar
    assert 145000 < params_default < 165000

def test_geometric_alignment_default():
    # Verify exact alignment math for Default (K9)
    # Input: 2112
    # Stem: 11, 11. Shrinkage: 20.
    # Body: K=9. Dilations: 1, 1, 2, 4, 8, 16, 32, 64. Sum=128.
    # Shrinkage per layer: (9-1) * dilation = 8 * dilation.
    # Total Body Shrinkage: 8 * 128 = 1024.
    # Head: K=45. Shrinkage: 44.
    # Total Shrinkage: 20 + 1024 + 44 = 1088.
    # Output: 2112 - 1088 = 1024.
    
    model = Pomeranian()
    x = torch.randn(1, 4, 2112)
    output = model(x)
    assert output.logits.shape[-1] == 1024

def test_geometric_alignment_k5():
    # Verify exact alignment math for K5
    # Input: 2112
    # Stem: 11, 11. Shrinkage: (11-1) + (11-1) = 20.
    # Body: K=5. Dilations: 1, 2, 4, 8, 16, 32, 64, 128. Sum=255.
    # Shrinkage per layer: (5-1) * dilation = 4 * dilation.
    # Total Body Shrinkage: 4 * 255 = 1020.
    # Head: K=49. Shrinkage: 48.
    # Total Shrinkage: 20 + 1020 + 48 = 1088.
    # Output: 2112 - 1088 = 1024.
    
    model = PomeranianK5()
    x = torch.randn(1, 4, 2112)
    output = model(x)
    assert output.logits.shape[-1] == 1024


def test_pomeranian_dilations_n_dilated_layers_mismatch():
    """Regression: n_dilated_layers must match len(dilations) when both are provided."""
    with pytest.raises(ValueError, match="n_dilated_layers=16 conflicts with len\\(dilations\\)=4"):
        Pomeranian(n_dilated_layers=16, dilations=[1, 2, 4, 8])


# --- Depthwise-only mode (expansion=0) ---

class TestPGCBlockDepthwiseOnly:
    """Tests for PGCBlock depthwise-only mode (expansion=0)."""

    def test_shape_same_padding(self):
        block = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=0, padding='same')
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_shape_valid_padding(self):
        block = PGCBlock(dim=16, kernel_size=9, dilation=4, expansion=0, padding='valid')
        x = torch.randn(2, 16, 128)
        out = block(x)
        # Shrinkage = dilation * (kernel_size - 1) = 4 * 8 = 32
        assert out.shape == (2, 16, 128 - 32)

    def test_no_pointwise_params(self):
        block = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=0)
        assert not hasattr(block, 'in_proj')
        assert not hasattr(block, 'out_proj')
        assert not hasattr(block, 'norm1')
        assert not hasattr(block, 'norm2')
        assert hasattr(block, 'norm')
        assert hasattr(block, 'conv')

    def test_fewer_params_than_full(self):
        dw_only = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=0)
        full_e1 = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=1)
        full_e2 = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=2)
        p_dw = count_params(dw_only)
        p_e1 = count_params(full_e1)
        p_e2 = count_params(full_e2)
        assert p_dw < p_e1 < p_e2

    def test_depthwise_only_flag(self):
        block_dw = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=0)
        block_full = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=1)
        assert block_dw.depthwise_only is True
        assert block_full.depthwise_only is False

    def test_conv_is_depthwise(self):
        block = PGCBlock(dim=16, kernel_size=9, dilation=1, expansion=0)
        assert block.conv.groups == 16  # groups == dim == depthwise

    def test_gradient_flows(self):
        block = PGCBlock(dim=8, kernel_size=5, dilation=2, expansion=0, padding='valid')
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestPomeranianDepthwiseOnly:
    """Tests for Pomeranian with expansion=0 (depthwise-only tower)."""

    def test_shape_default(self):
        model = Pomeranian(expansion=0)
        x = torch.randn(2, 4, 2112)
        output = model(x)
        assert output.logits.shape == (2, 1, 1024)
        assert output.log_counts.shape == (2, 1)

    def test_shape_k5(self):
        model = PomeranianK5(expansion=0)
        x = torch.randn(2, 4, 2112)
        output = model(x)
        assert output.logits.shape == (2, 1, 1024)
        assert output.log_counts.shape == (2, 1)

    def test_geometric_alignment(self):
        # Shrinkage is identical regardless of expansion
        # (depthwise conv kernel/dilation unchanged)
        model = Pomeranian(expansion=0)
        x = torch.randn(1, 4, 2112)
        output = model(x)
        assert output.logits.shape[-1] == 1024

    def test_fewer_params(self):
        model_dw = Pomeranian(expansion=0)
        model_e1 = Pomeranian(expansion=1)
        p_dw = count_params(model_dw)
        p_e1 = count_params(model_e1)
        # Depthwise-only should have significantly fewer params
        assert p_dw < p_e1
        # Stem and head are the same, only tower differs
        # With 64 filters, 8 layers: each full PGC block has ~8K pointwise params
        # depthwise-only removes all of those
        assert p_dw < p_e1 * 0.5

    def test_tower_blocks_are_depthwise_only(self):
        model = Pomeranian(expansion=0)
        for layer in model.layers:
            assert layer.depthwise_only is True  # type: ignore

    def test_custom_filters(self):
        # Small model like the bias experiments (filters=8)
        model = Pomeranian(
            filters=8, n_dilated_layers=10, expansion=0,
            dilations=[1, 1, 2, 4, 8, 16, 32, 64, 128, 256],
            dil_kernel_size=9, conv_kernel_size=[11, 11],
            profile_kernel_size=45,
            input_len=5186, output_len=1024, stem_expansion=1,
        )
        x = torch.randn(1, 4, 5186)
        output = model(x)
        assert output.logits.shape == (1, 1, 1024)
        assert output.log_counts.shape == (1, 1)

    def test_expansion_one_unchanged(self):
        """Regression: expansion=1 still works identically."""
        model = Pomeranian(expansion=1)
        for layer in model.layers:
            assert layer.depthwise_only is False  # type: ignore
            assert hasattr(layer, 'in_proj')
            assert hasattr(layer, 'out_proj')
