import torch
import pytest
from cerberus.models.pomeranian import Pomeranian, PomeranianK5
from cerberus.layers import ConvNeXtV2Block, PGCBlock

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
