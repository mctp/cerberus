import torch
import pytest
from cerberus.models.pomeranian import Pomeranian, Pomeranian1k
from cerberus.layers import ConvNeXtV2Block, PGCBlock

def test_pomeranian_defaults():
    # Test generic Pomeranian defaults (BPNet standard)
    model = Pomeranian(input_len=2114, output_len=1000)
    assert model.stem.dwconv.padding == 'valid'
    # Check that profile kernel is 75 (default for base class)
    assert model.profile_spatial.kernel_size == (75,)

def test_pomeranian1k_initialization():
    model = Pomeranian1k()
    assert model.input_len == 2112
    assert model.output_len == 1024
    assert model.n_input_channels == 4
    assert model.n_output_channels == 1
    # Check profile kernel is 49
    assert model.profile_spatial.kernel_size == (49,)
    
    # Check Heads
    assert hasattr(model, 'profile_pointwise')
    assert hasattr(model, 'profile_act')
    assert hasattr(model, 'profile_spatial')
    assert hasattr(model, 'count_mlp')
    
    # Check Count Head structure
    assert isinstance(model.count_mlp, torch.nn.Sequential)
    assert len(model.count_mlp) == 3 # Linear -> GELU -> Linear

def test_pomeranian1k_forward_shape():
    # Input: 2112, Output: 1024. Diff: 1088.
    # Stem (k=21) -> 20 shrinkage
    # 8 layers dilated (2, 4, ..., 256) -> sum(2 * 2^i) -> 1020 shrinkage
    # Profile Head (1x1 -> 49) -> 48 shrinkage
    # Total = 20 + 1020 + 48 = 1088.
    
    model = Pomeranian1k()
    input_len = model.input_len
    output_len = model.output_len
    
    x = torch.randn(2, 4, input_len)
    output = model(x)
    
    assert output.logits.shape == (2, 1, output_len)
    assert output.log_counts.shape == (2, 1)

def test_pomeranian1k_parameter_count():
    model = Pomeranian1k()
    total_params = sum(p.numel() for p in model.parameters())
    
    # Expected: ~150k
    # Allow some margin for head changes (Strategy A+C added ~6k)
    # 142k (base) + 6k = 148k.
    assert 140000 < total_params < 160000
    print(f"Total params: {total_params}")

def test_pomeranian_small_input():
    # Test with smallest possible input (using base class with k=49 explicitly or just logic)
    # Using Pomeranian1k configuration (shrinkage 1088)
    # Min input 1089 -> output 1
    input_len = 1100
    output_len = 1100 - 1088 # 12
    
    # We must use Pomeranian base with k=49 to match this shrinkage calculation,
    # or use Pomeranian1k but override lengths.
    model = Pomeranian1k(input_len=input_len, output_len=output_len)
    
    x = torch.randn(1, 4, input_len)
    output = model(x)
    assert output.logits.shape == (1, 1, output_len)

def test_layers_padding_same_regression():
    # Verify that padding='same' still works as expected (GemiNet usage)
    block = ConvNeXtV2Block(channels_in=64, channels_out=64, kernel_size=21, padding='same')
    x = torch.randn(2, 64, 100)
    out = block(x)
    assert out.shape == (2, 64, 100) # Length preserved
    
    pgc = PGCBlock(dim=64, kernel_size=3, dilation=2, padding='same')
    out_pgc = pgc(x)
    assert out_pgc.shape == (2, 64, 100) # Length preserved

def test_layers_padding_valid():
    # Verify padding='valid' works and shrinks correctly
    
    # ConvNeXtV2Block
    # k=21 -> shrinkage 20
    block = ConvNeXtV2Block(channels_in=64, channels_out=64, kernel_size=21, padding='valid')
    x = torch.randn(2, 64, 100)
    out = block(x)
    assert out.shape == (2, 64, 80) # 100 - 20
    
    # PGCBlock
    # k=3, d=2 -> (3-1)*2 = 4 shrinkage
    pgc = PGCBlock(dim=64, kernel_size=3, dilation=2, padding='valid')
    x = torch.randn(2, 64, 100)
    out_pgc = pgc(x)
    assert out_pgc.shape == (2, 64, 96) # 100 - 4

def test_pomeranian1k_internal_layer_length():
    # Verify length of the final PGC layer output (before heads)
    # Input: 2112
    # Stem shrinkage: 20
    # Tower shrinkage: 1020
    # Expected length: 2112 - 20 - 1020 = 1072
    
    model = Pomeranian1k()
    x = torch.randn(1, 4, 2112)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hook on the last layer of the tower
    # layers is a ModuleList, so we access the last one [-1]
    model.layers[-1].register_forward_hook(get_activation('last_layer'))
    
    _ = model(x)
    
    assert 'last_layer' in activations
    internal_shape = activations['last_layer'].shape
    print(f"Internal layer shape: {internal_shape}")
    
    # Expected shape: (1, 64, 1072)
    assert internal_shape == (1, 64, 1072)
    
    # Also verify that this matches the input to profile head
    # Profile head has k=49 (shrinkage 48).
    # 1072 - 48 = 1024 (Output len).
    assert internal_shape[-1] - 48 == 1024

def test_pomeranian1k_heads_feature_sharing():
    # Verify that both heads operate on the same feature map (x),
    # and that the count head pooling aligns with the profile head valid region.
    
    model = Pomeranian1k()
    input_len = model.input_len
    output_len = model.output_len
    x = torch.randn(1, 4, input_len)
    
    inputs = {}
    def get_input(name):
        def hook(model, input, output):
            inputs[name] = input[0].detach()
        return hook
    
    # Hook profile head input (Pointwise Conv)
    model.profile_pointwise.register_forward_hook(get_input('profile_in'))
    
    # Hook count head input (MLP first layer)
    # model.count_mlp is Sequential, access first layer [0]
    model.count_mlp[0].register_forward_hook(get_input('count_in'))
    
    _ = model(x)
    
    feat_profile = inputs['profile_in'] # (B, C, L_internal)
    feat_count_pooled = inputs['count_in'] # (B, C)
    
    print(f"Profile feature shape: {feat_profile.shape}")
    print(f"Count pooled input shape: {feat_count_pooled.shape}")
    
    # Verify alignment
    # We expect feat_count_pooled to be the mean of the center-cropped feat_profile.
    # Crop amount: L_internal - output_len
    current_len = feat_profile.shape[-1]
    diff = current_len - output_len
    crop_l = diff // 2
    crop_r = diff - crop_l
    
    feat_profile_cropped = feat_profile[..., crop_l:-crop_r]
    expected_pooled = feat_profile_cropped.mean(dim=-1)
    
    # Assert close equality (float precision)
    assert torch.allclose(feat_count_pooled, expected_pooled, atol=1e-6)

@pytest.mark.skipif(int(torch.__version__.split(".")[0]) < 2, reason="torch.compile requires PyTorch 2.0+")
def test_pomeranian1k_compile():
    # Verify that the model can be compiled and run
    model = Pomeranian1k()
    model.eval()
    input_len = model.input_len
    output_len = model.output_len
    
    # Compile
    try:
        compiled_model = torch.compile(model)
    except Exception as e:
        pytest.skip(f"torch.compile failed (possibly unsupported backend): {e}")

    x = torch.randn(2, 4, input_len)
    
    # Warmup run
    try:
        _ = compiled_model(x)
    except Exception as e:
        pytest.fail(f"Compiled model forward pass failed: {e}")
        
    # Actual run to check output
    out = compiled_model(x)
    assert out.logits.shape == (2, 1, output_len)
    assert out.log_counts.shape == (2, 1)
