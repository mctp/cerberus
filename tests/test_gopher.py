
import torch
from cerberus.models import GlobalProfileCNN
import pytest

def test_cnn_default():
    batch_size = 2
    input_len = 2048
    output_len = 2048
    output_bin_size = 4
    output_channels = ["signal"]
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size, output_channels=output_channels)
    
    # Input: (batch, 4, seq_len)
    x = torch.randn(batch_size, 4, input_len)
    
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, len(output_channels), expected_output_bins)
    
    assert output.log_rates.shape == expected_shape

def test_cnn_flexible_input():
    batch_size = 2
    input_len = 1536  # Must be divisible by 128 (128*12 = 1536)
    output_len = 512
    output_bin_size = 4
    output_channels = ["signal"]
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size, output_channels=output_channels)
    
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, len(output_channels), expected_output_bins)
    
    assert output.log_rates.shape == expected_shape

def test_cnn_large_input():
    batch_size = 2 # Need >1 for BatchNorm
    input_len = 4096 # Divisible by 128
    output_len = 1024
    output_bin_size = 4
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size)
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    assert output.log_rates.shape == (batch_size, 1, expected_output_bins)

def test_cnn_flexible_channels():
    batch_size = 2
    input_len = 2048
    output_len = 1024
    output_bin_size = 4
    output_channels = ["signal"]
    input_channels = ["c"] * 6  # Testing with 6 channels instead of 4
    
    model = GlobalProfileCNN(
        input_len=input_len, 
        output_len=output_len, 
        output_bin_size=output_bin_size, 
        output_channels=output_channels,
        input_channels=input_channels
    )
    
    # Input: (batch, len(input_channels), seq_len)
    x = torch.randn(batch_size, len(input_channels), input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, len(output_channels), expected_output_bins)
    
    assert output.log_rates.shape == expected_shape

def test_gopher_binning_options():
    """Test output shapes for different bin sizes."""
    input_len = 2048
    output_len = 1024
    batch_size = 2
    
    # Bin size 1
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=1)
    output = model(torch.randn(batch_size, 4, input_len))
    assert output.log_rates.shape == (batch_size, 1, 1024)
    
    # Bin size 32
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=32)
    output = model(torch.randn(batch_size, 4, input_len))
    assert output.log_rates.shape == (batch_size, 1, 32)

def test_gopher_bottleneck_config():
    """Test bottleneck channel configuration."""
    model = GlobalProfileCNN(input_len=2048, output_len=1024, output_bin_size=4, bottleneck_channels=16)
    # Check internal dense layer size
    # nr_bins (256) * bottleneck (16) = 4096
    assert model.projection_size == 4096
    
    x = torch.randn(2, 4, 2048)
    output = model(x)
    assert output.log_rates.shape == (2, 1, 256)

def test_gopher_fixed_input_constraint():
    """Test that model fails if input length changes at runtime."""
    model = GlobalProfileCNN(input_len=2048)
    
    # Should work
    model(torch.randn(2, 4, 2048))
    
    # Should fail due to input being shorter than required input_len
    with pytest.raises(ValueError, match="shorter than required"):
        model(torch.randn(2, 4, 1024))

def test_reshape_layer():
    """Test the internal _Reshape module."""
    from cerberus.models.gopher import _Reshape
    
    # Test 1: Simple reshape
    batch_size = 2
    channels = 8
    length = 10
    
    # Input flattened: (Batch, Channels * Length)
    x = torch.randn(batch_size, channels * length)
    
    # Reshape to (Channels, Length)
    reshape_layer = _Reshape(channels, length)
    output = reshape_layer(x)
    
    assert output.shape == (batch_size, channels, length)
    
    # Verify data is preserved (view mechanics)
    # Check first element of first batch
    assert output[0, 0, 0] == x[0, 0]
    
    # Test 2: Multi-dimensional reshape
    x = torch.randn(batch_size, 24)
    reshape_layer = _Reshape(2, 3, 4) # 2*3*4 = 24
    output = reshape_layer(x)
    assert output.shape == (batch_size, 2, 3, 4)

def test_compile_compatibility():
    """Test that the model can be compiled with torch.compile."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")
        
    batch_size = 2
    input_len = 2048
    output_len = 1024
    output_bin_size = 4
    
    model = GlobalProfileCNN(
        input_len=input_len, 
        output_len=output_len, 
        output_bin_size=output_bin_size
    )
    
    # Compile the model
    # Use backend='inductor' if available, otherwise default
    try:
        compiled_model = torch.compile(model)
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")

    # Run forward pass
    x = torch.randn(batch_size, 4, input_len)
    try:
        output = compiled_model(x)
    except Exception as e:
        pytest.fail(f"Forward pass with compiled model failed: {e}")
        
    expected_shape = (batch_size, 1, output_len // output_bin_size)
    assert output.log_rates.shape == expected_shape

def test_compile_graph_breaks():
    """Test that there are no graph breaks when compiling the model."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available (requires PyTorch 2.0+)")
        
    # Check if _dynamo is available (it should be if compile is)
    if not hasattr(torch, "_dynamo"):
        pytest.skip("torch._dynamo not available")
    
    batch_size = 2
    input_len = 2048
    output_len = 1024
    output_bin_size = 4
    
    model = GlobalProfileCNN(
        input_len=input_len, 
        output_len=output_len, 
        output_bin_size=output_bin_size
    )
    
    x = torch.randn(batch_size, 4, input_len)
    
    # Use torch._dynamo.explain to analyze graph breaks
    try:
        explanation = torch._dynamo.explain(model)(x)
    except Exception as e:
        pytest.fail(f"torch._dynamo.explain failed: {e}")
    
    # Check graph break count
    if explanation.graph_break_count > 0:
        # Print breaks for debugging if failure occurs
        print(f"Graph breaks found: {explanation.graph_break_count}")
        # Depending on pytorch version, explanation might have 'graph_breaks' list or we inspect other attributes
        # For now, just failing is enough, but printing if possible is good.
        if hasattr(explanation, 'graph_breaks'):
             for break_info in explanation.graph_breaks:
                 print(break_info)
        pytest.fail(f"Found {explanation.graph_break_count} graph breaks in GlobalProfileCNN")

if __name__ == "__main__":
    test_cnn_default()
    test_cnn_flexible_input()
    test_cnn_large_input()
    test_cnn_flexible_channels()
    test_gopher_binning_options()
    test_gopher_bottleneck_config()
    test_gopher_fixed_input_constraint()
    test_reshape_layer()
    test_compile_compatibility()
    test_compile_graph_breaks()
    print("All tests passed!")
