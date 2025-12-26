
import torch
from cerberus.models import GlobalProfileCNN
import pytest

def test_cnn_default():
    batch_size = 2
    input_len = 2048
    output_len = 2048
    output_bin_size = 4
    num_output_channels = 1
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size, num_output_channels=num_output_channels)
    
    # Input: (batch, 4, seq_len)
    x = torch.randn(batch_size, 4, input_len)
    
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

def test_cnn_flexible_input():
    batch_size = 2
    input_len = 1536  # Must be divisible by 128 (128*12 = 1536)
    output_len = 512
    output_bin_size = 4
    num_output_channels = 1
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size, num_output_channels=num_output_channels)
    
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

def test_cnn_large_input():
    batch_size = 2 # Need >1 for BatchNorm
    input_len = 4096 # Divisible by 128
    output_len = 1024
    output_bin_size = 4
    
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=output_bin_size)
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    assert output.shape == (batch_size, 1, expected_output_bins)

def test_cnn_flexible_channels():
    batch_size = 2
    input_len = 2048
    output_len = 1024
    output_bin_size = 4
    num_output_channels = 1
    num_input_channels = 6  # Testing with 6 channels instead of 4
    
    model = GlobalProfileCNN(
        input_len=input_len, 
        output_len=output_len, 
        output_bin_size=output_bin_size, 
        num_output_channels=num_output_channels,
        num_input_channels=num_input_channels
    )
    
    # Input: (batch, num_input_channels, seq_len)
    x = torch.randn(batch_size, num_input_channels, input_len)
    output = model(x)
    
    expected_output_bins = output_len // output_bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

def test_gopher_binning_options():
    """Test output shapes for different bin sizes."""
    input_len = 2048
    output_len = 1024
    batch_size = 2
    
    # Bin size 1
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=1)
    output = model(torch.randn(batch_size, 4, input_len))
    assert output.shape == (batch_size, 1, 1024)
    
    # Bin size 32
    model = GlobalProfileCNN(input_len=input_len, output_len=output_len, output_bin_size=32)
    output = model(torch.randn(batch_size, 4, input_len))
    assert output.shape == (batch_size, 1, 32)

def test_gopher_bottleneck_config():
    """Test bottleneck channel configuration."""
    model = GlobalProfileCNN(input_len=2048, output_len=1024, output_bin_size=4, bottleneck_channels=16)
    # Check internal dense layer size
    # nr_bins (256) * bottleneck (16) = 4096
    assert model.projection_size == 4096
    
    x = torch.randn(2, 4, 2048)
    output = model(x)
    assert output.shape == (2, 1, 256)

def test_gopher_fixed_input_constraint():
    """Test that model fails if input length changes at runtime."""
    model = GlobalProfileCNN(input_len=2048)
    
    # Should work
    model(torch.randn(2, 4, 2048))
    
    # Should fail due to Dense layer mismatch
    # Note: Flatten size will change, Linear layer will complain about input shape
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 4, 1024))

if __name__ == "__main__":
    test_cnn_default()
    test_cnn_flexible_input()
    test_cnn_large_input()
    test_cnn_flexible_channels()
    test_gopher_binning_options()
    test_gopher_bottleneck_config()
    test_gopher_fixed_input_constraint()
    print("All tests passed!")
