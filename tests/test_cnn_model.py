
import torch
from cerberus.models import VanillaCNN
import pytest

def test_vanilla_cnn_default():
    batch_size = 2
    input_len = 2048
    output_len = 2048
    bin_size = 4
    num_output_channels = 1
    
    model = VanillaCNN(input_len=input_len, output_len=output_len, bin_size=bin_size, num_output_channels=num_output_channels)
    
    # Input: (batch, 4, seq_len)
    x = torch.randn(batch_size, 4, input_len)
    
    output = model(x)
    
    expected_output_bins = output_len // bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

def test_vanilla_cnn_flexible_input():
    batch_size = 2
    input_len = 1500  # Different from default 2048, must be > ~1254
    output_len = 512
    bin_size = 4
    num_output_channels = 1
    
    model = VanillaCNN(input_len=input_len, output_len=output_len, bin_size=bin_size, num_output_channels=num_output_channels)
    
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

def test_vanilla_cnn_large_input():
    batch_size = 1
    input_len = 4096
    output_len = 1024
    bin_size = 4
    
    model = VanillaCNN(input_len=input_len, output_len=output_len, bin_size=bin_size)
    x = torch.randn(batch_size, 4, input_len)
    output = model(x)
    
    expected_output_bins = output_len // bin_size
    assert output.shape == (batch_size, 1, expected_output_bins)

def test_vanilla_cnn_flexible_channels():
    batch_size = 2
    input_len = 2048
    output_len = 1024
    bin_size = 4
    num_output_channels = 1
    num_input_channels = 6  # Testing with 6 channels instead of 4
    
    model = VanillaCNN(
        input_len=input_len, 
        output_len=output_len, 
        bin_size=bin_size, 
        num_output_channels=num_output_channels,
        num_input_channels=num_input_channels
    )
    
    # Input: (batch, num_input_channels, seq_len)
    x = torch.randn(batch_size, num_input_channels, input_len)
    output = model(x)
    
    expected_output_bins = output_len // bin_size
    expected_shape = (batch_size, num_output_channels, expected_output_bins)
    
    assert output.shape == expected_shape

if __name__ == "__main__":
    test_vanilla_cnn_default()
    test_vanilla_cnn_flexible_input()
    test_vanilla_cnn_large_input()
    test_vanilla_cnn_flexible_channels()
    print("All tests passed!")
