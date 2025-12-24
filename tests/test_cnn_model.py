import torch
from cerberus.models import VanillaCNN

def test_vanilla_cnn():
    batch_size = 2
    window_size = 2048
    bin_size = 4
    nr_tracks = 1
    
    model = VanillaCNN(window_size=window_size, bin_size=bin_size, nr_tracks=nr_tracks)
    
    # Input: (batch, seq_len, 4)
    x = torch.randn(batch_size, window_size, 4)
    
    output = model(x)
    
    expected_output_len = window_size // bin_size
    expected_shape = (batch_size, expected_output_len, nr_tracks)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape
    print("Test passed!")

if __name__ == "__main__":
    test_vanilla_cnn()
