import torch
import pytest
from cerberus.models.lyra import LyraNet

def test_lyranet_init():
    model = LyraNet(
        input_len=1000,
        output_len=500,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"],
        filters=32,
        pgc_layers=2,
        s4_layers=2
    )
    assert isinstance(model, LyraNet)
    assert model.input_len == 1000
    assert model.output_len == 500

def test_lyranet_forward_shapes():
    B, L, C_in = 2, 1024, 4
    L_out = 500
    C_out = 1
    
    model = LyraNet(
        input_len=L,
        output_len=L_out,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"],
        filters=32,
        pgc_layers=2,
        s4_layers=2,
        predict_total_count=True
    )
    
    x = torch.randn(B, C_in, L)
    output = model(x)
    
    # Check Profile Logits
    assert output.logits.shape == (B, C_out, L_out)
    
    # Check Log Counts
    assert output.log_counts.shape == (B, 1)

def test_lyranet_forward_multichannel():
    B, L, C_in = 2, 1024, 4
    L_out = 500
    output_channels = ["sig1", "sig2"]
    
    model = LyraNet(
        input_len=L,
        output_len=L_out,
        input_channels=["A", "C", "G", "T"],
        output_channels=output_channels,
        filters=32,
        pgc_layers=2,
        s4_layers=2,
        predict_total_count=False # Predict per channel
    )
    
    x = torch.randn(B, C_in, L)
    output = model(x)
    
    # Check Profile Logits
    assert output.logits.shape == (B, 2, L_out)
    
    # Check Log Counts
    assert output.log_counts.shape == (B, 2)

def test_lyranet_binning():
    B, L, C_in = 2, 1024, 4
    L_out = 500
    bin_size = 5
    
    model = LyraNet(
        input_len=L,
        output_len=L_out,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"],
        filters=32,
        pgc_layers=2,
        s4_layers=2,
        output_bin_size=bin_size
    )
    
    x = torch.randn(B, C_in, L)
    output = model(x)
    
    # Check Profile Logits
    # Output length should be L_out / bin_size
    expected_len = L_out // bin_size
    assert output.logits.shape == (B, 1, expected_len)

def test_lyranet_output_len_larger_than_input():
    # This should fail or error because LyraNet crops from valid/same convs
    # In LyraNet (all 'same'), internal length preserves L.
    # If output_len > input_len, cropping fails.
    
    model = LyraNet(
        input_len=100,
        output_len=200, # Larger than input
        input_channels=["A"],
        output_channels=["s"],
        filters=16
    )
    
    x = torch.randn(1, 1, 100)
    with pytest.raises(ValueError, match="smaller than requested"):
        model(x)

def test_lyranet_custom_kernel_size():
    # Verify that changing kernel sizes works
    conv_k = 1
    profile_k = 1
    
    model = LyraNet(
        input_len=100,
        output_len=50,
        input_channels=["A"],
        output_channels=["s"],
        filters=16,
        conv_kernel_size=conv_k,
        profile_kernel_size=profile_k
    )
    
    # Check if the kernel sizes were applied
    # Stem is Sequential -> Conv1d
    assert model.stem[0].kernel_size[0] == conv_k # type: ignore
    assert model.profile_conv.kernel_size[0] == profile_k # type: ignore
    
    x = torch.randn(2, 1, 100)
    out = model(x)
    assert out.logits.shape == (2, 1, 50)

def test_lyranet_s4_lr():
    lr = 0.005
    model = LyraNet(
        input_len=100,
        output_len=50,
        s4_lr=lr
    )
    # Check if the LR was registered in the _optim dict of the kernel parameters
    # The kernel is inside S4D -> kernel
    kernel = model.s4_layers[0].kernel # type: ignore
    assert kernel.log_dt._optim["lr"] == lr # type: ignore
