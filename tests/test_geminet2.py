
import torch
import torch.nn as nn
import pytest
from cerberus.models.geminet2 import GemiNet2, GemiNet2Medium
from cerberus.output import ProfileCountOutput
from cerberus.layers import ConvNeXtV2Block

def test_geminet2_instantiation():
    batch_size = 2
    input_len = 1000
    output_len = 500
    filters = 16
    n_dilated_layers = 2
    
    model = GemiNet2(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers
    )
    x = torch.randn(batch_size, 4, input_len)
    out = model(x)
    
    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (batch_size, 1, output_len)
    assert out.log_counts.shape == (batch_size, 1)

def test_geminet2_stem_is_convnext():
    model = GemiNet2(input_len=1000, output_len=500)
    assert isinstance(model.stem, ConvNeXtV2Block)

def test_geminet2_medium():
    model = GemiNet2Medium(input_len=1000, output_len=500)
    assert isinstance(model.stem, ConvNeXtV2Block)
    # Check filters
    # ConvNeXtV2Block stem: dwconv.in_channels=4, dwconv.out_channels=filters (128 for Medium)
    # Wait, ConvNeXtV2Block structure:
    # dwconv -> norm -> pwconv1 -> act -> grn -> pwconv2
    # input channels_in=4, channels_out=128.
    # dwconv has channels_out filters.
    assert model.stem.dwconv.out_channels == 128
