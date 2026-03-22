"""Tests for the BiasNet architecture (lightweight Tn5 bias model)."""

import pytest
import torch

from cerberus.models.biasnet import BiasNet
from cerberus.output import ProfileCountOutput


def test_biasnet_forward_shape():
    """Forward produces ProfileCountOutput with correct shapes."""
    model = BiasNet(input_len=1128, output_len=1024)
    x = torch.randn(2, 4, 1128)
    out = model(x)
    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (2, 1, 1024)
    assert out.log_counts.shape == (2, 1)


def test_biasnet_center_crop():
    """BiasNet center-crops oversized input."""
    model = BiasNet(input_len=1128, output_len=1024)
    x = torch.randn(2, 4, 2112)  # oversized
    out = model(x)
    assert out.logits.shape == (2, 1, 1024)


def test_biasnet_rejects_short_input():
    """BiasNet raises on input shorter than required."""
    model = BiasNet(input_len=1128, output_len=1024)
    x = torch.randn(2, 4, 500)
    with pytest.raises(ValueError, match="shorter than required"):
        model(x)


def test_biasnet_default_param_count():
    """Default BiasNet (f=12) has ~9.3K params."""
    model = BiasNet(input_len=1128, output_len=1024)
    params = sum(p.numel() for p in model.parameters())
    assert 8_000 < params < 12_000, f"Params {params} outside expected range"


def test_biasnet_backward():
    """Gradients flow through BiasNet."""
    model = BiasNet(input_len=1128, output_len=1024)
    x = torch.randn(2, 4, 1128)
    out = model(x)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"


def test_biasnet_linear_head():
    """Linear head mode uses a single spatial conv (no pointwise)."""
    model = BiasNet(input_len=1128, output_len=1024, linear_head=True)
    assert not hasattr(model, "profile_pointwise")
    assert hasattr(model, "profile_spatial")


def test_biasnet_nonlinear_head():
    """Non-linear head mode uses pointwise + ReLU + spatial."""
    model = BiasNet(input_len=1128, output_len=1024, linear_head=False)
    assert hasattr(model, "profile_pointwise")
    assert hasattr(model, "profile_act")
    assert hasattr(model, "profile_spatial")
    x = torch.randn(2, 4, 1128)
    out = model(x)
    assert out.logits.shape == (2, 1, 1024)


def test_biasnet_no_residual():
    """BiasNet works without residual connections."""
    model = BiasNet(input_len=1128, output_len=1024, residual=False)
    x = torch.randn(2, 4, 1128)
    out = model(x)
    assert out.logits.shape == (2, 1, 1024)


def test_biasnet_multi_channel():
    """BiasNet works with multiple output channels."""
    model = BiasNet(
        input_len=1128, output_len=1024,
        output_channels=["sample1", "sample2"],
        predict_total_count=False,
    )
    x = torch.randn(2, 4, 1128)
    out = model(x)
    assert out.logits.shape == (2, 2, 1024)
    assert out.log_counts.shape == (2, 2)


def test_biasnet_shrinkage_matches_geometry():
    """Verify that stem + tower + head shrinkage produces exact output_len."""
    # Default: stem [11,11]=20, tower 5*1*(9-1)=40, head 45-1=44 → total=104
    model = BiasNet(input_len=1128, output_len=1024)
    x = torch.randn(1, 4, 1128)
    out = model(x)
    assert out.logits.shape[-1] == 1024


def test_biasnet_custom_filters():
    """BiasNet works with different filter counts."""
    for f in [8, 16, 32]:
        # Shrinkage is independent of filters: 20+40+44=104
        model = BiasNet(input_len=1128, output_len=1024, filters=f)
        x = torch.randn(1, 4, 1128)
        out = model(x)
        assert out.logits.shape == (1, 1, 1024)


def test_biasnet_state_dict_roundtrip(tmp_path):
    """BiasNet state_dict saves and loads correctly."""
    model = BiasNet(input_len=1128, output_len=1024)
    path = tmp_path / "biasnet.pt"
    torch.save(model.state_dict(), path)

    model2 = BiasNet(input_len=1128, output_len=1024)
    model2.load_state_dict(torch.load(path, weights_only=True))

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch after load"


def test_biasnet_convenience_import():
    """BiasNet is importable from cerberus.models shortcut."""
    from cerberus.models import BiasNet as B
    assert B is BiasNet


def test_biasnet_via_import_class():
    """BiasNet can be instantiated via import_class (config pipeline)."""
    from cerberus.utils import import_class
    cls = import_class("cerberus.models.biasnet.BiasNet")
    model = cls(input_len=1128, output_len=1024)
    assert isinstance(model, BiasNet)
