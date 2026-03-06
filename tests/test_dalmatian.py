"""Tests for the Dalmatian architecture (end-to-end bias-factorized model)."""

import torch
from cerberus.output import DalmatianOutput, ProfileCountOutput, unbatch_modeloutput, compute_total_log_counts
from cerberus.models.dalmatian import Dalmatian


# --- Step 1: DalmatianOutput tests ---

def test_dalmatian_output_is_profile_count_output():
    """DalmatianOutput must be isinstance of ProfileCountOutput."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.randn(2, 1),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.randn(2, 1),
    )
    assert isinstance(out, ProfileCountOutput)


def test_dalmatian_output_detach():
    """detach() returns new DalmatianOutput instance with all tensors detached."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100, requires_grad=True),
        log_counts=torch.randn(2, 1, requires_grad=True),
        bias_logits=torch.randn(2, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(2, 1, requires_grad=True),
        signal_logits=torch.randn(2, 1, 100, requires_grad=True),
        signal_log_counts=torch.randn(2, 1, requires_grad=True),
    )
    det = out.detach()
    assert isinstance(det, DalmatianOutput)
    assert not det.logits.requires_grad
    assert not det.log_counts.requires_grad
    assert not det.bias_logits.requires_grad
    assert not det.bias_log_counts.requires_grad
    assert not det.signal_logits.requires_grad
    assert not det.signal_log_counts.requires_grad


def test_dalmatian_output_unbatch():
    """unbatch_modeloutput works with DalmatianOutput (all tensor fields split)."""
    out = DalmatianOutput(
        logits=torch.randn(4, 1, 100),
        log_counts=torch.randn(4, 1),
        bias_logits=torch.randn(4, 1, 100),
        bias_log_counts=torch.randn(4, 1),
        signal_logits=torch.randn(4, 1, 100),
        signal_log_counts=torch.randn(4, 1),
    )
    items = unbatch_modeloutput(out, 4)
    assert len(items) == 4
    # All tensor fields should be present and correctly shaped
    for item in items:
        assert "bias_logits" in item
        assert "signal_logits" in item
        assert "bias_log_counts" in item
        assert "signal_log_counts" in item
        assert item["bias_logits"].shape == (1, 100)
        assert item["signal_log_counts"].shape == (1,)


def test_dalmatian_output_compute_total_log_counts():
    """compute_total_log_counts sees combined log_counts from DalmatianOutput."""
    out = DalmatianOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.tensor([[3.0], [4.0]]),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.tensor([[2.0], [3.0]]),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.tensor([[1.0], [2.0]]),
    )
    lc = compute_total_log_counts(out)
    # Should use combined log_counts, not decomposed
    assert torch.allclose(lc, torch.tensor([3.0, 4.0]))


# --- Step 2: Dalmatian model tests ---

def test_dalmatian_forward_shape():
    """Forward produces DalmatianOutput with correct shapes."""
    model = Dalmatian(input_len=2112, output_len=1024)
    x = torch.randn(2, 4, 2112)
    out = model(x)
    assert isinstance(out, DalmatianOutput)
    assert out.logits.shape == (2, 1, 1024)
    assert out.log_counts.shape == (2, 1)
    assert out.bias_logits.shape == (2, 1, 1024)
    assert out.bias_log_counts.shape == (2, 1)
    assert out.signal_logits.shape == (2, 1, 1024)
    assert out.signal_log_counts.shape == (2, 1)


def test_dalmatian_zero_init():
    """Signal model outputs are zero/negligible at initialization."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)
    # Signal logits should be exactly 0
    assert torch.allclose(out.signal_logits, torch.zeros_like(out.signal_logits), atol=1e-6)
    # Signal log_counts should be ~ -10
    assert (out.signal_log_counts < -9.0).all()


def test_dalmatian_combined_equals_bias_at_init():
    """At initialization, combined output ~ bias-only output."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)
    # Profile: combined = bias + 0 = bias
    assert torch.allclose(out.logits, out.bias_logits, atol=1e-6)
    # Counts: logsumexp(bias, -10) ~ bias (when bias >> -10)
    diff = (out.log_counts - out.bias_log_counts).abs()
    assert diff.max() < 0.01


def test_dalmatian_backward():
    """Gradients flow through both sub-models."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"bias_model.{name} has no gradient"
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"signal_model.{name} has no gradient"


def test_dalmatian_param_count():
    """Parameter count is in expected range."""
    model = Dalmatian()
    total = sum(p.numel() for p in model.parameters())
    bias_params = sum(p.numel() for p in model.bias_model.parameters())
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 20_000 < bias_params < 150_000, f"Bias params {bias_params} out of range"
    assert 1_000_000 < signal_params < 5_000_000, f"Signal params {signal_params} out of range"
    assert total == bias_params + signal_params


def test_dalmatian_bias_input_crop():
    """BiasNet receives center-cropped input via Pomeranian's auto-crop."""
    model = Dalmatian()
    # bias_input_len should be less than full input_len
    assert model.bias_input_len < model.input_len
    # bias_input_len = output_len + bias_shrinkage = 1024 + 146 = 1170
    assert model.bias_input_len == 1170
    # Signal model uses full input
    assert model.signal_model.input_len == 2112


def test_dalmatian_shrinkage_validation():
    """Dalmatian rejects signal configs that don't produce exact output_len."""
    import pytest
    with pytest.raises(ValueError, match="SignalNet shrinkage"):
        Dalmatian(
            input_len=2112, output_len=1024,
            signal_dilations=[1, 1, 2, 4],  # too little shrinkage
        )
