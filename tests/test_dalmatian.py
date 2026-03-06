"""Tests for the Dalmatian architecture (end-to-end bias-factorized model)."""

import torch
from cerberus.output import FactorizedProfileCountOutput, ProfileCountOutput, unbatch_modeloutput, compute_total_log_counts
from cerberus.loss import DalmatianLoss
from cerberus.models.dalmatian import Dalmatian


# --- Step 1: FactorizedProfileCountOutput tests ---

def test_dalmatian_output_is_profile_count_output():
    """FactorizedProfileCountOutput must be isinstance of ProfileCountOutput."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.randn(2, 1),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.randn(2, 1),
    )
    assert isinstance(out, ProfileCountOutput)


def test_dalmatian_output_detach():
    """detach() returns new FactorizedProfileCountOutput instance with all tensors detached."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100, requires_grad=True),
        log_counts=torch.randn(2, 1, requires_grad=True),
        bias_logits=torch.randn(2, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(2, 1, requires_grad=True),
        signal_logits=torch.randn(2, 1, 100, requires_grad=True),
        signal_log_counts=torch.randn(2, 1, requires_grad=True),
    )
    det = out.detach()
    assert isinstance(det, FactorizedProfileCountOutput)
    assert not det.logits.requires_grad
    assert not det.log_counts.requires_grad
    assert not det.bias_logits.requires_grad
    assert not det.bias_log_counts.requires_grad
    assert not det.signal_logits.requires_grad
    assert not det.signal_log_counts.requires_grad


def test_dalmatian_output_unbatch():
    """unbatch_modeloutput works with FactorizedProfileCountOutput (all tensor fields split)."""
    out = FactorizedProfileCountOutput(
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
    """compute_total_log_counts sees combined log_counts from FactorizedProfileCountOutput."""
    out = FactorizedProfileCountOutput(
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
    """Forward produces FactorizedProfileCountOutput with correct shapes."""
    model = Dalmatian(input_len=2112, output_len=1024)
    x = torch.randn(2, 4, 2112)
    out = model(x)
    assert isinstance(out, FactorizedProfileCountOutput)
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


# --- Step 3: DalmatianLoss tests ---


def _make_factorized_output(batch_size=4, channels=1, length=100):
    """Helper to create a FactorizedProfileCountOutput for loss tests."""
    return FactorizedProfileCountOutput(
        logits=torch.randn(batch_size, channels, length, requires_grad=True),
        log_counts=torch.randn(batch_size, channels, requires_grad=True),
        bias_logits=torch.randn(batch_size, channels, length, requires_grad=True),
        bias_log_counts=torch.randn(batch_size, channels, requires_grad=True),
        signal_logits=torch.randn(batch_size, channels, length, requires_grad=True),
        signal_log_counts=torch.randn(batch_size, channels, requires_grad=True),
    )


def test_dalmatian_loss_instantiation():
    """DalmatianLoss instantiates with nested base loss from class string."""
    loss = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_weight": 1.0, "profile_weight": 1.0},
    )
    from cerberus.loss import MSEMultinomialLoss
    assert isinstance(loss.base_loss, MSEMultinomialLoss)


def test_dalmatian_loss_forward_mixed_batch():
    """Loss computes all three terms with mixed peak/background batch."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
        signal_background_weight=0.1,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    peak_status = torch.tensor([1, 0, 1, 0])

    loss = loss_fn(output, target, peak_status)
    assert loss.shape == ()
    assert loss.requires_grad


def test_dalmatian_loss_all_peaks():
    """When all examples are peaks, bias and signal_bg terms are zero."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
        signal_background_weight=0.1,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    peak_status = torch.ones(4, dtype=torch.long)

    loss_all_peak = loss_fn(output, target, peak_status)

    # Compare with just reconstruction loss
    recon_only = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=0.0,
        signal_background_weight=0.0,
    )
    loss_recon = recon_only(output, target, peak_status)
    assert torch.allclose(loss_all_peak, loss_recon)


def test_dalmatian_loss_all_background():
    """When all examples are background, all three terms contribute."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
        signal_background_weight=0.1,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    peak_status = torch.zeros(4, dtype=torch.long)

    loss = loss_fn(output, target, peak_status)
    assert loss.shape == ()
    assert loss.requires_grad


def test_dalmatian_loss_backward():
    """Gradients flow through all decomposed fields."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
        signal_background_weight=0.1,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    peak_status = torch.tensor([1, 0, 1, 0])

    loss = loss_fn(output, target, peak_status)
    loss.backward()

    # Combined fields get gradients from l_recon
    assert output.logits.grad is not None
    assert output.log_counts.grad is not None
    # Bias fields get gradients from l_bias (non-peak examples exist)
    assert output.bias_logits.grad is not None
    assert output.bias_log_counts.grad is not None
    # Signal fields get gradients from l_signal_bg
    assert output.signal_logits.grad is not None
    assert output.signal_log_counts.grad is not None


def test_dalmatian_loss_pseudocount_forwarding():
    """count_pseudocount is forwarded to the base loss."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        count_pseudocount=5.0,
    )
    assert loss_fn.base_loss.count_pseudocount == 5.0  # type: ignore[union-attr]


def test_dalmatian_loss_pseudocount_base_args_override():
    """Explicit base_loss_args.count_pseudocount overrides top-level."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_pseudocount": 3.0},
        count_pseudocount=5.0,
    )
    # base_loss_args takes precedence (setdefault doesn't overwrite)
    assert loss_fn.base_loss.count_pseudocount == 3.0  # type: ignore[union-attr]


def test_dalmatian_loss_with_poisson_base():
    """DalmatianLoss works with PoissonMultinomialLoss as base."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.PoissonMultinomialLoss",
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    peak_status = torch.tensor([1, 0, 1, 0])

    loss = loss_fn(output, target, peak_status)
    assert loss.shape == ()
    assert loss.requires_grad
