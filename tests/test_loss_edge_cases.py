import torch
import pytest
from cerberus.loss import (
    MSEMultinomialLoss, PoissonMultinomialLoss, ProfilePoissonNLLLoss
)
from cerberus.metrics import (
    ProfilePearsonCorrCoef, CountProfilePearsonCorrCoef, 
    CountProfileMeanSquaredError, ProfileMeanSquaredError
)
from cerberus.output import ProfileLogits, ProfileCountOutput, ProfileLogRates

# --- Fixtures ---

@pytest.fixture
def shapes():
    return {
        "standard": {"batch": 4, "channels": 2, "length": 100},
        "single_batch": {"batch": 1, "channels": 2, "length": 100},
        "single_channel": {"batch": 4, "channels": 1, "length": 100},
        "single_length": {"batch": 4, "channels": 2, "length": 1},
        "minimal": {"batch": 1, "channels": 1, "length": 1},
    }

# --- Implicit Log Targets Equivalence Tests ---

def test_bpnet_loss_implicit_log_equivalence():
    """Verify MSEMultinomialLoss(implicit_log=True) with log-targets equals MSEMultinomialLoss(implicit_log=False) with raw-targets."""
    torch.manual_seed(42)
    B, C, L = 2, 2, 50
    
    # Raw targets
    targets_raw = torch.randint(0, 10, (B, C, L)).float()
    targets_log = torch.log1p(targets_raw)
    
    # Predictions
    logits = torch.randn(B, C, L, requires_grad=True)
    log_counts = torch.randn(B, 1, requires_grad=True)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # 1. Standard Loss
    loss_std = MSEMultinomialLoss(implicit_log_targets=False)(preds, targets_raw)
    
    # 2. Implicit Log Loss
    loss_impl = MSEMultinomialLoss(implicit_log_targets=True)(preds, targets_log)
    
    assert torch.isclose(loss_std, loss_impl, atol=1e-5)

def test_poisson_loss_implicit_log_equivalence():
    """Verify PoissonMultinomialLoss implicit log equivalence."""
    torch.manual_seed(42)
    B, C, L = 2, 1, 50
    
    targets_raw = torch.randint(0, 10, (B, C, L)).float()
    targets_log = torch.log1p(targets_raw)
    
    logits = torch.randn(B, C, L, requires_grad=True)
    log_counts = torch.randn(B, 1, requires_grad=True)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    loss_std = PoissonMultinomialLoss(implicit_log_targets=False)(preds, targets_raw)
    loss_impl = PoissonMultinomialLoss(implicit_log_targets=True)(preds, targets_log)
    
    assert torch.isclose(loss_std, loss_impl, atol=1e-5)

def test_metrics_implicit_log_equivalence():
    """Verify all metrics handle implicit log targets correctly."""
    torch.manual_seed(42)
    B, C, L = 2, 2, 50
    
    targets_raw = torch.abs(torch.randn(B, C, L)) * 10 # ensure positive
    targets_log = torch.log1p(targets_raw)
    
    logits = torch.randn(B, C, L)
    log_counts = torch.randn(B, 1) # (B, 1) for decoupled
    
    preds_prof = ProfileLogits(logits=logits)
    preds_count = ProfileCountOutput(logits=logits, log_counts=log_counts)
    preds_rates = ProfileLogRates(log_rates=logits) # For Poisson loss which strictly requires rates
    
    metrics_to_test = [
        (ProfilePearsonCorrCoef(num_channels=C), preds_prof),
        (CountProfilePearsonCorrCoef(num_channels=C), preds_count),
        (CountProfileMeanSquaredError(), preds_count),
        (ProfileMeanSquaredError(), preds_prof),
        (ProfilePoissonNLLLoss(), preds_rates) # It's a loss but also used as metric sometimes
    ]
    
    for metric_cls, preds in metrics_to_test:
        metric_name = metric_cls.__class__.__name__
        
        # 1. Standard Update
        m_std = metric_cls.__class__(implicit_log_targets=False, **_get_init_kwargs(metric_cls))
        if isinstance(m_std, ProfilePoissonNLLLoss):
            val_std = m_std(preds, targets_raw)
        else:
            m_std.update(preds, targets_raw)
            val_std = m_std.compute()
            
        # 2. Implicit Log Update
        m_impl = metric_cls.__class__(implicit_log_targets=True, **_get_init_kwargs(metric_cls))
        if isinstance(m_impl, ProfilePoissonNLLLoss):
            val_impl = m_impl(preds, targets_log)
        else:
            m_impl.update(preds, targets_log)
            val_impl = m_impl.compute()
            
        assert torch.isclose(val_std, val_impl, atol=1e-5), f"{metric_name} failed implicit log check"

def _get_init_kwargs(metric_instance):
    """Helper to extract init args that need to be passed back."""
    kwargs = {}
    if hasattr(metric_instance, "num_channels"):
        kwargs["num_channels"] = metric_instance.num_channels
    if isinstance(metric_instance, (ProfilePoissonNLLLoss, torch.nn.PoissonNLLLoss)):
        kwargs["log_input"] = True
        kwargs["full"] = False
    return kwargs


# --- Dimension Edge Cases ---

def test_dimension_minimal(shapes):
    """Test Batch=1, Length=1 cases."""
    shape = shapes["minimal"]
    B, C, L = shape["batch"], shape["channels"], shape["length"]
    
    logits = torch.randn(B, C, L)
    targets = torch.randn(B, C, L).abs()
    
    # Profile MSE
    mse = ProfileMeanSquaredError()
    mse.update(ProfileLogits(logits=logits), targets)
    val = mse.compute()
    assert not torch.isnan(val)
    
    # Profile Pearson
    # With Length=1, correlation is undefined (variance=0) or NaN depending on implementation
    # PearsonCorrCoef usually returns NaN if input has no variance?
    # Or if we flatten over B*L = 1*1 = 1 sample, correlation is definitely undefined.
    # We need at least 2 samples.
    
    pearson = ProfilePearsonCorrCoef(num_channels=C)
    # This should probably produce NaN or raise error or handle gracefully
    # torchmetrics Pearson often returns NaN for insufficient data
    
    with pytest.warns(UserWarning, match="variance.*close to zero"):
        pearson.update(ProfileLogits(logits=logits), targets)
        val_p = pearson.compute()
    
    # Pearson on 1 sample is undefined/NaN
    assert torch.isnan(val_p)

def test_dimension_single_length(shapes):
    """Test Length=1 but Batch>1."""
    shape = shapes["single_length"] # B=4, C=2, L=1
    B, C, L = shape["batch"], shape["channels"], shape["length"]
    
    logits = torch.randn(B, C, L)
    targets = torch.randn(B, C, L).abs()
    
    pearson = ProfilePearsonCorrCoef(num_channels=C)
    # Flattening: (B, C, L) -> (B*L, C) -> (4, 2).
    # However, Softmax is applied along dim=-1 (Length).
    # With Length=1, Softmax([x]) = [1.0].
    # So all predictions are exactly 1.0. Variance is 0.
    # Pearson Correlation should be NaN (or raise warning).
    
    with pytest.warns(UserWarning, match="variance.*close to zero"):
        pearson.update(ProfileLogits(logits=logits), targets)
        val = pearson.compute()
    assert torch.isnan(val), "Length=1 should result in NaN correlation due to Softmax normalization (constant 1.0 predictions)"

# --- Numerical Stability ---

def test_profile_mse_zero_targets():
    """Test ProfileMeanSquaredError when targets sum to 0 (all zeros)."""
    B, C, L = 2, 1, 10
    logits = torch.randn(B, C, L)
    targets = torch.zeros(B, C, L) # All zeros
    
    # Should handle normalization division by zero safely
    mse = ProfileMeanSquaredError()
    mse.update(ProfileLogits(logits=logits), targets)
    val = mse.compute()
    assert not torch.isnan(val)
    # Target probs should be uniform or zeros?
    # Code uses: target / (sum + 1e-8). 0/1e-8 = 0.
    # So target "probs" are all 0. Sum is 0. Not a valid probability distribution.
    # But MSE should just compute difference from predicted probs (which sum to 1).
    # So error will be non-zero but finite.

def test_decoupled_mse_broadcasting():
    """Test CountProfileMeanSquaredError broadcasting behavior."""
    # Logits: (B, C, L)
    # LogCounts: (B, 1) or (B, 1, 1)
    # Should broadcast correctly.
    B, C, L = 2, 2, 10
    logits = torch.randn(B, C, L)
    
    # Case 1: (B, 1)
    log_counts_1 = torch.randn(B, 1)
    preds_1 = ProfileCountOutput(logits=logits, log_counts=log_counts_1)
    
    # Case 2: (B,) - older BPNet style might return this?
    # Our code checks dim() == 1 -> unsqueeze.
    log_counts_2 = torch.randn(B)
    preds_2 = ProfileCountOutput(logits=logits, log_counts=log_counts_2)
    
    targets = torch.randn(B, C, L).abs()
    
    mse = CountProfileMeanSquaredError()
    
    # Run both, should not crash
    mse.update(preds_1, targets)
    mse.reset()
    mse.update(preds_2, targets)

# --- Type Checking ---

def test_invalid_input_types():
    """Test that metrics raise TypeError for wrong input types."""
    metric = ProfileMeanSquaredError()
    with pytest.raises(TypeError):
        metric.update(torch.randn(10, 10), torch.randn(10, 10)) # type: ignore

    metric_dec = CountProfileMeanSquaredError()
    with pytest.raises(TypeError):
        metric_dec.update(ProfileLogits(logits=torch.randn(10)), torch.randn(10)) # type: ignore
