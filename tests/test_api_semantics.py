import torch
import pytest
import torch.nn.functional as F
from cerberus.output import ProfileLogits, ProfileCountOutput, ProfileLogRates, ProfileLogits
from cerberus.loss import (
    ProfilePoissonNLLLoss, 
    CoupledMSEMultinomialLoss, 
    CoupledPoissonMultinomialLoss,
    MSEMultinomialLoss
)
from cerberus.metrics import FlattenedPearsonCorrCoef, ProfileMeanSquaredError

# --- 1. Output Class Semantics ---

def test_output_class_structure():
    """Verify the structure and inheritance of output classes."""
    # ProfileLogRates
    rates = torch.randn(1, 1, 10)
    out_rates = ProfileLogRates(log_rates=rates)
    assert isinstance(out_rates, ProfileLogRates)
    assert not isinstance(out_rates, ProfileLogits) # Should be distinct
    assert hasattr(out_rates, 'log_rates')
    assert not hasattr(out_rates, 'logits')

    # ProfileLogits
    logits = torch.randn(1, 1, 10)
    out_logits = ProfileLogits(logits=logits)
    assert isinstance(out_logits, ProfileLogits)
    assert not isinstance(out_logits, ProfileLogRates)
    assert hasattr(out_logits, 'logits')
    
    # ProfileCountOutput (Inheritance)
    counts = torch.randn(1, 1)
    out_count = ProfileCountOutput(logits=logits, log_counts=counts)
    assert isinstance(out_count, ProfileCountOutput)
    assert isinstance(out_count, ProfileLogits) # Must inherit
    assert not isinstance(out_count, ProfileLogRates)
    assert hasattr(out_count, 'logits')
    assert hasattr(out_count, 'log_counts')

def test_profile_output_alias():
    """Verify ProfileLogits is an alias for ProfileLogits."""
    assert ProfileLogits is ProfileLogits
    out = ProfileLogits(logits=torch.randn(1,1,10))
    assert isinstance(out, ProfileLogits)

# --- 2. Loss Function Strictness ---

def test_profile_poisson_nll_strictness():
    """Verify ProfilePoissonNLLLoss enforces semantic strictness."""
    loss_fn = ProfilePoissonNLLLoss()
    targets = torch.randn(1, 1, 10).abs()
    
    # Valid input: ProfileLogRates
    rates = ProfileLogRates(log_rates=torch.randn(1, 1, 10))
    loss = loss_fn(rates, targets)
    assert not torch.isnan(loss)
    
    # Invalid input: ProfileLogits (Shape only)
    logits = ProfileLogits(logits=torch.randn(1, 1, 10))
    with pytest.raises(TypeError, match="requires ProfileLogRates"):
        loss_fn(logits, targets)
        
    # Invalid input: ProfileCountOutput
    counts = ProfileCountOutput(logits=torch.randn(1, 1, 10), log_counts=torch.randn(1, 1))
    with pytest.raises(TypeError, match="requires ProfileLogRates"):
        loss_fn(counts, targets)

def test_coupled_mse_multinomial_strictness():
    """Verify CoupledMSEMultinomialLoss enforces strictness."""
    loss_fn = CoupledMSEMultinomialLoss()
    targets = torch.randn(1, 1, 10).abs()
    
    # Valid: ProfileLogRates
    rates = ProfileLogRates(log_rates=torch.randn(1, 1, 10))
    loss = loss_fn(rates, targets)
    assert not torch.isnan(loss)
    
    # Invalid: ProfileLogits
    logits = ProfileLogits(logits=torch.randn(1, 1, 10))
    with pytest.raises(TypeError, match="requires ProfileLogRates"):
        loss_fn(logits, targets)
        
    # Invalid: ProfileCountOutput (Explicit check in code)
    counts = ProfileCountOutput(logits=torch.randn(1, 1, 10), log_counts=torch.randn(1, 1))
    with pytest.raises(TypeError, match="does not accept ProfileCountOutput"):
        loss_fn(counts, targets)

def test_coupled_poisson_multinomial_strictness():
    """Verify CoupledPoissonMultinomialLoss enforces strictness."""
    loss_fn = CoupledPoissonMultinomialLoss()
    targets = torch.randn(1, 1, 10).abs()
    
    # Valid: ProfileLogRates
    rates = ProfileLogRates(log_rates=torch.randn(1, 1, 10))
    loss = loss_fn(rates, targets)
    assert not torch.isnan(loss)
    
    # Invalid: ProfileLogits
    logits = ProfileLogits(logits=torch.randn(1, 1, 10))
    with pytest.raises(TypeError, match="requires ProfileLogRates"):
        loss_fn(logits, targets)

def test_mse_multinomial_requirements():
    """Verify MSEMultinomialLoss requires ProfileCountOutput."""
    loss_fn = MSEMultinomialLoss()
    targets = torch.randn(1, 1, 10).abs()
    
    # Valid
    counts = ProfileCountOutput(logits=torch.randn(1, 1, 10), log_counts=torch.randn(1, 1))
    loss_fn(counts, targets)
    
    # Invalid: ProfileLogits (missing counts)
    logits = ProfileLogits(logits=torch.randn(1, 1, 10))
    with pytest.raises(TypeError, match="requires ProfileCountOutput"):
        loss_fn(logits, targets)

# --- 3. Metrics Flexibility ---

def test_pearson_metric_polymorphism():
    """Verify FlattenedPearsonCorrCoef accepts both Logits and Rates."""
    metric = FlattenedPearsonCorrCoef(num_channels=1)
    
    # Setup consistent inputs
    # Let log_rates = [0, 1, 2]. Softmax -> probs
    # Let logits = [0, 1, 2]. Softmax -> probs (same shape)
    tensor = torch.tensor([[[0.0, 1.0, 2.0]]]) # (1, 1, 3)
    target = torch.tensor([[[0.1, 0.2, 0.7]]]) # Similar shape
    
    # Case 1: ProfileLogRates
    rates = ProfileLogRates(log_rates=tensor)
    metric.reset()
    metric.update(rates, target) # type: ignore
    val_rates = metric.compute()
    
    # Case 2: ProfileLogits
    logits = ProfileLogits(logits=tensor)
    metric.reset()
    metric.update(logits, target) # type: ignore
    val_logits = metric.compute()
    
    assert torch.isclose(val_rates, val_logits)
    assert not torch.isnan(val_rates)

def test_profile_mse_metric_polymorphism():
    """Verify ProfileMeanSquaredError accepts both Logits and Rates."""
    metric = ProfileMeanSquaredError()
    
    tensor = torch.tensor([[[0.0, 1.0, 2.0]]])
    target = torch.tensor([[[10.0, 20.0, 70.0]]]) # Probs: 0.1, 0.2, 0.7
    
    # Case 1: ProfileLogRates
    rates = ProfileLogRates(log_rates=tensor)
    metric.reset()
    metric.update(rates, target) # type: ignore
    val_rates = metric.compute()
    
    # Case 2: ProfileLogits
    logits = ProfileLogits(logits=tensor)
    metric.reset()
    metric.update(logits, target) # type: ignore
    val_logits = metric.compute()
    
    assert torch.isclose(val_rates, val_logits)
    assert not torch.isnan(val_rates)

def test_metrics_reject_invalid_types():
    """Verify metrics reject tensors or unknown types."""
    metric = FlattenedPearsonCorrCoef()
    with pytest.raises(TypeError, match="requires ProfileLogRates or ProfileLogits"):
        metric.update(torch.randn(1,1,10), torch.randn(1,1,10)) # type: ignore
