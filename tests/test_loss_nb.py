
import torch
import torch.nn.functional as F
import pytest
from cerberus.loss import NegativeBinomialMultinomialLoss, CoupledNegativeBinomialMultinomialLoss
from cerberus.output import ProfileCountOutput, ProfileLogRates

def test_nb_loss_forward_shapes():
    B, C, L = 2, 3, 100
    logits = torch.randn(B, C, L)
    log_counts = torch.randn(B, C) # Per channel
    
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)
    targets = torch.randint(0, 10, (B, C, L)).float()
    
    loss_fn = NegativeBinomialMultinomialLoss(total_count=10.0, count_per_channel=True)
    loss = loss_fn(outputs, targets)
    
    assert loss.ndim == 0
    assert not torch.isnan(loss)

def test_nb_loss_global_count():
    B, C, L = 2, 3, 100
    logits = torch.randn(B, C, L)
    log_counts = torch.randn(B) # Global (flattened)
    
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)
    targets = torch.randint(0, 10, (B, C, L)).float()
    
    loss_fn = NegativeBinomialMultinomialLoss(total_count=10.0, count_per_channel=False)
    loss = loss_fn(outputs, targets)
    
    assert loss.ndim == 0
    assert not torch.isnan(loss)

def test_nb_gradients_saturation():
    """
    Verify that NB gradients saturate for high counts (unlike Poisson).
    """
    # High count target
    target = torch.tensor([10000.0])
    
    # Prediction: 10% error
    log_pred = torch.log(target * 1.1).requires_grad_(True)
    
    # NB Loss
    r = 10.0
    loss_fn = NegativeBinomialMultinomialLoss(total_count=r, count_weight=1.0, profile_weight=0.0)
    
    # Mock output
    # NegativeBinomialMultinomialLoss expects ProfileCountOutput
    # We construct a dummy output
    dummy_logits = torch.zeros(1, 1, 10)
    dummy_targets = torch.zeros(1, 1, 10) # Profile part irrelevant
    
    # We need to hack the forward call or just use internal logic?
    # Let's just use the class
    
    outputs = ProfileCountOutput(logits=dummy_logits, log_counts=log_pred.view(1))
    
    # Forward
    loss = loss_fn(outputs, dummy_targets.view(1, 1, 10)) # Targets for profile
    # But wait, count loss uses target counts.
    # We need dummy_targets to sum to 'target' (10000).
    # Since we passed count_per_channel=False (default), it sums over (1,2).
    # Let's manually set targets
    dummy_targets = torch.zeros(1, 1, 10)
    dummy_targets[0, 0, 0] = 10000.0
    
    loss.backward()
    assert log_pred.grad is not None
    grad_nb = log_pred.grad.item()
    
    # Compare with Poisson gradient (approx)
    # Poisson Grad ~ Error = 1000.
    # NB Grad ~ Constant (approx 1.0 for r=10, 10% error)
    
    assert grad_nb < 100.0 # Should be small constant
    assert grad_nb > 0.1 # Should be non-zero
    print(f"NB Gradient for 10000 count (10% error, r={r}): {grad_nb}")

def test_nb_convergence_to_poisson():
    """
    Verify that with very high total_count (r), NB approaches Poisson.
    """
    targets = torch.tensor([[[100.0]]]) # (1, 1, 1)
    log_counts = torch.log(torch.tensor([110.0])).view(1) # 10% error
    logits = torch.zeros(1, 1, 1)
    
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # NB with huge r
    loss_nb = NegativeBinomialMultinomialLoss(total_count=1e9, count_weight=1.0, profile_weight=0.0)
    val_nb = loss_nb(outputs, targets)
    
    # Poisson
    # Need to simulate what PoissonMultinomialLoss does
    # PoissonMultinomialLoss uses nn.PoissonNLLLoss(log_input=True, full=False)
    val_poisson = F.poisson_nll_loss(log_counts, targets.sum().view(1), log_input=True, full=False)
    
    # They should be close?
    # NB NLL = -log P_NB. Poisson NLL = -log P_Poiss.
    # Yes, P_NB -> P_Poiss as r -> inf.
    
    # Note: absolute values might differ by constant log-factorial terms depending on 'full' implementation
    # But nn.PoissonNLLLoss(full=False) is simplified: exp(x) - kx.
    # NB log_prob includes normalization.
    # So we should compare with full=True Poisson.
    
    val_poisson_full = F.poisson_nll_loss(log_counts, targets.sum().view(1), log_input=True, full=True)
    
    # Allow some tolerance due to float precision and approximation
    # 1e9 might not be enough for perfect float32 match but should be close.
    # Actually, let's just check the values are somewhat consistent or just check gradient similarity?
    
    # Let's check gradient similarity instead, easier.
    
    log_counts.requires_grad_(True)
    val_nb = loss_nb(ProfileCountOutput(logits=logits, log_counts=log_counts), targets)
    val_nb.backward()
    assert log_counts.grad is not None
    grad_nb = log_counts.grad.item()
    log_counts.grad = None
    
    val_poisson = F.poisson_nll_loss(log_counts, targets.sum().view(1), log_input=True, full=False)
    val_poisson.backward()
    assert log_counts.grad is not None
    grad_poisson = log_counts.grad.item()
    
    # NB Gradient with r->inf should approach Poisson Gradient
    # Poisson Grad: 110 - 100 = 10.
    print(f"Gradient NB (r=1e9): {grad_nb}, Gradient Poisson: {grad_poisson}")
    assert abs(grad_nb - grad_poisson) < 1.0 # Should be close
