
import torch
from cerberus.loss import (
    MSEMultinomialLoss, CoupledMSEMultinomialLoss, 
    PoissonMultinomialLoss, CoupledPoissonMultinomialLoss
)
from cerberus.output import ProfileOutput, ProfileCountOutput

def test_bpnet_loss_count_per_channel():
    """Test BPNetLoss with count_per_channel=True"""
    batch_size = 2
    channels = 3
    length = 10
    
    loss_fn = MSEMultinomialLoss(count_weight=1.0, count_per_channel=True)
    
    # Targets: different counts per channel
    # Ch0: 10, Ch1: 20, Ch2: 30
    targets = torch.zeros(batch_size, channels, length)
    targets[:, 0, :] = 1.0 # Sum=10
    targets[:, 1, :] = 2.0 # Sum=20
    targets[:, 2, :] = 3.0 # Sum=30
    
    # Expected log counts per channel
    target_counts = targets.sum(dim=2) # (B, 3) -> [10, 20, 30]
    target_log_counts = torch.log1p(target_counts)
    
    # 1. Perfect Prediction
    pred_log_counts_perfect = target_log_counts.clone()
    pred_log_counts_perfect.requires_grad_(True)
    
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    
    outputs_perfect = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts_perfect)
    loss_perfect = loss_fn(outputs_perfect, targets)
    
    # Count component should be 0. Total loss = Profile Loss.
    
    # 2. Imperfect Prediction (mismatch on one channel)
    pred_log_counts_bad = target_log_counts.clone()
    pred_log_counts_bad[:, 1] += 1.0 # Error on Ch1
    pred_log_counts_bad.requires_grad_(True)
    
    outputs_bad = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts_bad)
    loss_bad = loss_fn(outputs_bad, targets)
    
    assert loss_bad > loss_perfect
    
    # Check gradients
    loss_bad.backward()
    assert pred_log_counts_bad.grad is not None
    # Gradient should be non-zero for Ch1
    assert torch.abs(pred_log_counts_bad.grad[:, 1]).sum() > 0
    # Gradient should be zero for Ch0, Ch2 (since prediction was perfect there)
    assert torch.allclose(pred_log_counts_bad.grad[:, 0], torch.zeros(batch_size))
    assert torch.allclose(pred_log_counts_bad.grad[:, 2], torch.zeros(batch_size))

def test_coupled_bpnet_loss_count_per_channel():
    """Test CoupledBPNetLoss with count_per_channel=True"""
    batch_size = 1
    channels = 2
    length = 2
    
    loss_fn = CoupledMSEMultinomialLoss(count_weight=1.0, count_per_channel=True)
    
    # Targets
    targets = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]]) # Ch0 sum=10, Ch1 sum=20
    
    # Logits designed to match counts
    # Ch0: log(5), log(5) -> sum exp = 10 -> log sum exp = log(10)
    # Ch1: log(10), log(10) -> sum exp = 20 -> log sum exp = log(20)
    val0 = torch.log(torch.tensor(5.0))
    val1 = torch.log(torch.tensor(10.0))
    
    logits_perfect = torch.zeros(batch_size, channels, length)
    logits_perfect[:, 0, :] = val0
    logits_perfect[:, 1, :] = val1
    logits_perfect.requires_grad_(True)
    
    outputs_perfect = ProfileOutput(logits=logits_perfect)
    
    # Count loss involves log1p(targets).
    # Coupled loss calculates logsumexp(logits).
    # logsumexp(logits) gives log(10) approx 2.30.
    # log1p(targets) gives log(11) approx 2.39.
    # So perfect match isn't exactly 0 loss unless we adjust.
    # But we can compare against a perturbed version.
    
    loss_perfect = loss_fn(outputs_perfect, targets)
    
    # Perturb Ch1 logits to be higher
    logits_bad = logits_perfect.clone()
    logits_bad[:, 1, :] += 1.0 # Increases count estimate for Ch1
    logits_bad.requires_grad_(True)
    
    outputs_bad = ProfileOutput(logits=logits_bad)
    loss_bad = loss_fn(outputs_bad, targets)
    
    assert loss_bad > loss_perfect

def test_poisson_multinomial_loss_count_per_channel():
    """Test PoissonMultinomialLoss with count_per_channel=True"""
    batch_size = 2
    channels = 2
    length = 5
    
    loss_fn = PoissonMultinomialLoss(count_weight=1.0, count_per_channel=True)
    
    targets = torch.randint(1, 10, (batch_size, channels, length)).float()
    target_counts = targets.sum(dim=2)
    
    # Poisson loss expects log counts
    pred_log_counts_perfect = torch.log(target_counts)
    pred_log_counts_perfect.requires_grad_(True)
    
    profile_logits = torch.randn(batch_size, channels, length)
    
    outputs_perfect = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts_perfect)
    loss_perfect = loss_fn(outputs_perfect, targets)
    
    # Perturb Ch1
    pred_log_counts_bad = pred_log_counts_perfect.detach().clone()
    pred_log_counts_bad[:, 1] += 1.0
    pred_log_counts_bad.requires_grad_(True)
    
    outputs_bad = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts_bad)
    loss_bad = loss_fn(outputs_bad, targets)
    
    assert loss_bad > loss_perfect
    
    loss_bad.backward()
    # Check gradients separate per channel
    grad = pred_log_counts_bad.grad
    assert torch.abs(grad[:, 1]).sum() > 0 # Ch1 changed
    # Ch0 perfect, but Poisson loss derivative isn't 0 at the mode unless using Stirling approx or specific conditions?
    # Poisson NLL = lambda - k * log(lambda). d/d(log_lam) = lam - k.
    # If log_lam = log(k), then lam = k. Then lam - k = 0.
    # So gradient should be 0 at optimum.
    assert torch.allclose(grad[:, 0], torch.zeros(batch_size), atol=1e-5)

def test_coupled_poisson_multinomial_loss_count_per_channel():
    """Test CoupledPoissonMultinomialLoss with count_per_channel=True"""
    batch_size = 1
    channels = 2
    length = 2
    
    loss_fn = CoupledPoissonMultinomialLoss(count_weight=1.0, count_per_channel=True)
    
    # Targets: Ch0=10, Ch1=20
    targets = torch.tensor([[[5.0, 5.0], [10.0, 10.0]]]) 
    
    # Logits matching counts
    val0 = torch.log(torch.tensor(5.0))
    val1 = torch.log(torch.tensor(10.0))
    
    logits = torch.zeros(batch_size, channels, length)
    logits[:, 0, :] = val0
    logits[:, 1, :] = val1
    logits.requires_grad_(True)
    
    outputs = ProfileOutput(logits=logits)
    
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    # Gradients should be close to zero because logits sum to targets
    # Sum(exp(logits)) = targets.
    # Poisson gradient w.r.t log_lambda is lambda - k.
    # Here lambda = sum(exp(logits)) = k. So count loss grad is 0.
    # Profile loss grad might not be 0 depending on shape, but shape is flat and targets are flat.
    # Targets [5, 5] -> uniform. Logits [log5, log5] -> uniform.
    # So shape loss should also be minimal/optimal.
    
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-5)
