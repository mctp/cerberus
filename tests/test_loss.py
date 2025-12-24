import torch
import pytest
from torchmetrics import PearsonCorrCoef
import torch.nn as nn
from torchmetrics import MeanSquaredError
from cerberus.loss import BPNetLoss, BPNetPoissonLoss, FlattenedPearsonCorrCoef, get_default_metrics, get_default_loss

def test_get_default_loss():
    loss = get_default_loss()
    assert isinstance(loss, nn.PoissonNLLLoss)
    assert loss.log_input is False
    assert loss.full is False

def test_get_default_metrics():
    metrics = get_default_metrics(num_channels=3)
    assert "pearson" in metrics
    assert "mse" in metrics
    assert isinstance(metrics["pearson"], FlattenedPearsonCorrCoef)
    assert metrics["pearson"].num_channels == 3
    assert isinstance(metrics["mse"], MeanSquaredError)

def test_bpnet_loss_forward_flatten_true():
    """Test forward pass with flatten_channels=True (default BPNet behavior)"""
    batch_size = 2
    channels = 2
    length = 10
    
    loss_fn = BPNetLoss(alpha=1.0, flatten_channels=True)
    
    # Inputs
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = (profile_logits, pred_log_counts)
    
    # Targets (counts must be non-negative)
    targets = torch.randint(0, 5, (batch_size, channels, length)).float()
    
    # Forward
    loss = loss_fn(outputs, targets)
    
    # Checks
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Backward
    loss.backward()
    assert profile_logits.grad is not None
    assert pred_log_counts.grad is not None

def test_bpnet_loss_forward_flatten_false():
    """Test forward pass with flatten_channels=False"""
    batch_size = 2
    channels = 2
    length = 10
    
    loss_fn = BPNetLoss(alpha=1.0, flatten_channels=False)
    
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = (profile_logits, pred_log_counts)
    targets = torch.randint(0, 5, (batch_size, channels, length)).float()
    
    loss = loss_fn(outputs, targets)
    
    assert loss.dim() == 0
    loss.backward()
    assert profile_logits.grad is not None

def test_bpnet_loss_count_component():
    """Test that count loss behaves correctly (MSE on log1p)"""
    loss_fn = BPNetLoss(alpha=10.0, flatten_channels=True)
    
    # Example: Total counts = 10. log1p(10) = log(11).
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 10.0
    
    true_log_total = torch.log1p(torch.tensor([10.0]))
    
    # Case 1: Perfect prediction
    pred_log_counts_perfect = true_log_total.clone().reshape(1, 1)
    pred_log_counts_perfect.requires_grad_(True)
    
    # Case 2: Bad prediction
    pred_log_counts_bad = (true_log_total + 1.0).clone().reshape(1, 1)
    pred_log_counts_bad.requires_grad_(True)
    
    logits = torch.zeros(1, 1, 10, requires_grad=True)
    
    # Loss 1
    loss1 = loss_fn((logits, pred_log_counts_perfect), targets)
    loss1.backward()
    grad1 = pred_log_counts_perfect.grad
    
    # Loss 2
    loss2 = loss_fn((logits, pred_log_counts_bad), targets)
    loss2.backward()
    grad2 = pred_log_counts_bad.grad
    
    # For perfect prediction, gradient w.r.t count should be 0
    assert grad1 is not None
    assert torch.allclose(grad1, torch.zeros_like(grad1), atol=1e-6)
    
    # For bad prediction, gradient should be non-zero
    assert grad2 is not None
    assert not torch.allclose(grad2, torch.zeros_like(grad2))
    
    # Count loss contribution should be 0 for perfect case.
    # Total loss = Profile Loss + Alpha * Count Loss
    # We can check if loss increases when count prediction gets worse
    assert loss2 > loss1

def test_bpnet_loss_profile_logic():
    """Test basic profile logic: predicting higher logits where counts are high should lower loss."""
    loss_fn = BPNetLoss(alpha=0.0, flatten_channels=True) # Ignore count loss
    
    targets = torch.tensor([[[10.0, 0.0]]]) # 1 channel, length 2.
    
    # Good logits: high for index 0
    logits_good = torch.tensor([[[10.0, 0.0]]], requires_grad=True)
    
    # Bad logits: high for index 1
    logits_bad = torch.tensor([[[0.0, 10.0]]], requires_grad=True)
    
    dummy_counts = torch.zeros(1, 1)
    
    loss_good = loss_fn((logits_good, dummy_counts), targets)
    loss_bad = loss_fn((logits_bad, dummy_counts), targets)
    
    assert loss_good < loss_bad

def test_bpnet_loss_count_component_batch_gt_1():
    """Test count loss component with batch size > 1 to ensure shape handling is correct."""
    loss_fn = BPNetLoss(alpha=1.0, flatten_channels=True)
    
    batch_size = 3
    # Targets: (Batch, Channels, Length)
    targets = torch.zeros(batch_size, 1, 10)
    targets[0, 0, 0] = 10.0
    targets[1, 0, 0] = 20.0
    targets[2, 0, 0] = 30.0
    
    true_totals = torch.tensor([10.0, 20.0, 30.0])
    true_log_counts = torch.log1p(true_totals)
    
    # Predict exactly the true log counts
    # Shape (Batch, 1) to test the flattening logic
    pred_log_counts = true_log_counts.clone().reshape(batch_size, 1) 
    
    # Dummy profile logits
    profile_logits = torch.zeros(batch_size, 1, 10)
    
    # Calculate loss with perfect prediction
    # We set alpha=1.0, so Total Loss = Profile Loss + Count Loss.
    # Count Loss should be 0.
    loss_perfect = loss_fn((profile_logits, pred_log_counts), targets)
    
    # Perturb predictions
    pred_log_counts_bad = pred_log_counts.clone()
    pred_log_counts_bad[0] += 1.0 # Error on first item
    
    loss_bad = loss_fn((profile_logits, pred_log_counts_bad), targets)
    
    # Loss should increase
    assert loss_bad > loss_perfect
    
    # Also verify backward pass works
    pred_log_counts.requires_grad_(True)
    loss = loss_fn((profile_logits, pred_log_counts), targets)
    loss.backward()
    assert pred_log_counts.grad is not None
    assert pred_log_counts.grad.shape == (batch_size, 1)

def test_bpnet_loss_with_log_transform():
    """Test compatibility with log1p-transformed targets when configured correctly."""
    
    # 1. Setup
    targets_raw = torch.tensor([[[10.0, 20.0, 0.0]]]) # Batch=1, Chan=1, Len=3
    targets_logged = torch.log1p(targets_raw)
    
    logits = torch.randn(1, 1, 3, requires_grad=True)
    # Total count = 30. log1p(30) approx 3.434
    true_log_total = torch.log1p(torch.tensor([30.0]))
    pred_log_counts = true_log_total.clone().reshape(1, 1)
    pred_log_counts.requires_grad_(True)
    
    # 2. Standard loss with raw targets
    loss_fn_std = BPNetLoss(alpha=1.0)
    loss_std = loss_fn_std((logits, pred_log_counts), targets_raw)
    
    # 3. Loss with logged targets, implicit_log_targets=False (Should be WRONG)
    loss_fn_wrong = BPNetLoss(alpha=1.0, implicit_log_targets=False)
    loss_wrong = loss_fn_wrong((logits, pred_log_counts), targets_logged)
    
    assert not torch.isclose(loss_std, loss_wrong), "Loss should differ when targets are logged but not handled"
    
    # 4. Loss with logged targets, implicit_log_targets=True (Should match std)
    loss_fn_fixed = BPNetLoss(alpha=1.0, implicit_log_targets=True)
    loss_fixed = loss_fn_fixed((logits, pred_log_counts), targets_logged)
    
    assert torch.isclose(loss_std, loss_fixed, atol=1e-5), "Loss should match when implicit_log_targets is True"

def test_flattened_pearson_single_channel():
    """Test FlattenedPearsonCorrCoef with single channel input"""
    metric = FlattenedPearsonCorrCoef(num_channels=1)
    
    # Random perfect correlation
    preds = torch.randn(2, 1, 10)
    targets = preds * 2 + 1
    
    val = metric(preds, targets)
    assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)
    
    # Check that it matches manual calculation via base class with flattening
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    base_metric = PearsonCorrCoef(num_outputs=1)
    base_val = base_metric(preds_flat, targets_flat)
    
    assert torch.isclose(val, base_val)

def test_flattened_pearson_multi_channel():
    """Test FlattenedPearsonCorrCoef with multiple channels"""
    num_channels = 2
    metric = FlattenedPearsonCorrCoef(num_channels=num_channels)
    
    # Batch=1, Len=100
    # Ch1: Perfect positive correlation
    # Ch2: Perfect negative correlation
    c1 = torch.arange(100).float()
    c2 = torch.arange(100).float()
    
    preds = torch.stack([c1, c2], dim=0).unsqueeze(0) # (1, 2, 100)
    
    # Targets
    t1 = c1
    t2 = -c2
    targets = torch.stack([t1, t2], dim=0).unsqueeze(0) # (1, 2, 100)
    
    val = metric(preds, targets)
    
    # Expected: Mean(1.0, -1.0) = 0.0
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-5)

def test_flattened_pearson_vs_global():
    """Verify that per-channel mean is NOT equivalent to global correlation"""
    metric_channel = FlattenedPearsonCorrCoef(num_channels=2)
    metric_global = PearsonCorrCoef(num_outputs=1) # Treat everything as one vector
    
    # Construct scenario where channels differ in scale
    # Ch1: Range near 0
    # Ch2: Range near 100
    # Both are random noise (uncorrelated)
    
    # Using fixed seed for reproducibility within test logic if needed, 
    # but statistical property holds for random data usually.
    torch.manual_seed(42)
    
    preds = torch.zeros(1, 2, 10)
    targets = torch.zeros(1, 2, 10)
    
    # Ch1: uncorrelated noise
    preds[0, 0, :] = torch.randn(10)
    targets[0, 0, :] = torch.randn(10)
    
    # Ch2: uncorrelated noise + large offset
    preds[0, 1, :] = torch.randn(10) + 100
    targets[0, 1, :] = torch.randn(10) + 100
    
    # Per-channel correlation should be near 0 (average of two uncorrelated signals)
    val_channel = metric_channel(preds, targets)
    
    # Global correlation will see two clusters: one at 0, one at 100.
    # It will draw a line through them -> High positive correlation
    val_global = metric_global(preds.flatten(), targets.flatten())
    
    # Global correlation is dominated by the offset difference between channels
    assert val_global > 0.9 
    
    # Per-channel correlation correctly identifies lack of correlation
    assert abs(val_channel) < 0.5 
    
    assert not torch.isclose(val_channel, val_global, atol=0.1)

def test_bpnet_poisson_loss_forward():
    """Test forward pass of BPNetPoissonLoss"""
    batch_size = 2
    channels = 2
    length = 10
    
    loss_fn = BPNetPoissonLoss(alpha=1.0, flatten_channels=True)
    
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    # Poisson loss input log_input=True expects log(counts)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = (profile_logits, pred_log_counts)
    
    targets = torch.randint(0, 5, (batch_size, channels, length)).float()
    
    loss = loss_fn(outputs, targets)
    
    assert loss.dim() == 0
    loss.backward()
    assert profile_logits.grad is not None
    assert pred_log_counts.grad is not None

def test_bpnet_poisson_loss_count_component():
    """Test that count loss component uses PoissonNLLLoss"""
    loss_fn = BPNetPoissonLoss(alpha=10.0, flatten_channels=True)
    
    # Total counts = 10
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 10.0
    true_total = torch.tensor([10.0])
    
    # Perfect prediction: log(10) (since we use log_input=True)
    pred_log_counts_perfect = torch.log(true_total).reshape(1, 1)
    pred_log_counts_perfect.requires_grad_(True)
    
    # Bad prediction
    pred_log_counts_bad = torch.log(true_total + 10.0).reshape(1, 1)
    pred_log_counts_bad.requires_grad_(True)
    
    logits = torch.zeros(1, 1, 10, requires_grad=True)
    
    loss1 = loss_fn((logits, pred_log_counts_perfect), targets)
    loss2 = loss_fn((logits, pred_log_counts_bad), targets)
    
    assert loss2 > loss1

def test_bpnet_poisson_loss_implicit_log_targets():
    """Test implicit log targets handling in BPNetPoissonLoss"""
    loss_fn = BPNetPoissonLoss(alpha=1.0, implicit_log_targets=True)
    
    # Targets are log(x+1)
    targets_raw = torch.tensor([[[10.0]]])
    targets_logged = torch.log1p(targets_raw)
    
    outputs = (torch.zeros(1, 1, 1), torch.zeros(1, 1))
    
    loss = loss_fn(outputs, targets_logged)
    assert not torch.isnan(loss)
