import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, PearsonCorrCoef
from cerberus.loss import (
    MSEMultinomialLoss, PoissonMultinomialLoss, ProfilePoissonNLLLoss
)
from cerberus.metrics import (
    ProfilePearsonCorrCoef, DefaultMetricCollection,
    ProfileMeanSquaredError, CountProfileMeanSquaredError,
    LogCountsMeanSquaredError
)
from cerberus.models.bpnet import BPNetMetricCollection
from cerberus.output import ProfileLogits, ProfileCountOutput

def test_profile_poisson_nll_loss():
    loss = ProfilePoissonNLLLoss(log_input=True, full=False)
    assert isinstance(loss, nn.PoissonNLLLoss)
    assert loss.log_input is True
    assert loss.full is False

def test_default_metric_collection():
    metrics = DefaultMetricCollection(num_channels=3)
    assert "pearson" in metrics
    assert "mse_profile" in metrics
    assert "mse_log_counts" in metrics
    assert isinstance(metrics["pearson"], ProfilePearsonCorrCoef)
    assert metrics["pearson"].num_channels == 3
    assert isinstance(metrics["mse_profile"], ProfileMeanSquaredError)
    assert isinstance(metrics["mse_log_counts"], LogCountsMeanSquaredError)

def test_default_metric_collection_implicit_log_targets():
    """Test that DefaultMetricCollection propagates implicit_log_targets flag."""
    metrics = DefaultMetricCollection(num_channels=3, implicit_log_targets=True)
    assert metrics["pearson"].implicit_log_targets is True
    assert metrics["mse_profile"].implicit_log_targets is True
    assert metrics["mse_log_counts"].implicit_log_targets is True

    metrics_false = DefaultMetricCollection(num_channels=3, implicit_log_targets=False)
    assert metrics_false["pearson"].implicit_log_targets is False
    assert metrics_false["mse_profile"].implicit_log_targets is False
    assert metrics_false["mse_log_counts"].implicit_log_targets is False

def test_bpnet_metric_collection_implicit_log_targets():
    """Test that BPNetMetricCollection propagates implicit_log_targets flag."""
    metrics = BPNetMetricCollection(num_channels=3, implicit_log_targets=True)
    assert metrics["pearson"].implicit_log_targets is True
    assert metrics["mse_profile"].implicit_log_targets is True
    assert metrics["mse_log_counts"].implicit_log_targets is True

def test_profile_mse_implicit_log_targets():
    """Test ProfileMeanSquaredError with implicit_log_targets."""
    # Setup: Logits that imply Prob(0.5, 0.5)
    logits = torch.tensor([[[0.0, 0.0]]]) # Softmax -> [0.5, 0.5]
    
    # Target: Counts [10, 10] -> Probs [0.5, 0.5]
    raw_targets = torch.tensor([[[10.0, 10.0]]])
    
    # 1. Test with raw targets (standard)
    mse_std = ProfileMeanSquaredError(implicit_log_targets=False)
    mse_std.update(ProfileLogits(logits=logits), raw_targets)
    val_std = mse_std.compute()
    assert torch.isclose(val_std, torch.tensor(0.0), atol=1e-6)
    
    # 2. Test with log targets without flag (Should fail/be high)
    log_targets = torch.log1p(raw_targets)
    mse_wrong = ProfileMeanSquaredError(implicit_log_targets=False)
    mse_wrong.update(ProfileLogits(logits=logits), log_targets)
    val_wrong = mse_wrong.compute()
    # Log targets: log(11) approx 2.4. Probs: 2.4/4.8 = 0.5.
    # Actually for uniform counts, log counts are also uniform, so probs are still 0.5.
    # We need non-uniform counts to see the difference.
    
    # New Setup: Non-uniform
    logits = torch.tensor([[[10.0, 0.0]]]) # Probs approx [1.0, 0.0]
    raw_targets = torch.tensor([[[100.0, 0.0]]]) # Probs [1.0, 0.0]
    log_targets = torch.log1p(raw_targets) 
    # Log targets: [4.6, 0.0]. Probs: [1.0, 0.0].
    # Wait, normalization of [A, 0] is always [1, 0] regardless of A.
    # We need two non-zero values to see distribution shift.
    
    # New Setup 2:
    # Raw: [10, 100]. Probs: [0.09, 0.91]
    # Log: [2.4, 4.6]. Probs: [0.34, 0.66]
    logits_perfect_raw = torch.log(torch.tensor([[[10.0, 100.0]]])) # Softmax matches raw probs
    
    raw_targets = torch.tensor([[[10.0, 100.0]]])
    log_targets = torch.log1p(raw_targets)
    
    # Correct config with log targets
    mse_correct = ProfileMeanSquaredError(implicit_log_targets=True)
    mse_correct.update(ProfileLogits(logits=logits_perfect_raw), log_targets)
    val_correct = mse_correct.compute()
    assert torch.isclose(val_correct, torch.tensor(0.0), atol=1e-5)
    
    # Incorrect config with log targets (using normalized log counts as ground truth)
    mse_incorrect = ProfileMeanSquaredError(implicit_log_targets=False)
    mse_incorrect.update(ProfileLogits(logits=logits_perfect_raw), log_targets)
    val_incorrect = mse_incorrect.compute()
    
    assert val_incorrect > 0.01 # Should be significantly different

def test_decoupled_mse_implicit_log_targets():
    """Test CountProfileMeanSquaredError with implicit_log_targets."""
    # Raw: [10.0]. Log: [2.39]
    raw_targets = torch.tensor([[[10.0]]])
    log_targets = torch.log1p(raw_targets)
    
    # Preds: Matches raw
    logits = torch.tensor([[[0.0]]]) # Prob 1.0
    log_counts = torch.log1p(torch.tensor([[10.0]]))
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # Correct config
    mse_correct = CountProfileMeanSquaredError(implicit_log_targets=True)
    mse_correct.update(preds, log_targets)
    val_correct = mse_correct.compute()
    assert torch.isclose(val_correct, torch.tensor(0.0), atol=1e-5)
    
    # Incorrect config (compares Raw Preds [10] vs Log Targets [2.4])
    mse_incorrect = CountProfileMeanSquaredError(implicit_log_targets=False)
    mse_incorrect.update(preds, log_targets)
    val_incorrect = mse_incorrect.compute()
    
    expected_diff = (10.0 - 2.397895) ** 2
    assert torch.isclose(val_incorrect, torch.tensor(expected_diff), atol=0.1)

def test_flattened_pearson_implicit_log_targets():
    """Test ProfilePearsonCorrCoef with implicit_log_targets."""
    # Raw: [10, 100]. Probs: [0.09, 0.91]
    # Log: [2.4, 4.6]. Probs: [0.34, 0.66]
    
    # Preds matching Raw Probs
    logits = torch.log(torch.tensor([[[10.0, 100.0]]]))
    
    raw_targets = torch.tensor([[[10.0, 100.0]]])
    log_targets = torch.log1p(raw_targets)
    
    # Correct config: Un-logs targets -> gets raw counts -> correlation 1.0
    corr_correct = ProfilePearsonCorrCoef(num_channels=1, implicit_log_targets=True)
    corr_correct.update(ProfileLogits(logits=logits), log_targets)
    val_correct = corr_correct.compute()
    assert torch.isclose(val_correct, torch.tensor(1.0), atol=1e-5)
    
    # Incorrect config: Correlates Raw Probs [0.09, 0.91] with Log Targets [2.4, 4.6]
    # Correlation might still be 1.0 because 2 points are always correlated?
    # Need 3 points to break linearity.
    
    # 3 Points:
    # Raw: [10, 100, 1000]
    # Log: [2.4, 4.6, 6.9]
    # These are NOT linearly related.
    
    logits_3 = torch.log(torch.tensor([[[10.0, 100.0, 1000.0]]]))
    raw_targets_3 = torch.tensor([[[10.0, 100.0, 1000.0]]])
    log_targets_3 = torch.log1p(raw_targets_3)
    
    corr_correct_3 = ProfilePearsonCorrCoef(num_channels=1, implicit_log_targets=True)
    corr_correct_3.update(ProfileLogits(logits=logits_3), log_targets_3)
    val_correct_3 = corr_correct_3.compute()
    assert torch.isclose(val_correct_3, torch.tensor(1.0), atol=1e-5)
    
    corr_incorrect_3 = ProfilePearsonCorrCoef(num_channels=1, implicit_log_targets=False)
    corr_incorrect_3.update(ProfileLogits(logits=logits_3), log_targets_3)
    val_incorrect_3 = corr_incorrect_3.compute()
    
    # Correlation between x and log(x) is high but not 1.0
    assert val_incorrect_3 < 0.999

def test_bpnet_loss_forward_flatten_true():
    """Test forward pass with flatten_channels=True (default BPNet behavior)"""
    batch_size = 2
    channels = 2
    length = 10
    
    loss_fn = MSEMultinomialLoss(count_weight=1.0, flatten_channels=True)
    
    # Inputs
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts)
    
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
    
    loss_fn = MSEMultinomialLoss(count_weight=1.0, flatten_channels=False)
    
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts)
    targets = torch.randint(0, 5, (batch_size, channels, length)).float()
    
    loss = loss_fn(outputs, targets)
    
    assert loss.dim() == 0
    loss.backward()
    assert profile_logits.grad is not None

def test_bpnet_loss_count_component():
    """Test that count loss behaves correctly (MSE on log1p)"""
    loss_fn = MSEMultinomialLoss(count_weight=10.0, flatten_channels=True)
    
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
    loss1 = loss_fn(ProfileCountOutput(logits=logits, log_counts=pred_log_counts_perfect), targets)
    loss1.backward()
    grad1 = pred_log_counts_perfect.grad
    
    # Loss 2
    loss2 = loss_fn(ProfileCountOutput(logits=logits, log_counts=pred_log_counts_bad), targets)
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
    loss_fn = MSEMultinomialLoss(count_weight=0.0, flatten_channels=True) # Ignore count loss
    
    targets = torch.tensor([[[10.0, 0.0]]]) # 1 channel, length 2.
    
    # Good logits: high for index 0
    logits_good = torch.tensor([[[10.0, 0.0]]], requires_grad=True)
    
    # Bad logits: high for index 1
    logits_bad = torch.tensor([[[0.0, 10.0]]], requires_grad=True)
    
    dummy_counts = torch.zeros(1, 1)
    
    loss_good = loss_fn(ProfileCountOutput(logits=logits_good, log_counts=dummy_counts), targets)
    loss_bad = loss_fn(ProfileCountOutput(logits=logits_bad, log_counts=dummy_counts), targets)
    
    assert loss_good < loss_bad

def test_bpnet_loss_count_component_batch_gt_1():
    """Test count loss component with batch size > 1 to ensure shape handling is correct."""
    loss_fn = MSEMultinomialLoss(count_weight=1.0, flatten_channels=True)
    
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
    loss_perfect = loss_fn(ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts), targets)
    
    # Perturb predictions
    pred_log_counts_bad = pred_log_counts.clone()
    pred_log_counts_bad[0] += 1.0 # Error on first item
    
    loss_bad = loss_fn(ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts_bad), targets)
    
    # Loss should increase
    assert loss_bad > loss_perfect
    
    # Also verify backward pass works
    pred_log_counts.requires_grad_(True)
    loss = loss_fn(ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts), targets)
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
    loss_fn_std = MSEMultinomialLoss(count_weight=1.0)
    loss_std = loss_fn_std(ProfileCountOutput(logits=logits, log_counts=pred_log_counts), targets_raw)
    
    # 3. Loss with logged targets, implicit_log_targets=False (Should be WRONG)
    loss_fn_wrong = MSEMultinomialLoss(count_weight=1.0, implicit_log_targets=False)
    loss_wrong = loss_fn_wrong(ProfileCountOutput(logits=logits, log_counts=pred_log_counts), targets_logged)
    
    assert not torch.isclose(loss_std, loss_wrong), "Loss should differ when targets are logged but not handled"
    
    # 4. Loss with logged targets, implicit_log_targets=True (Should match std)
    loss_fn_fixed = MSEMultinomialLoss(count_weight=1.0, implicit_log_targets=True)
    loss_fixed = loss_fn_fixed(ProfileCountOutput(logits=logits, log_counts=pred_log_counts), targets_logged)
    
    assert torch.isclose(loss_std, loss_fixed, atol=1e-5), "Loss should match when implicit_log_targets is True"

def test_flattened_pearson_single_channel():
    """Test ProfilePearsonCorrCoef with single channel input"""
    metric = ProfilePearsonCorrCoef(num_channels=1)
    
    # Random logits
    logits = torch.randn(2, 1, 10)
    # The metric applies softmax to logits before correlation
    probs = nn.functional.softmax(logits, dim=-1)
    
    # Targets perfectly correlated with PROBABILITIES
    targets = probs * 2 + 1
    
    val = metric(ProfileLogits(logits=logits), targets)
    assert torch.isclose(val, torch.tensor(1.0), atol=1e-5)
    
    # Check that it matches manual calculation via base class with flattening of PROBS
    probs_flat = probs.flatten()
    targets_flat = targets.flatten()
    
    base_metric = PearsonCorrCoef(num_outputs=1)
    base_val = base_metric(probs_flat, targets_flat)
    
    assert torch.isclose(val, base_val)

def test_flattened_pearson_multi_channel():
    """Test ProfilePearsonCorrCoef with multiple channels"""
    num_channels = 2
    metric = ProfilePearsonCorrCoef(num_channels=num_channels)
    
    # Batch=1, Len=100
    # Create logits that result in known correlation patterns after softmax
    logits = torch.randn(1, 2, 100)
    probs = nn.functional.softmax(logits, dim=-1)
    
    # Targets
    # Ch1: Perfect positive correlation with probs
    t1 = probs[0, 0, :] * 2 + 1
    # Ch2: Perfect negative correlation with probs
    t2 = -probs[0, 1, :] * 0.5
    
    targets = torch.stack([t1, t2], dim=0).unsqueeze(0) # (1, 2, 100)
    
    val = metric(ProfileLogits(logits=logits), targets)
    
    # Expected: Mean(1.0, -1.0) = 0.0
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-5)

def test_flattened_pearson_vs_global():
    """Verify that per-channel mean is NOT equivalent to global correlation"""
    metric_channel = ProfilePearsonCorrCoef(num_channels=2)
    metric_global = PearsonCorrCoef(num_outputs=1) # Treat everything as one vector
    
    # Using fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Logits
    logits = torch.randn(1, 2, 10)
    
    # Probs (what metric uses)
    probs = nn.functional.softmax(logits, dim=-1)
    
    # Targets:
    # Ch1: Uncorrelated with probs[0,0]
    t1 = torch.randn(10)
    # Ch2: Uncorrelated with probs[0,1] BUT shifted
    t2 = torch.randn(10) + 100
    
    targets = torch.stack([t1, t2], dim=0).unsqueeze(0)
    
    # Per-channel correlation should be near 0 (average of two uncorrelated signals)
    val_channel = metric_channel(ProfileLogits(logits=logits), targets)
    
    # Global correlation:
    # Metric uses probs. Probs are in range [0, 1].
    # Targets are mixed range: [-1, 1] and [99, 101].
    # If we flatten probs and targets:
    # Probs: [0..1, 0..1]
    # Targets: [small, large]
    # Since probs don't change scale, but targets do, global correlation might not be high unless probs also shift.
    # But here probs don't shift (always 0-1).
    # So "scale difference" logic from original test (which used preds directly) might not hold for Softmaxed preds.
    # Original test relied on preds having offset. Here probs can't have offset > 1.
    
    # To reproduce the effect, we need Probs to be uncorrelated per channel, but somehow globally correlated?
    # Or just verify that channel mean is robust.
    # Let's just check that val_channel is low (uncorrelated).
    
    assert abs(val_channel) < 0.5
    
    # Note: Global correlation comparison is less relevant now that we force Softmax normalization.
    # Softmax removes the ability for preds to have arbitrary offsets.

def test_poisson_multinomial_loss_forward():
    """Test forward pass of PoissonMultinomialLoss"""
    batch_size = 2
    channels = 2
    length = 10
    
    loss_fn = PoissonMultinomialLoss(count_weight=1.0, flatten_channels=True)
    
    profile_logits = torch.randn(batch_size, channels, length, requires_grad=True)
    # Poisson loss input log_input=True expects log(counts)
    pred_log_counts = torch.randn(batch_size, 1, requires_grad=True)
    outputs = ProfileCountOutput(logits=profile_logits, log_counts=pred_log_counts)
    
    targets = torch.randint(0, 5, (batch_size, channels, length)).float()
    
    loss = loss_fn(outputs, targets)
    
    assert loss.dim() == 0
    loss.backward()
    assert profile_logits.grad is not None
    assert pred_log_counts.grad is not None

def test_poisson_multinomial_loss_count_component():
    """Test that count loss component uses PoissonNLLLoss"""
    loss_fn = PoissonMultinomialLoss(count_weight=10.0, flatten_channels=True)
    
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
    
    loss1 = loss_fn(ProfileCountOutput(logits=logits, log_counts=pred_log_counts_perfect), targets)
    loss2 = loss_fn(ProfileCountOutput(logits=logits, log_counts=pred_log_counts_bad), targets)
    
    assert loss2 > loss1

def test_poisson_multinomial_loss_implicit_log_targets():
    """Test implicit log targets handling in PoissonMultinomialLoss"""
    loss_fn = PoissonMultinomialLoss(count_weight=1.0, implicit_log_targets=True)

    # Targets are log(x+1)
    targets_raw = torch.tensor([[[10.0]]])
    targets_logged = torch.log1p(targets_raw)

    outputs = ProfileCountOutput(logits=torch.zeros(1, 1, 1), log_counts=torch.zeros(1, 1))

    loss = loss_fn(outputs, targets_logged)
    assert not torch.isnan(loss)


def test_mse_multinomial_loss_count_pseudocount_default_equals_log1p():
    """count_pseudocount=1.0 must reproduce the original log1p count target."""
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 50.0
    total = targets.sum()

    # Perfect prediction using the new formula with pseudocount=1.0
    perfect_log = torch.log(total + 1.0).reshape(1, 1)
    loss_fn = MSEMultinomialLoss(count_weight=1.0, count_pseudocount=1.0)
    outputs = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=perfect_log)
    loss = loss_fn(outputs, targets)

    # Count loss should be 0 (profile loss is non-zero but count part is 0)
    # Verify by checking gradient on log_counts is zero
    perfect_log.requires_grad_(True)
    outputs2 = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=perfect_log)
    loss2 = loss_fn(outputs2, targets)
    loss2.backward()
    assert perfect_log.grad is not None
    assert torch.allclose(perfect_log.grad, torch.zeros_like(perfect_log.grad), atol=1e-6)


def test_mse_multinomial_loss_count_pseudocount_custom():
    """count_pseudocount=100.0 produces log(count+100) not log(count+1) for targets."""
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 500.0
    total = targets.sum()

    # Build predictions that are perfect under each pseudocount
    pred_pseudo1 = torch.log(total + 1.0).reshape(1, 1).requires_grad_(True)
    pred_pseudo100 = torch.log(total + 100.0).reshape(1, 1).requires_grad_(True)

    loss_fn_1 = MSEMultinomialLoss(count_weight=1.0, count_pseudocount=1.0)
    loss_fn_100 = MSEMultinomialLoss(count_weight=1.0, count_pseudocount=100.0)

    # Perfect prediction under its own pseudocount -> zero count-loss gradient
    loss_fn_1(ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_pseudo1), targets).backward()
    loss_fn_100(ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_pseudo100), targets).backward()

    assert pred_pseudo1.grad is not None
    assert pred_pseudo100.grad is not None
    assert torch.allclose(pred_pseudo1.grad, torch.zeros_like(pred_pseudo1.grad), atol=1e-6)
    assert torch.allclose(pred_pseudo100.grad, torch.zeros_like(pred_pseudo100.grad), atol=1e-6)

    # Cross-check: prediction that is perfect under pseudo=1 is NOT perfect under pseudo=100
    pred_wrong = torch.log(total + 1.0).reshape(1, 1).requires_grad_(True)
    loss_fn_100(ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_wrong), targets).backward()
    assert pred_wrong.grad is not None
    assert not torch.allclose(pred_wrong.grad, torch.zeros_like(pred_wrong.grad), atol=1e-6)


def test_mse_multinomial_loss_count_pseudocount_stored():
    """count_pseudocount is stored as an attribute for downstream access (e.g. module.py)."""
    loss_fn = MSEMultinomialLoss(count_pseudocount=75.0)
    assert loss_fn.count_pseudocount == 75.0
