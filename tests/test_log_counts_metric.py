import torch
import torch.nn.functional as F
import pytest
from cerberus.metrics import LogCountsMeanSquaredError
from cerberus.output import ProfileCountOutput, ProfileLogRates, ProfileLogits

def test_log_counts_mse_basic():
    """Test LogCountsMeanSquaredError with simple inputs."""
    metric = LogCountsMeanSquaredError(count_per_channel=False)
    
    # Target: 10 counts globally
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 10.0
    true_log_total = torch.log1p(torch.tensor([10.0]))
    
    # Pred: Matches target
    pred_log_counts = true_log_total.clone().reshape(1, 1)
    
    # Use ProfileCountOutput
    out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_log_counts)
    
    metric.update(out, targets)
    val = metric.compute()
    
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)

def test_log_counts_mse_per_channel():
    """Test per-channel count MSE."""
    metric = LogCountsMeanSquaredError(count_per_channel=True)
    
    # Targets: Ch0=10, Ch1=100
    targets = torch.zeros(1, 2, 10)
    targets[0, 0, 0] = 10.0
    targets[0, 1, 0] = 100.0
    
    target_log_counts = torch.log1p(torch.tensor([[10.0, 100.0]])) # (1, 2)
    
    # Preds
    pred_log_counts = target_log_counts.clone()
    
    out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_log_counts)
    
    metric.update(out, targets)
    val = metric.compute()
    
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)
    
    # Perturb one channel
    pred_bad = pred_log_counts.clone()
    pred_bad[0, 0] += 1.0 # Error of 1.0 in log space
    
    out_bad = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_bad)
    metric.reset()
    metric.update(out_bad, targets)
    val_bad = metric.compute()
    
    # MSE: (1^2 + 0^2) / 2 = 0.5
    assert torch.isclose(val_bad, torch.tensor(0.5), atol=1e-6)

def test_log_counts_mse_from_log_rates():
    """Test derivation of log counts from ProfileLogRates."""
    metric = LogCountsMeanSquaredError(count_per_channel=False)
    
    # Rates: [10, 20] -> Sum 30. log(30).
    rates = torch.log(torch.tensor([[[10.0, 20.0]]])) # (1, 1, 2)
    out = ProfileLogRates(log_rates=rates)
    
    # Target: Sum 30.
    targets = torch.tensor([[[10.0, 20.0]]])
    
    metric.update(out, targets)
    val = metric.compute()
    
    # Should be close to 0 as logsumexp(rates) == log(30)
    # and log1p(sum(targets)) == log(31).
    # Wait, log1p vs log.
    # Metric uses log1p(target_sum).
    # Pred uses logsumexp(rates) which represents log(sum(exp(rates))).
    # Typically rates sum to total count.
    # So pred is log(30). Target is log(31).
    # Difference: log(31) - log(30) = log(31/30) approx log(1.033) approx 0.032.
    # MSE approx 0.001.
    
    expected_diff = (torch.log1p(torch.tensor(30.0)) - torch.log(torch.tensor(30.0))) ** 2
    assert torch.isclose(val, expected_diff, atol=1e-6)

def test_log_counts_implicit_log_targets():
    """Test with implicit_log_targets=True."""
    metric = LogCountsMeanSquaredError(implicit_log_targets=True)
    
    # Raw target total = 10.
    # Input target is log1p(raw)
    raw_target = torch.tensor([[[10.0]]])
    input_target = torch.log1p(raw_target)
    
    # Pred: log(11).
    # If implicit_log_targets is True, metric un-logs input_target -> 10.
    # Then sums -> 10. Then takes log1p -> log(11).
    # So if pred is log(11), MSE is 0.
    
    pred_log_counts = torch.log(torch.tensor([[11.0]]))
    out = ProfileCountOutput(logits=torch.zeros(1,1,1), log_counts=pred_log_counts)
    
    metric.update(out, input_target)
    val = metric.compute()
    
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-5)

def test_log_counts_global_aggregation():
    """Test aggregation of per-channel counts to global count."""
    metric = LogCountsMeanSquaredError(count_per_channel=False)
    
    # Pred: Per-channel log counts [log(10), log(20)]
    # Global count should be 10+20=30. log(30).
    pred_log_counts = torch.log(torch.tensor([[10.0, 20.0]])) # (1, 2)
    out = ProfileCountOutput(logits=torch.zeros(1,2,1), log_counts=pred_log_counts)
    
    # Target: [10, 20]. Sum=30. log1p(30).
    targets = torch.tensor([[[10.0], [20.0]]]).permute(0, 1, 2) # (1, 2, 1)
    
    metric.update(out, targets)
    val = metric.compute()
    
    expected_diff = (torch.log1p(torch.tensor(30.0)) - torch.log(torch.tensor(30.0))) ** 2
    assert torch.isclose(val, expected_diff, atol=1e-5)

def test_invalid_input_type():
    metric = LogCountsMeanSquaredError()
    with pytest.raises(TypeError, match="requires ProfileCountOutput or ProfileLogRates"):
        metric.update(ProfileLogits(logits=torch.randn(1,1,1)), torch.randn(1,1,1)) # type: ignore
