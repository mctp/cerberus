import pytest
import torch

from cerberus.metrics import LogCountsMeanSquaredError
from cerberus.output import ProfileCountOutput, ProfileLogits, ProfileLogRates


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

def test_log_counts_log1p_targets():
    """Test with log1p_targets=True."""
    metric = LogCountsMeanSquaredError(log1p_targets=True)
    
    # Raw target total = 10.
    # Input target is log1p(raw)
    raw_target = torch.tensor([[[10.0]]])
    input_target = torch.log1p(raw_target)
    
    # Pred: log(11).
    # If log1p_targets is True, metric un-logs input_target -> 10.
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


def test_log_counts_mse_pseudocount_default_equals_log1p():
    """count_pseudocount=1.0 (default) must be numerically equivalent to the original log1p."""
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 50.0

    # Metric computes log(50 + 1.0); perfect pred is log(51).
    pred = torch.log(torch.tensor([[51.0]]))
    out = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred)

    metric = LogCountsMeanSquaredError(count_pseudocount=1.0)
    metric.update(out, targets)
    assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-6)


def test_log_counts_mse_pseudocount_100():
    """count_pseudocount=100.0 uses log(count+100) as the target."""
    targets = torch.zeros(1, 1, 10)
    targets[0, 0, 0] = 500.0  # total = 500

    # Perfect prediction under pseudocount=100: log(600)
    pred_perfect = torch.log(torch.tensor([[600.0]]))
    out_perfect = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_perfect)

    metric = LogCountsMeanSquaredError(count_pseudocount=100.0)
    metric.update(out_perfect, targets)
    assert torch.isclose(metric.compute(), torch.tensor(0.0), atol=1e-6)

    # Prediction using pseudocount=1 is wrong under pseudocount=100
    pred_wrong = torch.log(torch.tensor([[501.0]]))
    out_wrong = ProfileCountOutput(logits=torch.zeros(1, 1, 10), log_counts=pred_wrong)

    metric2 = LogCountsMeanSquaredError(count_pseudocount=100.0)
    metric2.update(out_wrong, targets)
    val = metric2.compute()
    expected = (torch.log(torch.tensor(501.0)) - torch.log(torch.tensor(600.0))) ** 2
    assert torch.isclose(val, expected, atol=1e-5)


def test_log_counts_mse_pseudocount_stored():
    """count_pseudocount is accessible as an attribute."""
    metric = LogCountsMeanSquaredError(count_pseudocount=42.0)
    assert metric.count_pseudocount == 42.0


def test_log_counts_mse_multichannel_offset_log():
    """Multi-channel aggregation in offset-log space must use invert-sum-reapply.

    Without log_counts_include_pseudocount=True, logsumexp gives
    log(c0 + c1 + 2*p) instead of the correct log(c0 + c1 + p).
    """
    from cerberus.metrics import LogCountsPearsonCorrCoef

    p = 150.0
    c0, c1 = 1000.0, 2000.0
    total = c0 + c1  # 3000

    # Per-channel log_counts in offset-log space: log(count + p)
    pred_log_counts = torch.log(torch.tensor([[c0 + p, c1 + p]]))  # (1, 2)
    out = ProfileCountOutput(logits=torch.zeros(1, 2, 10), log_counts=pred_log_counts)

    # Target: total = 3000, target_log = log(3000 + 150) = log(3150)
    targets = torch.zeros(1, 2, 10)
    targets[0, 0, 0] = c0
    targets[0, 1, 0] = c1

    # --- With the fix (log_counts_include_pseudocount=True): MSE should be 0 ---
    metric_fixed = LogCountsMeanSquaredError(
        count_pseudocount=p, log_counts_include_pseudocount=True,
    )
    metric_fixed.update(out, targets)
    val_fixed = metric_fixed.compute()
    assert torch.isclose(val_fixed, torch.tensor(0.0), atol=1e-5), (
        f"Invert-sum-reapply should give zero MSE, got {val_fixed.item()}"
    )

    # --- Without the fix (log_counts_include_pseudocount=False): MSE > 0 ---
    metric_broken = LogCountsMeanSquaredError(
        count_pseudocount=p, log_counts_include_pseudocount=False,
    )
    metric_broken.update(out, targets)
    val_broken = metric_broken.compute()
    # logsumexp gives log(c0+p + c1+p) = log(total + 2p) = log(3300)
    # target is log(total + p) = log(3150)
    expected_broken = (torch.log(torch.tensor(total + 2 * p)) - torch.log(torch.tensor(total + p))) ** 2
    assert torch.isclose(val_broken, expected_broken, atol=1e-5), (
        f"Plain logsumexp should give non-zero MSE, got {val_broken.item()}"
    )
    assert val_broken > 1e-5, "Bug should be measurable"

    # --- Same test for LogCountsPearsonCorrCoef: verify it doesn't crash ---
    pearson = LogCountsPearsonCorrCoef(
        count_pseudocount=p, log_counts_include_pseudocount=True,
    )
    # Need >1 example for correlation
    targets2 = torch.zeros(2, 2, 10)
    targets2[0, 0, 0] = c0
    targets2[0, 1, 0] = c1
    targets2[1, 0, 0] = 500.0
    targets2[1, 1, 0] = 800.0
    pred2 = torch.log(torch.tensor([[c0 + p, c1 + p], [500 + p, 800 + p]]))
    out2 = ProfileCountOutput(logits=torch.zeros(2, 2, 10), log_counts=pred2)
    pearson.update(out2, targets2)
    r = pearson.compute()
    assert torch.isclose(r, torch.tensor(1.0), atol=1e-4), (
        f"Perfect predictions should give r≈1, got {r.item()}"
    )
