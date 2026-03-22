
import torch

from cerberus.metrics import CountProfileMeanSquaredError, CountProfilePearsonCorrCoef
from cerberus.output import ProfileCountOutput


def test_count_profile_pearson_expm1():
    # Setup
    length = 10
    
    # Create fake preds
    # log_counts = log1p(10) -> approx 2.3979
    # We want total_counts to be 10.
    # If we use exp, we get 11. If we use expm1, we get 10.
    
    expected_count = 10.0
    log_counts = torch.log1p(torch.tensor([[expected_count]])) # (B, 1)
    
    # Logits: non-uniform to ensure variance for Pearson
    logits = torch.arange(float(length)).unsqueeze(0).unsqueeze(0) # 0, 1, ..., 9
    torch.softmax(logits, dim=-1)
    
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # Initialize metric
    metric = CountProfilePearsonCorrCoef()
    
    # We want to check what total_counts is used inside update.
    # We create a target that matches preds_counts exactly if total_counts is 10.
    # If total_counts is 11, there will be a mismatch.
    
    # We need variance for Pearson.
    
    # Not asserting here because Pearson is scale invariant.
    # This test function is mostly a placeholder to ensure no crash
    # and we rely on the MSE test for value correctness.
    
    # Use arange to ensure variance in target
    target = torch.arange(float(length)).unsqueeze(0).unsqueeze(0) + 1.0
    
    metric.update(preds, target)
    val = metric.compute()
    assert not torch.isnan(val)

def test_count_profile_mse_expm1():
    # MSE definitely cares about scale.
    
    batch_size = 1
    channels = 1
    length = 10
    
    expected_count = 10.0
    log_counts = torch.log1p(torch.tensor([[expected_count]]))
    
    logits = torch.zeros(batch_size, channels, length)
    
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # Target that matches the correct reconstruction
    # Correct: preds_counts = 1/10 * 10 = 1.0 per position
    target = torch.ones(batch_size, channels, length) * 1.0
    
    metric = CountProfileMeanSquaredError()
    metric.update(preds, target)
    mse = metric.compute()
    
    # If correct (expm1): preds_counts = 1.0. MSE = (1.0 - 1.0)^2 = 0.
    # If incorrect (exp): preds_counts = 1/10 * 11 = 1.1. MSE = (1.1 - 1.0)^2 = 0.01.
    
    assert mse < 1e-6, f"MSE should be close to 0, got {mse}. Likely using exp instead of expm1."
