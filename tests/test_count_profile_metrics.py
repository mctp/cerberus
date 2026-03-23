import pytest
import torch

from cerberus.metrics import CountProfileMeanSquaredError, CountProfilePearsonCorrCoef
from cerberus.output import ProfileCountOutput

# ---------------------------------------------------------------------------
# Multi-channel global-count reconstruction tests
# ---------------------------------------------------------------------------


class TestMultiChannelGlobalCount:
    """Tests for predict_total_count=True with n_output_channels > 1.

    When log_counts is (B, 1) (global) but logits is (B, C, L) with C > 1,
    the total count must be divided by C before multiplying per-channel
    softmax probabilities.  Otherwise the reconstructed signal sums to
    C * total instead of total.
    """

    @pytest.fixture()
    def multichannel_data(self):
        """B=2, C=2, L=8 with global log_counts (B, 1)."""
        B, C, L = 2, 2, 8
        total_count = 100.0
        pseudocount = 1.0
        log_counts = torch.log(torch.tensor(total_count + pseudocount)).expand(B, 1)

        # Non-uniform logits so softmax has shape
        logits = torch.randn(B, C, L)

        # Build matching targets: softmax(logits) * (total / C) per channel
        probs = torch.softmax(logits, dim=-1)
        per_channel_count = total_count / C
        targets = probs * per_channel_count  # (B, C, L), sums to total per example

        return ProfileCountOutput(logits=logits, log_counts=log_counts), targets

    def test_mse_zero_for_perfect_prediction(self, multichannel_data):
        """MSE should be ~0 when predictions match targets after equal split."""
        preds, targets = multichannel_data
        metric = CountProfileMeanSquaredError(count_pseudocount=1.0)
        metric.update(preds, targets)
        mse = metric.compute()
        assert mse < 1e-5, f"MSE should be ~0 for perfect prediction, got {mse}"

    def test_mse_nonzero_without_fix(self, multichannel_data):
        """Demonstrates the C× overcounting bug: without the /C fix,
        MSE would be large even for a 'perfect' prediction."""
        preds, targets = multichannel_data
        # Manually compute the broken reconstruction (no /C)
        probs = torch.softmax(preds.logits, dim=-1)
        total = (torch.exp(preds.log_counts.float()) - 1.0).clamp_min(0.0)
        broken_recon = probs * total.unsqueeze(-1)  # each channel gets full total
        broken_mse = (broken_recon - targets).pow(2).mean()
        assert broken_mse > 1.0, (
            f"Without /C fix, MSE should be large, got {broken_mse}"
        )

    def test_reconstructed_signal_sums_to_total(self, multichannel_data):
        """The reconstructed signal should sum to total_count per example."""
        preds, _ = multichannel_data
        pseudocount = 1.0
        probs = torch.softmax(preds.logits, dim=-1)
        total = (torch.exp(preds.log_counts.float()) - pseudocount).clamp_min(0.0)
        C = preds.logits.shape[1]
        per_channel = total / C
        recon = probs * per_channel.unsqueeze(-1)
        recon_sum = recon.sum(dim=(1, 2))  # sum over channels and length
        expected = total.flatten()
        assert torch.allclose(recon_sum, expected, atol=1e-4), (
            f"Reconstructed sum {recon_sum} != total {expected}"
        )

    def test_pearson_unaffected_by_scaling(self, multichannel_data):
        """Pearson correlation is scale-invariant, so /C doesn't change the value."""
        preds, targets = multichannel_data
        metric = CountProfilePearsonCorrCoef(count_pseudocount=1.0)
        metric.update(preds, targets)
        val = metric.compute()
        # Perfect correlation since targets = softmax(logits) * const
        assert val > 0.99, f"Expected near-perfect Pearson, got {val}"

    def test_single_channel_unaffected(self):
        """When C=1, the fix should be a no-op."""
        B, C, L = 2, 1, 8
        total_count = 100.0
        log_counts = torch.log(torch.tensor(total_count + 1.0)).expand(B, 1)
        logits = torch.randn(B, C, L)
        probs = torch.softmax(logits, dim=-1)
        targets = probs * total_count  # (B, 1, L)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfileMeanSquaredError(count_pseudocount=1.0)
        metric.update(preds, targets)
        mse = metric.compute()
        assert mse < 1e-5, f"Single-channel MSE should be ~0, got {mse}"

    def test_per_channel_counts_unaffected(self):
        """When log_counts is (B, C) (per-channel), no division should happen."""
        B, C, L = 2, 2, 8
        per_channel_counts = torch.tensor([40.0, 60.0])
        pseudocount = 1.0
        log_counts = (
            torch.log(per_channel_counts + pseudocount).unsqueeze(0).expand(B, -1)
        )
        logits = torch.randn(B, C, L)
        probs = torch.softmax(logits, dim=-1)
        targets = probs * per_channel_counts.view(1, C, 1)

        preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
        metric = CountProfileMeanSquaredError(count_pseudocount=1.0)
        metric.update(preds, targets)
        mse = metric.compute()
        assert mse < 1e-5, f"Per-channel MSE should be ~0, got {mse}"


def test_count_profile_pearson_expm1():
    # Setup
    length = 10

    # Create fake preds
    # log_counts = log1p(10) -> approx 2.3979
    # We want total_counts to be 10.
    # If we use exp, we get 11. If we use expm1, we get 10.

    expected_count = 10.0
    log_counts = torch.log1p(torch.tensor([[expected_count]]))  # (B, 1)

    # Logits: non-uniform to ensure variance for Pearson
    logits = torch.arange(float(length)).unsqueeze(0).unsqueeze(0)  # 0, 1, ..., 9
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

    assert mse < 1e-6, (
        f"MSE should be close to 0, got {mse}. Likely using exp instead of expm1."
    )
