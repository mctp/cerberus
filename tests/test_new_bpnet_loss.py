import torch
import torch.nn.functional as F

from cerberus.loss import MSEMultinomialLoss, PoissonMultinomialLoss
from cerberus.models.bpnet import BPNetLoss
from cerberus.output import ProfileCountOutput


def test_mse_multinomial_loss_average_channels():
    # Setup: 2 Channels
    batch_size = 1
    channels = 2
    length = 5

    # logits: (B, C, L)
    logits = torch.zeros(batch_size, channels, length)
    # targets: (B, C, L)
    targets = torch.zeros(batch_size, channels, length)

    # Channel 0: Perfect prediction (Loss approx 0 + constant)
    # Channel 1: High error
    # Let's use simple logic. Profile loss involves -sum(target * log_probs).
    # If targets are all 0, profile loss depends on log_fact_sum term only?
    # log_gamma(0+1) = 0.
    # So if targets=0, loss = 0 - 0 + 0 = 0?

    # Make Ch0 loss = A, Ch1 loss = B.
    # Summing: A + B. Averaging: (A + B)/2.

    # Ch0: targets=[1, 0...], logits=[10, 0...] (Good)
    # Ch1: targets=[1, 0...], logits=[0, 10...] (Bad)

    targets[:, 0, 0] = 1.0
    targets[:, 1, 0] = 1.0

    logits[:, 0, 0] = 10.0  # High prob at index 0
    logits[:, 1, 1] = 10.0  # High prob at index 1 (Mismatch)

    # Calculate losses manually (conceptually)
    # Loss0 is low. Loss1 is high.

    # 1. Summing (default)
    loss_fn_sum = MSEMultinomialLoss(count_weight=0.0, average_channels=False)
    # Dummy counts (use (B, 1) to avoid broadcasting warning with count_per_channel=False)
    preds = ProfileCountOutput(logits=logits, log_counts=torch.zeros(batch_size, 1))

    loss_sum = loss_fn_sum(preds, targets)

    # 2. Averaging
    loss_fn_avg = MSEMultinomialLoss(count_weight=0.0, average_channels=True)
    loss_avg = loss_fn_avg(preds, targets)

    # loss_sum should be approx 2 * loss_avg (since B=1)
    assert torch.isclose(loss_sum, loss_avg * 2.0, rtol=1e-5)


def test_mse_multinomial_loss_profile_weight():
    # Setup
    loss_fn_base = MSEMultinomialLoss(count_weight=0.0, profile_weight=1.0)
    loss_fn_scaled = MSEMultinomialLoss(count_weight=0.0, profile_weight=5.0)

    logits = torch.randn(1, 1, 10)
    targets = torch.randint(0, 5, (1, 1, 10)).float()
    preds = ProfileCountOutput(logits=logits, log_counts=torch.zeros(1, 1))

    l1 = loss_fn_base(preds, targets)
    l2 = loss_fn_scaled(preds, targets)

    assert torch.isclose(l2, l1 * 5.0)


def test_bpnet_loss_alpha_beta():
    """Test BPNetLoss alpha/beta parameter mapping"""
    # alpha -> count_weight
    # beta -> profile_weight
    # average_channels=True

    alpha = 2.0
    beta = 3.0

    loss_fn = BPNetLoss(alpha=alpha, beta=beta)

    assert loss_fn.count_weight == alpha
    assert loss_fn.profile_weight == beta
    assert loss_fn.average_channels is True

    # Check computation
    logits = torch.randn(1, 2, 10)
    log_counts = torch.randn(1, 1)  # Global count prediction logic usually
    # Note: BPNetLoss inherits MSEMultinomialLoss which expects count shapes to match if count_per_channel=True
    # Default count_per_channel=False. So it flattens.

    targets = torch.randint(0, 5, (1, 2, 10)).float()

    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)

    loss = loss_fn(preds, targets)

    # Manual calculation with components
    # Extract profile loss (averaged)
    prof_loss_fn = MSEMultinomialLoss(
        count_weight=0.0, profile_weight=1.0, average_channels=True
    )
    prof_loss = prof_loss_fn(preds, targets)

    # Extract count loss (global sum)
    # MSEMultinomialLoss count logic: F.mse_loss(pred.flatten(), log1p(targets.sum(1,2)))
    target_counts = torch.log1p(targets.sum(dim=(1, 2)))
    count_loss = F.mse_loss(log_counts.flatten(), target_counts)

    expected_loss = beta * prof_loss + alpha * count_loss

    assert torch.isclose(loss, expected_loss)


def test_poisson_multinomial_loss_average_channels():
    # Setup same as MSE test
    batch_size = 1
    channels = 2
    length = 5

    targets = torch.zeros(batch_size, channels, length)
    targets[:, 0, 0] = 1.0
    targets[:, 1, 0] = 1.0

    logits = torch.zeros(batch_size, channels, length)
    logits[:, 0, 0] = 10.0
    logits[:, 1, 1] = 10.0

    preds = ProfileCountOutput(
        logits=logits, log_counts=torch.zeros(batch_size, 1)
    )  # Dummy counts

    # 1. Averaging (Default for PoissonMultinomialLoss currently?)
    # I set average_channels=True as default in my update.
    loss_fn_avg = PoissonMultinomialLoss(count_weight=0.0, average_channels=True)
    loss_avg = loss_fn_avg(preds, targets)

    # 2. Summing
    loss_fn_sum = PoissonMultinomialLoss(count_weight=0.0, average_channels=False)
    loss_sum = loss_fn_sum(preds, targets)

    # loss_sum should be approx 2 * loss_avg
    assert torch.isclose(loss_sum, loss_avg * 2.0, rtol=1e-5)


def test_poisson_multinomial_loss_profile_weight():
    loss_fn_base = PoissonMultinomialLoss(count_weight=0.0, profile_weight=1.0)
    loss_fn_scaled = PoissonMultinomialLoss(count_weight=0.0, profile_weight=5.0)

    logits = torch.randn(1, 1, 10)
    targets = torch.randint(0, 5, (1, 1, 10)).float()
    preds = ProfileCountOutput(logits=logits, log_counts=torch.zeros(1, 1))

    l1 = loss_fn_base(preds, targets)
    l2 = loss_fn_scaled(preds, targets)

    assert torch.isclose(l2, l1 * 5.0)


def test_bpnet_loss_weights_zero():
    """Test BPNetLoss with zero weights."""
    loss_fn = BPNetLoss(alpha=0.0, beta=0.0)
    logits = torch.randn(1, 1, 10)
    log_counts = torch.randn(1, 1)
    targets = torch.randn(1, 1, 10).abs()

    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_bpnet_loss_batch_size_variation():
    """Test BPNetLoss with different batch sizes to ensure averaging is correct."""
    loss_fn = BPNetLoss(alpha=1.0, beta=1.0)  # profile averaged over batch and channels

    # Batch = 2
    # Case 1: Identical items -> Loss should be same as Batch=1
    logits = torch.randn(1, 1, 10)
    log_counts = torch.randn(1, 1)
    targets = torch.randn(1, 1, 10).abs()

    logits_2 = logits.repeat(2, 1, 1)
    log_counts_2 = log_counts.repeat(2, 1)
    targets_2 = targets.repeat(2, 1, 1)

    preds_1 = ProfileCountOutput(logits=logits, log_counts=log_counts)
    preds_2 = ProfileCountOutput(logits=logits_2, log_counts=log_counts_2)

    loss_1 = loss_fn(preds_1, targets)
    loss_2 = loss_fn(preds_2, targets_2)

    assert torch.isclose(loss_1, loss_2)
