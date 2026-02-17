"""
Verify gradients and training for BPNet using synthetic data.

Designed to complete in < 0.5s while testing numerical accuracy.

Tiny model dimensions (used for numerical accuracy and training tests):
  input_len=200, n_dilated_layers=1, filters=4, profile_kernel_size=75
  -> iconv (k=21):            200 - 20  = 180
  -> dilated block (k=3, d=2): 180 - 4   = 176
  -> profile conv (k=75):      176 - 74  = 102  -> crop to output_len=100

Default model dimensions (used for shape / gradient-flow smoke tests):
  input_len=2114, n_dilated_layers=8, filters=64, profile_kernel_size=75
  -> iconv (k=21):            2114 - 20   = 2094
  -> 8 dilated blocks (total): 2094 - 1020 = 1074
  -> profile conv (k=75):      1074 - 74   = 1000  (exact, no cropping)
"""

import pytest
import torch
import torch.nn.functional as F

from cerberus.models.bpnet import BPNet, BPNetLoss


SEED = 0
BATCH_SIZE = 2
INPUT_LEN = 200
OUTPUT_LEN = 100
FILTERS = 4
N_DILATED_LAYERS = 1
PROFILE_KERNEL = 75


@pytest.fixture
def model():
    torch.manual_seed(SEED)
    return BPNet(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        filters=FILTERS,
        n_dilated_layers=N_DILATED_LAYERS,
        conv_kernel_size=21,
        dil_kernel_size=3,
        profile_kernel_size=PROFILE_KERNEL,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"],
        predict_total_count=True,
    )


@pytest.fixture
def synthetic_batch():
    torch.manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, 4, INPUT_LEN)
    # Non-zero integer targets so multinomial NLL and count loss are both non-trivial
    targets = torch.randint(1, 10, (BATCH_SIZE, 1, OUTPUT_LEN)).float()
    return x, targets


# ---------------------------------------------------------------------------
# 1. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow_all_params(model, synthetic_batch):
    """Every learnable parameter receives a finite, non-zero gradient."""
    x, targets = synthetic_batch
    loss_fn = BPNetLoss()

    out = model(x)
    loss = loss_fn(out, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter '{name}'"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in '{name}'"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for '{name}'"


def test_gradient_wrt_input(model, synthetic_batch):
    """Gradient flows back through the model to the input tensor."""
    x, targets = synthetic_batch
    x = x.requires_grad_(True)
    loss_fn = BPNetLoss()

    out = model(x)
    loss = loss_fn(out, targets)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 2. Numerical accuracy — loss formula
# ---------------------------------------------------------------------------

def test_loss_profile_component_numerical_accuracy(model, synthetic_batch):
    """Profile loss matches manually computed multinomial NLL."""
    x, targets = synthetic_batch
    # BPNetLoss: average_channels=True, flatten_channels=False
    loss_fn = BPNetLoss(alpha=0.0, beta=1.0)  # profile only

    out = model(x)
    actual = loss_fn(out, targets)

    logits = out.logits  # (B, C, L)
    log_probs = F.log_softmax(logits, dim=-1)
    profile_counts = targets.sum(dim=-1)                   # (B, C)
    log_fact_sum = torch.lgamma(profile_counts + 1)        # (B, C)
    log_prod_fact = torch.lgamma(targets + 1).sum(dim=-1)  # (B, C)
    log_prod_exp = (targets * log_probs).sum(dim=-1)       # (B, C)
    per_channel = -log_fact_sum + log_prod_fact - log_prod_exp  # (B, C)
    expected = per_channel.mean()

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Profile loss mismatch: got {actual.item():.7f}, expected {expected.item():.7f}"
    )


def test_loss_count_component_numerical_accuracy(model, synthetic_batch):
    """Count loss matches manually computed MSE on log1p(global counts)."""
    x, targets = synthetic_batch
    loss_fn = BPNetLoss(alpha=1.0, beta=0.0)  # count only

    out = model(x)
    actual = loss_fn(out, targets)

    target_global = targets.sum(dim=(1, 2))           # (B,)
    target_log = torch.log1p(target_global)            # (B,)
    pred_log = out.log_counts.flatten()                # (B,) — log_counts is (B,1)
    expected = F.mse_loss(pred_log, target_log)

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Count loss mismatch: got {actual.item():.7f}, expected {expected.item():.7f}"
    )


def test_loss_combined_is_sum_of_components(model, synthetic_batch):
    """Combined BPNetLoss equals profile loss + count loss (with alpha=beta=1)."""
    x, targets = synthetic_batch
    loss_fn = BPNetLoss(alpha=1.0, beta=1.0)
    loss_profile = BPNetLoss(alpha=0.0, beta=1.0)
    loss_count = BPNetLoss(alpha=1.0, beta=0.0)

    out = model(x)
    combined = loss_fn(out, targets)
    expected = loss_profile(out, targets) + loss_count(out, targets)

    assert torch.allclose(combined, expected, atol=1e-5), (
        f"Combined loss {combined.item():.7f} != sum of parts {expected.item():.7f}"
    )


# ---------------------------------------------------------------------------
# 3. Finite-difference gradient check
# ---------------------------------------------------------------------------

def test_finite_difference_gradient(model, synthetic_batch):
    """
    Analytic gradient of iconv.weight[0,0,0] matches central-difference FD
    approximation to within 1% relative error.
    """
    x, targets = synthetic_batch
    loss_fn = BPNetLoss()
    eps = 1e-3

    param = model.iconv.weight  # (filters, 4, k)

    with torch.no_grad():
        orig = param[0, 0, 0].item()
        param[0, 0, 0] = orig + eps
        loss_plus = loss_fn(model(x), targets).item()
        param[0, 0, 0] = orig - eps
        loss_minus = loss_fn(model(x), targets).item()
        param[0, 0, 0] = orig  # restore

    fd_grad = (loss_plus - loss_minus) / (2.0 * eps)

    model.zero_grad()
    loss = loss_fn(model(x), targets)
    loss.backward()
    analytic_grad = param.grad[0, 0, 0].item()

    rel_err = abs(fd_grad - analytic_grad) / (abs(fd_grad) + 1e-8)
    assert rel_err < 0.01, (
        f"FD gradient {fd_grad:.6f} vs analytic {analytic_grad:.6f} "
        f"(relative error {rel_err:.4f} >= 0.01)"
    )


# ---------------------------------------------------------------------------
# 4. Training convergence
# ---------------------------------------------------------------------------

def test_training_loss_decreases(model, synthetic_batch):
    """Loss decreases over five Adam steps on a fixed synthetic batch."""
    x, targets = synthetic_batch
    loss_fn = BPNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease after 5 steps: "
        f"{losses[0]:.5f} -> {losses[-1]:.5f}  (all: {losses})"
    )


def test_training_gradients_stable_across_steps(model, synthetic_batch):
    """Parameter gradients remain finite throughout multiple training steps."""
    x, targets = synthetic_batch
    loss_fn = BPNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, targets)
        loss.backward()
        for name, param in model.named_parameters():
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient in '{name}' at step {step}"
            )
        optimizer.step()


# ---------------------------------------------------------------------------
# 5. Default BPNet size smoke tests
#    batch_size=1 to keep wall-time well under budget; no_grad where possible.
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model():
    torch.manual_seed(SEED)
    return BPNet()  # all defaults: input_len=2114, output_len=1000, filters=64


@pytest.fixture
def default_batch():
    torch.manual_seed(SEED)
    x = torch.randn(1, 4, 2114)
    targets = torch.randint(1, 10, (1, 1, 1000)).float()
    return x, targets


def test_default_bpnet_output_shapes(default_model, default_batch):
    """Default BPNet produces tensors of the documented output shapes."""
    x, _ = default_batch
    with torch.no_grad():
        out = default_model(x)
    assert out.logits.shape == (1, 1, 1000), f"Unexpected logits shape: {out.logits.shape}"
    assert out.log_counts.shape == (1, 1), f"Unexpected log_counts shape: {out.log_counts.shape}"


def test_default_bpnet_loss_finite(default_model, default_batch):
    """BPNetLoss on default model output is a finite scalar."""
    x, targets = default_batch
    loss_fn = BPNetLoss()
    with torch.no_grad():
        out = default_model(x)
    loss = loss_fn(out, targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_default_bpnet_gradient_flow(default_model, default_batch):
    """All parameters of the default BPNet receive finite gradients."""
    x, targets = default_batch
    loss_fn = BPNetLoss()

    out = default_model(x)
    loss = loss_fn(out, targets)
    loss.backward()

    for name, param in default_model.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in '{name}'"


def test_default_bpnet_multichannel_output_shapes(default_batch):
    """Default BPNet with multiple output channels produces correct shapes."""
    torch.manual_seed(SEED)
    model = BPNet(output_channels=["plus", "minus"])
    x, _ = default_batch
    with torch.no_grad():
        out = model(x)
    # predict_total_count=True by default → single scalar count regardless of channels
    assert out.logits.shape == (1, 2, 1000), f"Unexpected logits shape: {out.logits.shape}"
    assert out.log_counts.shape == (1, 1), f"Unexpected log_counts shape: {out.log_counts.shape}"
