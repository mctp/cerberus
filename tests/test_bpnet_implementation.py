import torch
import torch.nn as nn
import torch.nn.utils.parametrize as nn_parametrize
import pytest
from cerberus.models.bpnet import BPNet, BPNet1024, BPNetMetricCollection, BPNetLoss
from cerberus.layers import DilatedResidualBlock
from cerberus.loss import PoissonMultinomialLoss, MSEMultinomialLoss, CoupledMSEMultinomialLoss, CoupledPoissonMultinomialLoss
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError
from cerberus.output import ProfileCountOutput, ProfileLogRates

def test_bpnet_xavier_init():
    """BPNet must initialize conv/linear weights with Xavier uniform and biases to zero.

    This matches the TensorFlow/Keras default used by the original BPNet and
    chrombpnet-pytorch. The test checks both the statistical properties of the
    weight distribution and that all biases are exactly zero.
    """
    model = BPNet(
        input_len=1000,
        output_len=800,
        filters=64,
        n_dilated_layers=3,
    )
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            # Biases must be exactly zero
            if m.bias is not None:
                assert m.bias.data.eq(0).all(), f"{name}.bias not zero-initialized"
            # Weights must be non-trivially initialized (not all zeros or all same)
            assert m.weight.data.std() > 0, f"{name}.weight has zero variance after init"
            # Xavier uniform bound: sqrt(6 / (fan_in + fan_out))
            # Weights must not exceed a generous multiple of that bound
            fan_in = m.weight.data.shape[1] * (m.weight.data.shape[2] if m.weight.data.dim() == 3 else 1)
            fan_out = m.weight.data.shape[0] * (m.weight.data.shape[2] if m.weight.data.dim() == 3 else 1)
            bound = (6 / (fan_in + fan_out)) ** 0.5
            assert m.weight.data.abs().max().item() <= bound + 1e-5, (
                f"{name}.weight exceeds Xavier uniform bound {bound:.4f}"
            )


def test_bpnet_residual_block_cropping():
    filters = 16
    kernel_size = 3
    dilation = 2
    block = DilatedResidualBlock(filters, kernel_size, dilation)
    length = 20
    x = torch.randn(1, filters, length)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (1, filters, 16)

def test_bpnet_architecture_defaults():
    batch_size = 2
    input_len = 1200
    output_len = 1000
    filters = 16
    n_dilated_layers = 2

    model = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        input_channels=["A", "C", "G", "T"],
        output_channels=["pos", "neg"]
    )
    x = torch.randn(batch_size, 4, input_len)
    with torch.no_grad():
        out = model(x)

    assert out.logits.shape == (batch_size, 2, output_len)
    assert out.log_counts.shape == (batch_size, 1)

def test_bpnet_counts_head_dimensionality_param():
    batch_size = 2
    input_len = 1200
    output_len = 1000
    filters = 16
    n_dilated_layers = 2

    x = torch.randn(batch_size, 4, input_len)

    model_total = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        output_channels=["pos", "neg"],
        predict_total_count=True
    )
    with torch.no_grad():
        out = model_total(x)
    assert out.log_counts.shape == (batch_size, 1)

    model_per_channel = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        output_channels=["pos", "neg"],
        predict_total_count=False
    )
    with torch.no_grad():
        out_pc = model_per_channel(x)
    assert out_pc.log_counts.shape == (batch_size, 2)

def test_poisson_multinomial_loss_bpnet_input():
    loss_fn = PoissonMultinomialLoss(count_weight=0.2)
    batch_size = 2
    channels = 2
    length = 100
    
    logits = torch.randn(batch_size, channels, length, requires_grad=True)
    # Global loss expects (B, 1) log_counts
    log_counts = torch.randn(batch_size, 1, requires_grad=True)
    predictions = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    loss = loss_fn(predictions, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    
    loss.backward()
    assert logits.grad is not None
    assert log_counts.grad is not None

def test_poisson_multinomial_loss_bpnet_flattened():
    loss_fn = PoissonMultinomialLoss(count_weight=0.2, flatten_channels=True)
    batch_size = 2
    channels = 2
    length = 100
    
    logits = torch.randn(batch_size, channels, length, requires_grad=True)
    log_counts = torch.randn(batch_size, 1, requires_grad=True)
    predictions = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    loss = loss_fn(predictions, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    loss.backward()

def test_decoupled_pearson_metric():
    metric = CountProfilePearsonCorrCoef()
    batch_size = 2
    channels = 2
    length = 50
    
    # Use deterministic inputs with sufficient variance
    logits = torch.randn(batch_size, channels, length) * 10.0
    log_counts = torch.abs(torch.randn(batch_size, channels)) + 1.0 # Positive log counts
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    targets = torch.randn(batch_size, channels, length) * 10.0
    
    metric.update(preds, targets)
    result = metric.compute()
    assert result.dim() == 0

def test_bpnet_metric_collection():
    metrics = BPNetMetricCollection()
    assert "pearson" in metrics
    assert isinstance(metrics["pearson"], CountProfilePearsonCorrCoef)
    assert "mse_profile" in metrics
    assert isinstance(metrics["mse_profile"], CountProfileMeanSquaredError)
    assert "mse_log_counts" in metrics
    assert isinstance(metrics["mse_log_counts"], LogCountsMeanSquaredError)

def test_decoupled_mse_metric():
    metric = CountProfileMeanSquaredError()
    batch_size = 2
    channels = 2
    length = 50
    
    logits = torch.randn(batch_size, channels, length)
    log_counts = torch.randn(batch_size, channels)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    targets = torch.randn(batch_size, channels, length)
    
    metric.update(preds, targets)
    result = metric.compute()
    assert result.dim() == 0
    
    probs = nn.functional.softmax(logits, dim=-1)
    counts = torch.expm1(log_counts.float()).clamp_min(0.0).unsqueeze(-1)
    expected_preds = probs * counts
    expected_mse = nn.functional.mse_loss(expected_preds, targets)
    assert torch.allclose(result, expected_mse, atol=1e-5)

def test_bpnet_compilation():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    
    batch_size = 2
    input_len = 1000
    output_len = 800
    filters = 8
    n_dilated_layers = 2
    
    model = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        input_channels=["A", "C", "G", "T"],
        output_channels=["signal"]
    )
    model.eval()
    
    x = torch.randn(batch_size, 4, input_len)
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            out = compiled_model(x)
        assert out.logits.shape == (batch_size, 1, output_len)
        assert out.log_counts.shape == (batch_size, 1)
    except Exception as e:
        if "GraphBreak" in str(e) or "Unsupported" in str(e):
             pytest.fail(f"Graph break or unsupported operation detected: {e}")
        else:
             print(f"Compilation failed likely due to backend issues: {e}")
             try:
                 compiled_model_eager = torch.compile(model, fullgraph=True, backend="aot_eager")
                 with torch.no_grad():
                     compiled_model_eager(x)
             except Exception as e2:
                 pytest.fail(f"Model failed to compile even with aot_eager (Graph Capture issue): {e2}")

def test_bpnet_loss_integration():
    model = BPNet(
        input_len=1000,
        output_len=500,
        filters=8,
        n_dilated_layers=1,
        output_channels=["signal"]
    )
    loss_fn = BPNetLoss()
    x = torch.randn(2, 4, 1000)
    with torch.no_grad():
        out = model(x)
    targets = torch.randint(0, 10, (2, 1, 500)).float()

    loss = loss_fn(out, targets)
    assert not torch.isnan(loss)
    assert loss.dim() == 0

    # Test Coupled Loss with ProfileLogits (simulated counts)
    loss_coupled = CoupledMSEMultinomialLoss()
    model_multi = BPNet(
        input_len=1000,
        output_len=500,
        filters=8,
        n_dilated_layers=1,
        output_channels=["plus", "minus"],
        predict_total_count=False  # per-channel counts
    )
    with torch.no_grad():
        out_m = model_multi(x)
    targets_m = torch.randint(0, 10, (2, 2, 500)).float()

    # Treat BPNet logits as log-rates for the purpose of testing coupled loss mechanics
    out_profile_only = ProfileLogRates(log_rates=out_m.logits)
    loss_c = loss_coupled(out_profile_only, targets_m)
    assert not torch.isnan(loss_c)

    with pytest.raises(TypeError, match="does not accept ProfileCountOutput"):
        loss_coupled(out_m, targets_m)

def test_poisson_multinomial_loss_integration():
    model = BPNet(
        input_len=1000, 
        output_len=500,
        filters=8,
        n_dilated_layers=1,
        output_channels=["signal"]
    )
    loss_fn = PoissonMultinomialLoss()
    assert loss_fn.count_loss_fn.log_input is True
    x = torch.zeros(2, 4, 1000)
    x[:, :, 0] = 1.0 
    out = model(x)
    targets = torch.randint(0, 10, (2, 1, 500)).float()
    
    loss = loss_fn(out, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    loss.backward()
    assert model.count_dense.weight.grad is not None

def test_bpnet_coupled_equivalence():
    """
    Verify that MSEMultinomialLoss and CoupledMSEMultinomialLoss are mathematically equivalent
    when provided with corresponding inputs (where log_counts are derived from logits).
    """
    batch_size = 2
    channels = 2
    length = 100
    
    logits = torch.randn(batch_size, channels, length)
    
    # Derived log_counts (Global Sum)
    # This matches CoupledMSEMultinomialLoss logic: flatten(1) -> logsumexp(-1)
    logits_flat = logits.flatten(start_dim=1)
    log_counts_global = torch.logsumexp(logits_flat, dim=-1) # (B,)
    log_counts_input = log_counts_global.unsqueeze(1) # (B, 1) for MSEMultinomialLoss
    
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    # 1. MSEMultinomialLoss
    loss_fn_std = MSEMultinomialLoss()
    out_std = ProfileCountOutput(logits=logits, log_counts=log_counts_input)
    loss_std = loss_fn_std(out_std, targets)
    
    # 2. CoupledMSEMultinomialLoss
    loss_fn_coupled = CoupledMSEMultinomialLoss()
    out_coupled = ProfileLogRates(log_rates=logits)
    loss_coupled = loss_fn_coupled(out_coupled, targets)
    
    # Tolerances might need adjustment if logsumexp precision varies
    assert torch.isclose(loss_std, loss_coupled, atol=1e-6)

def test_poisson_coupled_equivalence():
    """
    Verify that PoissonMultinomialLoss and CoupledPoissonMultinomialLoss are equivalent.
    """
    batch_size = 2
    channels = 2
    length = 100
    
    logits = torch.randn(batch_size, channels, length)
    
    # Global log counts
    logits_flat = logits.flatten(start_dim=1)
    log_counts_global = torch.logsumexp(logits_flat, dim=-1)
    log_counts_input = log_counts_global.unsqueeze(1)
    
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    # 1. Standard
    loss_fn_std = PoissonMultinomialLoss()
    out_std = ProfileCountOutput(logits=logits, log_counts=log_counts_input)
    loss_std = loss_fn_std(out_std, targets)
    
    # 2. Coupled
    loss_fn_coupled = CoupledPoissonMultinomialLoss()
    out_coupled = ProfileLogRates(log_rates=logits)
    loss_coupled = loss_fn_coupled(out_coupled, targets)

    assert torch.isclose(loss_std, loss_coupled, atol=1e-6)


# ---------------------------------------------------------------------------
# Stable training mode: DilatedResidualBlock with activation and weight_norm
# ---------------------------------------------------------------------------

def test_dilated_residual_block_gelu():
    """DilatedResidualBlock with activation='gelu' produces the same output shape as relu."""
    filters, kernel_size, dilation, length = 16, 3, 2, 20
    block_relu = DilatedResidualBlock(filters, kernel_size, dilation, activation="relu")
    block_gelu = DilatedResidualBlock(filters, kernel_size, dilation, activation="gelu")
    x = torch.randn(1, filters, length)
    with torch.no_grad():
        out_relu = block_relu(x)
        out_gelu = block_gelu(x)
    assert out_relu.shape == out_gelu.shape


def test_dilated_residual_block_weight_norm():
    """DilatedResidualBlock with weight_norm=True parametrizes the conv weight."""
    filters, kernel_size, dilation = 16, 3, 2
    block = DilatedResidualBlock(filters, kernel_size, dilation, weight_norm=True)
    # parametrizations.weight_norm registers 'weight' as a parametrized attribute
    assert nn_parametrize.is_parametrized(block.conv, 'weight'), \
        "weight is not parametrized on weight-normed conv"
    # .weight should be a computed property, not a leaf parameter
    assert 'weight' not in dict(block.conv.named_parameters()), \
        "weight should not be a leaf parameter when weight_norm is applied"
    # Forward pass still works
    x = torch.randn(1, filters, 20)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (1, filters, 16)


def test_dilated_residual_block_invalid_activation():
    """DilatedResidualBlock raises ValueError for unknown activation strings."""
    with pytest.raises(ValueError, match="unsupported activation"):
        DilatedResidualBlock(16, 3, 2, activation="swish")


def test_bpnet_stable_output_shapes():
    """BPNet with weight_norm=True and activation='gelu' produces the same output shapes as baseline.

    Dimension notes: with n_dilated_layers=3 (dilations 2,4,8), conv_kernel=21, profile_kernel=75:
      reduction = 20 (iconv) + 4+8+16 (dilated tower) + 74 (profile_conv) = 122
      output_len must be <= input_len - 122.  input_len=600 -> max output=478, use 350.
    """
    batch_size = 2
    input_len = 600
    output_len = 350
    filters = 8
    n_dilated_layers = 3

    model_base = BPNet(
        input_len=input_len, output_len=output_len, filters=filters,
        n_dilated_layers=n_dilated_layers,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="relu", weight_norm=False,
    )
    model_stable = BPNet(
        input_len=input_len, output_len=output_len, filters=filters,
        n_dilated_layers=n_dilated_layers,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    x = torch.randn(batch_size, 4, input_len)
    with torch.no_grad():
        out_base = model_base(x)
        out_stable = model_stable(x)
    assert out_base.logits.shape == out_stable.logits.shape
    assert out_base.log_counts.shape == out_stable.log_counts.shape


def test_bpnet_stable_xavier_init():
    """With weight_norm=True, the effective weight is Xavier-distributed and biases are zero.

    Weight norm is applied after _tf_style_reinit, so right_inverse decomposes the
    Xavier-initialized weight into weight_g and weight_v.  We verify this by checking
    that the computed m.weight has non-trivial variance and biases are zero.
    """
    model = BPNet(
        input_len=600, output_len=350, filters=16, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if nn_parametrize.is_parametrized(m, 'weight'):
                # Effective weight is computed (Xavier-derived via right_inverse)
                assert m.weight.data.std() > 0, f"{name}.weight has zero variance after wn init"
            elif m.weight is not None:
                assert m.weight.data.std() > 0, f"{name}.weight has zero variance after Xavier init"
            if m.bias is not None:
                assert m.bias.data.eq(0).all(), f"{name}.bias not zero-initialized"


def test_bpnet_weight_norm_init_matches_xavier():
    """The effective weight of a weight-normed BPNet matches the Xavier bound.

    BPNet builds layers plain, runs _tf_style_reinit (Xavier), then applies weight_norm.
    PyTorch's right_inverse decomposes: weight_g = ||w||, weight_v = w / ||w||, so the
    computed weight w_new = weight_g * (weight_v / ||weight_v||) == w_original.
    Therefore the effective weight after weight_norm must be identical to the plain
    Xavier-initialised weight — not a fresh default initialisation.
    """
    torch.manual_seed(42)

    # Build a weight-normed model
    model_wn = BPNet(
        input_len=600, output_len=350, filters=16, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    # Build a plain model with the same seed to get the same Xavier weights
    torch.manual_seed(42)
    model_plain = BPNet(
        input_len=600, output_len=350, filters=16, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=False,
    )

    # The effective weight of each parametrized layer must match the plain Xavier weight
    wn_mods = {n: m for n, m in model_wn.named_modules() if isinstance(m, (nn.Conv1d, nn.Linear))}
    plain_mods = {n: m for n, m in model_plain.named_modules() if isinstance(m, (nn.Conv1d, nn.Linear))}

    for name in wn_mods:
        m_wn = wn_mods[name]
        m_plain = plain_mods[name]
        if nn_parametrize.is_parametrized(m_wn, 'weight') and m_plain.weight is not None:
            assert torch.allclose(m_wn.weight.data, m_plain.weight.data, atol=1e-6), (
                f"{name}: effective weight after weight_norm differs from plain Xavier init"
            )

    # Also verify the Xavier bound: all effective weights must satisfy |w| <= bound
    for name, m in model_wn.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and nn_parametrize.is_parametrized(m, 'weight'):
            fan_in = m.weight.data.shape[1] * (m.weight.data.shape[2] if m.weight.data.dim() == 3 else 1)
            fan_out = m.weight.data.shape[0] * (m.weight.data.shape[2] if m.weight.data.dim() == 3 else 1)
            bound = (6 / (fan_in + fan_out)) ** 0.5
            assert m.weight.data.abs().max().item() <= bound + 1e-5, (
                f"{name}: effective weight exceeds Xavier bound {bound:.4f}"
            )


def test_bpnet_stable_gradient_wrt_input():
    """Input gradients flow through stable BPNet (required for captum DeepLIFT/SHAP)."""
    model = BPNet(
        input_len=600, output_len=350, filters=8, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    model.eval()
    x = torch.randn(2, 4, 600, requires_grad=True)
    out = model(x)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()
    assert x.grad is not None, "No gradient w.r.t. input"
    assert x.grad.isfinite().all(), "Input gradients contain non-finite values"
    assert x.grad.abs().sum() > 0, "Input gradients are all zero"


def test_bpnet_stable_parameter_gradients():
    """All parameters in stable BPNet receive finite non-zero gradients."""
    model = BPNet(
        input_len=600, output_len=350, filters=8, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    x = torch.randn(2, 4, 600)
    out = model(x)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.isfinite().all(), f"Non-finite gradient for {name}"


def test_bpnet1024_stable_passthrough():
    """BPNet1024 correctly passes activation and weight_norm to the parent BPNet."""
    model = BPNet1024(
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (2, 1, 1024)
    assert out.log_counts.shape == (2, 1)
    # Confirm weight_norm was applied to iconv
    assert nn_parametrize.is_parametrized(model.iconv, 'weight'), \
        "weight is not parametrized on BPNet1024 iconv"


def test_bpnet_stable_deeplift():
    """Stable BPNet produces finite attributions with captum DeepLift (skipped if captum absent)."""
    try:
        from captum.attr import DeepLift  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("captum not installed")

    model = BPNet(
        input_len=600, output_len=350, filters=8, n_dilated_layers=3,
        input_channels=["A", "C", "G", "T"], output_channels=["signal"],
        activation="gelu", weight_norm=True,
    )
    model.eval()

    def profile_metric(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return out.logits.sum(dim=(-1, -2))  # (B,)

    dl = DeepLift(profile_metric)
    x = torch.randn(2, 4, 600)
    baseline = torch.zeros_like(x)
    attributions = dl.attribute(x, baseline)
    assert attributions.shape == x.shape
    assert attributions.isfinite().all(), "DeepLift attributions contain non-finite values"
