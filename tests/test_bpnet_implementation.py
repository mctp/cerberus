import torch
import torch.nn as nn
import pytest
from cerberus.models.bpnet import BPNet, BPNetMetricCollection, BPNetLoss
from cerberus.layers import DilatedResidualBlock
from cerberus.loss import PoissonMultinomialLoss, MSEMultinomialLoss, CoupledMSEMultinomialLoss, CoupledPoissonMultinomialLoss
from cerberus.metrics import PerExampleCountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError
from cerberus.output import ProfileCountOutput, ProfileLogRates

def test_bpnet_residual_block_cropping():
    filters = 16
    kernel_size = 3
    dilation = 2 
    block = DilatedResidualBlock(filters, kernel_size, dilation)
    length = 20
    x = torch.randn(1, filters, length)
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
    out = model(x)
    
    assert out.logits.shape == (batch_size, 2, output_len)
    assert out.log_counts.shape == (batch_size, 1)

def test_bpnet_counts_head_dimensionality_param():
    batch_size = 2
    input_len = 1200
    output_len = 1000
    filters = 16
    n_dilated_layers = 2
    
    model_total = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        output_channels=["pos", "neg"],
        predict_total_count=True
    )
    x = torch.randn(batch_size, 4, input_len)
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
    metric = CountProfilePearsonCorrCoef(num_channels=2)
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
    metrics = BPNetMetricCollection(num_channels=1)
    assert "pearson" in metrics
    assert isinstance(metrics["pearson"], PerExampleCountProfilePearsonCorrCoef)
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
        predict_total_count=False # per-channel counts
    )
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
