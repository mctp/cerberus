import torch
import torch.nn as nn
import pytest
from cerberus.models.bpnet import BPNet, _ResidualBlock
from cerberus.loss import PoissonMultinomialLoss, BPNetLoss, DecoupledFlattenedPearsonCorrCoef, DecoupledMeanSquaredError, get_bpnet_metrics

def test_bpnet_residual_block_cropping():
    filters = 16
    kernel_size = 3
    dilation = 2 # Effective kernel width = (K-1)*D + 1 = (2)*2 + 1 = 5
    
    block = _ResidualBlock(filters, kernel_size, dilation)
    
    # Input Length: 20
    length = 20
    x = torch.randn(1, filters, length)
    
    # Conv output length: L_in - (K_eff - 1) = 20 - (5-1) = 20 - 4 = 16
    out = block(x)
    
    assert out.shape == (1, filters, 16)
    
    # Verify values logic manually roughly?
    # Center crop of input should be added.
    # Input center crop: 20 -> 16. Crop (20-16)/2 = 2 from each side.
    # x_cropped = x[..., 2:-2]
    # We can't easily verify the conv value, but we verify shape is correct and it runs.

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
    
    logits, log_counts = model(x)
    
    # Check shapes
    # Profile: (B, 2, 1000)
    assert logits.shape == (batch_size, 2, output_len)
    
    # Counts: Default predict_total_count=True -> (B, 1)
    assert log_counts.shape == (batch_size, 1)

def test_bpnet_counts_head_dimensionality_param():
    batch_size = 2
    input_len = 1200
    output_len = 1000
    filters = 16
    n_dilated_layers = 2
    
    # Case 1: predict_total_count=True (Default)
    model_total = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        output_channels=["pos", "neg"],
        predict_total_count=True
    )
    x = torch.randn(batch_size, 4, input_len)
    _, log_counts = model_total(x)
    assert log_counts.shape == (batch_size, 1)
    
    # Case 2: predict_total_count=False (Per-channel)
    model_per_channel = BPNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        output_channels=["pos", "neg"],
        predict_total_count=False
    )
    _, log_counts_pc = model_per_channel(x)
    assert log_counts_pc.shape == (batch_size, 2)

def test_poisson_multinomial_loss_bpnet_input():
    # Setup
    loss_fn = PoissonMultinomialLoss(count_weight=0.2)
    batch_size = 2
    channels = 2
    length = 100
    
    # Mock BPNet outputs
    logits = torch.randn(batch_size, channels, length, requires_grad=True)
    log_counts = torch.randn(batch_size, channels, requires_grad=True)
    predictions = (logits, log_counts)
    
    # Mock Targets (counts)
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    # Forward
    loss = loss_fn(predictions, targets)
    
    # Check scalar
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    
    # Backward
    loss.backward()
    assert logits.grad is not None
    assert log_counts.grad is not None

def test_poisson_multinomial_loss_counts_input():
    # Test support for raw counts input (as per prompt original spec)
    loss_fn = PoissonMultinomialLoss()
    batch_size = 2
    channels = 2
    length = 100
    
    # Mock Count predictions (must be non-negative)
    predictions = torch.rand(batch_size, channels, length) + 0.1
    predictions.requires_grad_(True)
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    loss = loss_fn(predictions, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    
    loss.backward()
    assert predictions.grad is not None

def test_poisson_multinomial_loss_bpnet_flattened():
    # Setup
    loss_fn = PoissonMultinomialLoss(count_weight=0.2, flatten_channels=True)
    batch_size = 2
    channels = 2
    length = 100
    
    # Mock BPNet outputs
    logits = torch.randn(batch_size, channels, length, requires_grad=True)
    log_counts = torch.randn(batch_size, channels, requires_grad=True)
    predictions = (logits, log_counts)
    
    targets = torch.randint(0, 10, (batch_size, channels, length)).float()
    
    # Forward
    loss = loss_fn(predictions, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    loss.backward()

def test_decoupled_pearson_metric():
    metric = DecoupledFlattenedPearsonCorrCoef(num_channels=2)
    batch_size = 2
    channels = 2
    length = 50
    
    # Mock BPNet outputs
    logits = torch.randn(batch_size, channels, length)
    log_counts = torch.randn(batch_size, channels)
    preds = (logits, log_counts)
    
    # Targets
    targets = torch.randn(batch_size, channels, length)
    
    # Update
    metric.update(preds, targets) # type: ignore
    result = metric.compute()
    
    assert result.dim() == 0
    
    # Verify manual calculation logic
    # Preds should be softmax(logits) * exp(log_counts)
    probs = nn.functional.softmax(logits, dim=-1)
    counts = torch.exp(log_counts).unsqueeze(-1)
    expected_preds = probs * counts
    
    # We can't easily verify the pearson value without re-implementing, 
    # but we checked no crash.

def test_get_bpnet_metrics():
    metrics = get_bpnet_metrics(num_channels=1)
    assert "pearson" in metrics
    assert isinstance(metrics["pearson"], DecoupledFlattenedPearsonCorrCoef)
    assert "mse" in metrics
    assert isinstance(metrics["mse"], DecoupledMeanSquaredError)

def test_decoupled_mse_metric():
    metric = DecoupledMeanSquaredError()
    batch_size = 2
    channels = 2
    length = 50
    
    # Mock BPNet outputs
    logits = torch.randn(batch_size, channels, length)
    log_counts = torch.randn(batch_size, channels)
    preds = (logits, log_counts)
    
    # Targets
    targets = torch.randn(batch_size, channels, length)
    
    # Update
    metric.update(preds, targets) # type: ignore
    result = metric.compute()
    
    assert result.dim() == 0
    
    # Verify manual calculation logic
    probs = nn.functional.softmax(logits, dim=-1)
    counts = torch.exp(log_counts).unsqueeze(-1)
    expected_preds = probs * counts
    
    expected_mse = nn.functional.mse_loss(expected_preds, targets)
    # Allow some floating point tolerance
    assert torch.allclose(result, expected_mse, atol=1e-5)

def test_bpnet_compilation():
    """Test that BPNet can be compiled with torch.compile without graph breaks."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    
    # Check if we are on a platform that supports default compilation
    # For now, we just try it and catch expected errors if platform related,
    # but the goal is to test the model code itself.
    
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
    
    # We use fullgraph=True to ensure no graph breaks occur in the model definition.
    # We use backend="eager" or "aot_eager" to test graph capture without necessarily codegen 
    # if dependencies like Triton are missing (common on Mac).
    # However, to be safe and standard, we'll try default first, or just rely on fullgraph check.
    # The most robust way to check "does not break compilation graph" is fullgraph=True.
    
    try:
        # Just compiling doesn't run it. We need to run it to trigger compilation.
        compiled_model = torch.compile(model, fullgraph=True)
        
        x = torch.randn(batch_size, 4, input_len)
        
        # First run triggers compilation
        with torch.no_grad():
            logits, log_counts = compiled_model(x)
            
        assert logits.shape == (batch_size, 1, output_len)
        assert log_counts.shape == (batch_size, 1)
        
    except Exception as e:
        # If failure is due to missing Triton or platform issues, we might want to warn but not fail
        # unless it's a GraphBreak.
        # But 'fullgraph=True' raises error on GraphBreak.
        # Errors like "BackendCompilerFailed" might happen if inductor fails.
        if "GraphBreak" in str(e) or "Unsupported" in str(e):
             pytest.fail(f"Graph break or unsupported operation detected: {e}")
        else:
             # On some platforms (like Mac without Triton), compile might fail at codegen stage
             # but we want to know if the graph capture was successful.
             # If it got past capture and failed at backend, the model structure is likely fine.
             print(f"Compilation failed likely due to backend issues: {e}")
             # We can try with backend='aot_eager' to verify graph capture only
             try:
                 compiled_model_eager = torch.compile(model, fullgraph=True, backend="aot_eager")
                 with torch.no_grad():
                     compiled_model_eager(x)
             except Exception as e2:
                 pytest.fail(f"Model failed to compile even with aot_eager (Graph Capture issue): {e2}")

def test_bpnet_loss_integration():
    """Test that BPNet output is compatible with BPNetLoss."""
    model = BPNet(
        input_len=1000, 
        output_len=500,
        filters=8,
        n_dilated_layers=1,
        output_channels=["signal"]
    )
    loss_fn = BPNetLoss()
    
    x = torch.randn(2, 4, 1000)
    logits, log_counts = model(x)
    
    # Targets: Counts (Batch, Channels, Length)
    # BPNetLoss expects raw counts in targets (unless implicit_log_targets=True)
    targets = torch.randint(0, 10, (2, 1, 500)).float()
    
    loss = loss_fn((logits, log_counts), targets)
    assert not torch.isnan(loss)
    assert loss.dim() == 0

    # Also test with multiple channels and per-channel counts
    model_multi = BPNet(
        input_len=1000,
        output_len=500,
        filters=8,
        n_dilated_layers=1,
        output_channels=["plus", "minus"],
        predict_total_count=False # per-channel counts
    )
    logits_m, log_counts_m = model_multi(x)
    targets_m = torch.randint(0, 10, (2, 2, 500)).float()
    
    loss_m = loss_fn((logits_m, log_counts_m), targets_m)
    assert not torch.isnan(loss_m)
