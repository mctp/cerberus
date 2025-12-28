import torch
import torch.nn as nn
import pytest
from cerberus.models.bpnet import BPNet, _ResidualBlock
from cerberus.loss import PoissonMultinomialLoss, DecoupledFlattenedPearsonCorrCoef, get_bpnet_metrics

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
    metric.update(preds, targets)
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
