import torch
import torch.nn as nn
import pytest
from cerberus.models.geminet import GemiNet, PGCBlock, GemiNetMetricCollection
from cerberus.models.bpnet import BPNetLoss
from cerberus.output import ProfileCountOutput

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_pgc_block_shape():
    dim = 16
    expansion = 2
    block = PGCBlock(dim, kernel_size=3, dilation=1, expansion=expansion)
    length = 20
    x = torch.randn(1, dim, length)
    out = block(x)
    assert out.shape == (1, dim, length) # Should preserve shape due to 'same' padding

def test_geminet_instantiation_defaults():
    batch_size = 2
    input_len = 1200
    output_len = 1000
    filters = 16
    n_dilated_layers = 2
    
    model = GemiNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=n_dilated_layers,
        input_channels=["A", "C", "G", "T"],
        output_channels=["pos", "neg"]
    )
    x = torch.randn(batch_size, 4, input_len)
    out = model(x)
    
    assert isinstance(out, ProfileCountOutput)
    assert out.logits.shape == (batch_size, 2, output_len)
    assert out.log_counts.shape == (batch_size, 1) # Default predict_total_count=True

def test_geminet_flexible_lengths():
    """Verify GemiNet handles input_len >= output_len correctly"""
    filters = 8
    model = GemiNet(
        input_len=2000,
        output_len=1000,
        filters=filters,
        n_dilated_layers=2
    )
    x = torch.randn(2, 4, 2000)
    out = model(x)
    assert out.logits.shape == (2, 1, 1000)

    # Test equal length
    model_eq = GemiNet(
        input_len=1000,
        output_len=1000,
        filters=filters,
        n_dilated_layers=2
    )
    x_eq = torch.randn(2, 4, 1000)
    out_eq = model_eq(x_eq)
    assert out_eq.logits.shape == (2, 1, 1000)

def test_geminet_invalid_lengths():
    """Verify GemiNet raises error if input_len < output_len (implicitly via forward)"""
    # Note: Constructor just stores lengths. Error happens during forward if cropping fails.
    # Actually, constructor stores output_len. If input passed is smaller, it fails.
    
    model = GemiNet(
        input_len=500, # Expected input
        output_len=1000, # Expected output
        filters=8,
        n_dilated_layers=2
    )
    
    # If we pass input matching input_len, it will error because current_len (500) < target_len (1000)
    x = torch.randn(1, 4, 500)
    with pytest.raises(ValueError, match="Output length.*smaller than requested"):
        model(x)

def test_geminet_expansion_param():
    """Verify expansion parameter affects parameter count"""
    filters = 16
    
    model_e1 = GemiNet(input_len=100, output_len=100, filters=filters, n_dilated_layers=1, expansion=1)
    params_e1 = count_params(model_e1)
    
    model_e2 = GemiNet(input_len=100, output_len=100, filters=filters, n_dilated_layers=1, expansion=2)
    params_e2 = count_params(model_e2)
    
    assert params_e2 > params_e1
    
    # Rough check: PGC block params
    # E=1: 16->32 Proj, 32 Conv, 32->16 Proj.
    # E=2: 16->64 Proj, 64 Conv, 64->16 Proj.
    # Projections dominate.
    
    print(f"Params E=1: {params_e1}")
    print(f"Params E=2: {params_e2}")

def test_geminet_compilation():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    
    batch_size = 2
    input_len = 500
    output_len = 400
    filters = 8
    
    model = GemiNet(
        input_len=input_len,
        output_len=output_len,
        filters=filters,
        n_dilated_layers=2
    )
    model.eval()
    
    x = torch.randn(batch_size, 4, input_len)
    try:
        compiled_model = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            out = compiled_model(x)
        assert out.logits.shape == (batch_size, 1, output_len)
    except Exception as e:
        if "GraphBreak" in str(e) or "Unsupported" in str(e):
             pytest.fail(f"Graph break or unsupported operation detected: {e}")
        else:
             print(f"Compilation failed likely due to backend issues: {e}")
             # Attempt eager backend
             try:
                 compiled_model_eager = torch.compile(model, fullgraph=True, backend="aot_eager")
                 with torch.no_grad():
                     compiled_model_eager(x)
             except Exception as e2:
                 pytest.fail(f"Model failed to compile even with aot_eager: {e2}")

def test_geminet_loss_integration():
    model = GemiNet(
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
    
    loss.backward()
    assert model.profile_conv.weight.grad is not None

def test_geminet_metrics():
    metrics = GemiNetMetricCollection()
    
    # Mock output
    logits = torch.randn(2, 2, 100)
    log_counts = torch.randn(2, 1)
    preds = ProfileCountOutput(logits=logits, log_counts=log_counts)
    
    # Mock targets (counts)
    targets = torch.abs(torch.randn(2, 2, 100))
    
    # Update
    metrics.update(preds, targets)
    result = metrics.compute()
    
    assert "pearson" in result
    assert "mse_profile" in result
    assert "mse_log_counts" in result
    assert result["pearson"].ndim == 0
