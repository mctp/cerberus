
import pytest
import torch
import torch.nn as nn
from cerberus.module import instantiate_model
from cerberus.config import ModelConfig, DataConfig
from unittest.mock import patch, MagicMock

# Mock import_class to return a DummyModel
class DummyModel(nn.Module):
    def __init__(self, input_len, output_len, output_bin_size):
        super().__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        return self.linear(x)

def test_instantiate_model_compile():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    model_config = {
        "name": "test",
        "model_cls": "dummy.DummyModel",
        "loss_cls": "dummy.Loss",
        "metrics_cls": "dummy.Metrics",
        "model_args": {},
        "loss_args": {},
        "metrics_args": {}
    }
    
    data_config = {
        "input_len": 10,
        "output_len": 10,
        "output_bin_size": 1,
        "inputs": {},
        "targets": {},
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "count_pseudocount": 1.0,
    }

    with patch("cerberus.module.import_class") as mock_import:
        mock_import.return_value = DummyModel
        
        # Test compile=True
        # We need to ensure torch.compile doesn't fail. 
        # On some systems/CI environments without GPU or triton, torch.compile might fail or warn.
        # But here we just want to ensure our code calling it is correct (no syntax errors or import errors).
        
        # We can also mock torch.compile to verify it's called if we want to be safe against environment issues.
        with patch("torch.compile") as mock_compile:
            mock_compile.side_effect = lambda m: m # identity
            
            model = instantiate_model(model_config, data_config, compile=True) # type: ignore
            
            assert isinstance(model, DummyModel)
            mock_compile.assert_called_once()
            
        # Also try real execution if possible?
        # try:
        #     model = instantiate_model(model_config, data_config, compile=True) # type: ignore
        # except Exception as e:
        #     # If it fails due to torch compilation backend issues, that's fine, but we want to catch "cast" issues?
        #     # We removed cast, so that shouldn't be an issue.
        #     pass

if __name__ == "__main__":
    pytest.main([__file__])
