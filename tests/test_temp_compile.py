from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from cerberus.config import DataConfig, ModelConfig
from cerberus.module import instantiate_model


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

    model_config = ModelConfig.model_construct(
        name="test",
        model_cls="dummy.DummyModel",
        loss_cls="dummy.Loss",
        metrics_cls="dummy.Metrics",
        model_args={},
        loss_args={},
        metrics_args={},
        pretrained=[],
        count_pseudocount=0.0,
    )

    data_config = DataConfig.model_construct(
        input_len=10,
        output_len=10,
        output_bin_size=1,
        inputs={},
        targets={},
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    with patch("cerberus.module.import_class") as mock_import:
        mock_import.return_value = DummyModel

        with patch("torch.compile") as mock_compile:
            mock_compile.side_effect = lambda m: m  # identity

            model = instantiate_model(model_config, data_config, compile=True)

            assert isinstance(model, DummyModel)
            mock_compile.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
