"""Tests for the bias-model ISM plotting helpers.

The full ISM-plus-plot pipeline needs a real model + FASTA fixture, so we
only pin the batched-forward helper and the model-type dispatch hook here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools.plot_biasnet_ism import _batch_forward_center  # noqa: E402


class _ConstantLogitsModel(nn.Module):
    """Tiny model whose forward returns a known logits tensor.

    Used to pin _batch_forward_center's output-channel slicing without
    depending on a real BPNet / BiasNet checkpoint.
    """

    def __init__(self, n_channels: int, length: int, value: float = 1.0):
        super().__init__()
        self.n_channels = n_channels
        self.length = length
        self.value = value
        # One throwaway parameter so .parameters() is non-empty (some
        # downstream callers iterate it).
        self.register_parameter("_p", nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor):
        from cerberus.output import ProfileCountOutput
        batch = x.shape[0]
        logits = torch.full(
            (batch, self.n_channels, self.length),
            self.value, dtype=torch.float32,
        )
        return ProfileCountOutput(
            logits=logits,
            log_counts=torch.zeros(batch, self.n_channels),
        )


def test_batch_forward_center_returns_centre_value_array():
    """The helper returns one float per batch row -- the value of the
    model's logit at the requested centre position."""
    model = _ConstantLogitsModel(n_channels=1, length=32, value=3.14)
    batch = torch.zeros(8, 4, 64)
    centre = 16
    out = _batch_forward_center(model, batch, output_idx=centre, batch_size=4)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8,)
    assert np.allclose(out, 3.14)


def test_batch_forward_center_chunks_obey_batch_size():
    """When ``batch_size`` is smaller than the input batch, the helper
    iterates in chunks; the assembled output still covers every row."""
    model = _ConstantLogitsModel(n_channels=1, length=8, value=1.5)
    batch = torch.zeros(13, 4, 16)
    out = _batch_forward_center(model, batch, output_idx=4, batch_size=5)
    assert out.shape == (13,)
    assert np.allclose(out, 1.5)


def test_load_biasnet_recognises_chrombpnet_bias_model_name():
    """The model-name dispatch in load_biasnet covers the
    ChromBPNetBiasBPNet checkpoint type so stage-1 ChromBPNet bias
    branches can be plotted alongside standalone BiasNet runs."""
    import inspect
    from tools.plot_biasnet_ism import load_biasnet
    # Body-source inspection guards the dispatch branch without needing
    # a real checkpoint on disk.  This is the lightest possible regression
    # guard: a future refactor that drops the name has to update the test.
    src = inspect.getsource(load_biasnet)
    assert "ChromBPNetBiasBPNet" in src
    assert "cerberus.models.bpnet.BPNet" in src
