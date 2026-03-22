"""Test that count_per_channel=True rejects scalar (total-count) predictions.

When predict_total_count=True, the model outputs log_counts of shape (B, 1).
With count_per_channel=True, the loss expects (B, C). PyTorch would silently
broadcast (B, 1) -> (B, C), training the count head on the wrong objective.
These tests verify that a ValueError is raised for each affected loss class.
"""

import pytest
import torch

from cerberus.loss import (
    MSEMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    PoissonMultinomialLoss,
)
from cerberus.output import ProfileCountOutput

B, C, L = 2, 3, 10


def _make_inputs(pred_log_counts_shape, integer_targets=False):
    """Create dummy targets and ProfileCountOutput with given log_counts shape."""
    if integer_targets:
        targets = torch.randint(1, 10, (B, C, L)).float()
    else:
        targets = torch.rand(B, C, L)
    logits = torch.randn(B, C, L)
    log_counts = torch.randn(*pred_log_counts_shape)
    outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)
    return outputs, targets


class TestCountPerChannelShapeMismatch:
    """count_per_channel=True + (B,1) log_counts must raise ValueError."""

    def test_mse_multinomial_rejects_total_count(self):
        loss_fn = MSEMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, 1))
        with pytest.raises(ValueError, match="count_per_channel=True"):
            loss_fn(outputs, targets)

    def test_poisson_multinomial_rejects_total_count(self):
        loss_fn = PoissonMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, 1))
        with pytest.raises(ValueError, match="count_per_channel=True"):
            loss_fn(outputs, targets)

    def test_nb_multinomial_rejects_total_count(self):
        loss_fn = NegativeBinomialMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, 1), integer_targets=True)
        with pytest.raises(ValueError, match="count_per_channel=True"):
            loss_fn(outputs, targets)


class TestCountPerChannelCorrectShape:
    """count_per_channel=True + (B, C) log_counts must work normally."""

    def test_mse_multinomial_accepts_per_channel(self):
        loss_fn = MSEMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, C))
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_poisson_multinomial_accepts_per_channel(self):
        loss_fn = PoissonMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, C))
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_nb_multinomial_accepts_per_channel(self):
        loss_fn = NegativeBinomialMultinomialLoss(count_per_channel=True)
        outputs, targets = _make_inputs((B, C), integer_targets=True)
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestGlobalCountStillWorks:
    """count_per_channel=False (default) with (B, 1) must still work."""

    def test_mse_multinomial_global_count(self):
        loss_fn = MSEMultinomialLoss(count_per_channel=False)
        outputs, targets = _make_inputs((B, 1))
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_poisson_multinomial_global_count(self):
        loss_fn = PoissonMultinomialLoss(count_per_channel=False)
        outputs, targets = _make_inputs((B, 1))
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_nb_multinomial_global_count(self):
        loss_fn = NegativeBinomialMultinomialLoss(count_per_channel=False)
        outputs, targets = _make_inputs((B, 1), integer_targets=True)
        loss = loss_fn(outputs, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)
