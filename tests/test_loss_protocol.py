"""Tests that all loss classes conform to the CerberusLoss protocol."""

import pytest
import torch
import torch.nn as nn

from cerberus.loss import (
    CoupledMSEMultinomialLoss,
    CoupledNegativeBinomialMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    DalmatianLoss,
    MSEMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    PoissonMultinomialLoss,
    ProfilePoissonNLLLoss,
)
from cerberus.output import (
    FactorizedProfileCountOutput,
    ProfileCountOutput,
    ProfileLogRates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_targets(batch: int = 4, channels: int = 2, length: int = 16) -> torch.Tensor:
    """Non-negative integer-valued targets (required by NB loss)."""
    return torch.poisson(torch.ones(batch, channels, length) * 5.0)


def _make_profile_count_output(
    batch: int = 4, channels: int = 2, length: int = 16
) -> ProfileCountOutput:
    # log_counts shape (B, 1) for global count mode (count_per_channel=False default)
    return ProfileCountOutput(
        logits=torch.randn(batch, channels, length),
        log_counts=torch.randn(batch, 1),
    )


def _make_profile_log_rates(
    batch: int = 4, channels: int = 2, length: int = 16
) -> ProfileLogRates:
    return ProfileLogRates(log_rates=torch.randn(batch, channels, length))


# ---------------------------------------------------------------------------
# Protocol structural check
# ---------------------------------------------------------------------------

ALL_LOSS_CLASSES: list[type[nn.Module]] = [
    ProfilePoissonNLLLoss,
    MSEMultinomialLoss,
    CoupledMSEMultinomialLoss,
    PoissonMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    CoupledNegativeBinomialMultinomialLoss,
    DalmatianLoss,
]


@pytest.mark.parametrize("cls", ALL_LOSS_CLASSES, ids=lambda c: c.__name__)
def test_has_loss_components(cls):
    """Every loss class must have a loss_components method."""
    assert hasattr(cls, "loss_components"), f"{cls.__name__} is missing loss_components"


# ---------------------------------------------------------------------------
# ProfileCountOutput losses
# ---------------------------------------------------------------------------

PROFILE_COUNT_LOSSES = [
    MSEMultinomialLoss,
    PoissonMultinomialLoss,
    NegativeBinomialMultinomialLoss,
]


@pytest.mark.parametrize("cls", PROFILE_COUNT_LOSSES, ids=lambda c: c.__name__)
def test_profile_count_loss_components(cls):
    """ProfileCount losses return dict with profile_loss and count_loss keys."""
    loss_fn = cls()
    outputs = _make_profile_count_output()
    targets = _make_targets()

    components = loss_fn.loss_components(outputs, targets)
    assert isinstance(components, dict)
    assert "profile_loss" in components
    assert "count_loss" in components
    for v in components.values():
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 0  # scalar


@pytest.mark.parametrize("cls", PROFILE_COUNT_LOSSES, ids=lambda c: c.__name__)
def test_profile_count_forward_scalar(cls):
    """forward() returns a scalar tensor."""
    loss_fn = cls()
    outputs = _make_profile_count_output()
    targets = _make_targets()

    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Coupled (ProfileLogRates) losses
# ---------------------------------------------------------------------------

COUPLED_LOSSES = [
    CoupledMSEMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    CoupledNegativeBinomialMultinomialLoss,
]


@pytest.mark.parametrize("cls", COUPLED_LOSSES, ids=lambda c: c.__name__)
def test_coupled_loss_components(cls):
    """Coupled losses return dict with profile_loss and count_loss keys."""
    loss_fn = cls()
    outputs = _make_profile_log_rates()
    targets = _make_targets()

    components = loss_fn.loss_components(outputs, targets)
    assert isinstance(components, dict)
    assert "profile_loss" in components
    assert "count_loss" in components
    for v in components.values():
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 0


@pytest.mark.parametrize("cls", COUPLED_LOSSES, ids=lambda c: c.__name__)
def test_coupled_forward_scalar(cls):
    """forward() returns a scalar tensor."""
    loss_fn = cls()
    outputs = _make_profile_log_rates()
    targets = _make_targets()

    loss = loss_fn(outputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# ProfilePoissonNLLLoss
# ---------------------------------------------------------------------------


def test_poisson_nll_loss_components():
    """ProfilePoissonNLLLoss returns dict with poisson_nll_loss key."""
    loss_fn = ProfilePoissonNLLLoss(log_input=True, full=False)
    outputs = _make_profile_log_rates(channels=1, length=1)
    targets = _make_targets(channels=1, length=1)

    components = loss_fn.loss_components(outputs, targets)
    assert isinstance(components, dict)
    assert "poisson_nll_loss" in components
    assert components["poisson_nll_loss"].ndim == 0


def test_poisson_nll_forward_matches_components():
    """forward() result equals the single component value."""
    loss_fn = ProfilePoissonNLLLoss(log_input=True, full=False)
    outputs = _make_profile_log_rates(channels=1, length=1)
    targets = _make_targets(channels=1, length=1)

    loss = loss_fn(outputs, targets)
    components = loss_fn.loss_components(outputs, targets)
    torch.testing.assert_close(loss, components["poisson_nll_loss"])


# ---------------------------------------------------------------------------
# DalmatianLoss
# ---------------------------------------------------------------------------


def _make_dalmatian_output(
    B: int = 4, C: int = 2, L: int = 16
) -> FactorizedProfileCountOutput:
    return FactorizedProfileCountOutput(
        logits=torch.randn(B, C, L),
        log_counts=torch.randn(B, 1),
        bias_logits=torch.randn(B, C, L),
        bias_log_counts=torch.randn(B, 1),
        signal_logits=torch.randn(B, C, L),
        signal_log_counts=torch.randn(B, 1),
    )


def test_dalmatian_loss_components():
    """DalmatianLoss returns dict with recon_loss and bias_loss keys."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_weight": 1.0, "profile_weight": 1.0},
    )
    B, C, L = 4, 2, 16
    output = _make_dalmatian_output(B, C, L)
    targets = _make_targets(B, C, L)
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    components = loss_fn.loss_components(
        output, targets, interval_source=interval_source
    )
    assert isinstance(components, dict)
    assert "recon_loss" in components
    assert "bias_loss" in components
    for v in components.values():
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 0


def test_dalmatian_forward_uses_components():
    """forward() result is consistent with loss_components."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_weight": 1.0, "profile_weight": 1.0},
        bias_weight=0.5,
    )
    B, C, L = 4, 2, 16
    output = _make_dalmatian_output(B, C, L)
    targets = _make_targets(B, C, L)
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    loss = loss_fn(output, targets, interval_source=interval_source)
    components = loss_fn.loss_components(
        output, targets, interval_source=interval_source
    )
    expected = components["recon_loss"] + 0.5 * components["bias_loss"]
    torch.testing.assert_close(loss, expected)


# ---------------------------------------------------------------------------
# Consistency: forward() matches weighted loss_components for all dual losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", PROFILE_COUNT_LOSSES, ids=lambda c: c.__name__)
def test_forward_matches_weighted_components_profile_count(cls):
    """forward() equals profile_weight * profile_loss + count_weight * count_loss."""
    loss_fn = cls(profile_weight=1.5, count_weight=0.3)
    outputs = _make_profile_count_output()
    targets = _make_targets()

    loss = loss_fn(outputs, targets)
    components = loss_fn.loss_components(outputs, targets)
    expected = 1.5 * components["profile_loss"] + 0.3 * components["count_loss"]
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("cls", COUPLED_LOSSES, ids=lambda c: c.__name__)
def test_forward_matches_weighted_components_coupled(cls):
    """forward() equals profile_weight * profile_loss + count_weight * count_loss."""
    loss_fn = cls(profile_weight=1.5, count_weight=0.3)
    outputs = _make_profile_log_rates()
    targets = _make_targets()

    loss = loss_fn(outputs, targets)
    components = loss_fn.loss_components(outputs, targets)
    expected = 1.5 * components["profile_loss"] + 0.3 * components["count_loss"]
    torch.testing.assert_close(loss, expected)
