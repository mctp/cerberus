from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cerberus.attribution import (
    ATTRIBUTION_MODES,
    DIFFERENTIAL_ATTRIBUTION_MODES,
    AttributionTarget,
    DifferentialAttributionTarget,
    apply_off_simplex_gradient_correction,
    compute_ism_attributions,
    resolve_ism_span,
)


class _ToyCerberusModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        logits = x[:, :2, :]
        log_counts = logits.sum(dim=-1)
        return SimpleNamespace(logits=logits, log_counts=log_counts)


class _WeightedScalarTarget(torch.nn.Module):
    weights: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :4, :] * self.weights).sum(dim=(1, 2))


def test_attribution_target_invalid_mode_raises() -> None:
    model = _ToyCerberusModel()
    with pytest.raises(ValueError, match="Unsupported mode"):
        AttributionTarget(
            model=model,
            mode="bad_mode",
            channel=0,
            bin_index=None,
            window_start=None,
            window_end=None,
        )


def test_attribution_modes_constant() -> None:
    assert "log_counts" in ATTRIBUTION_MODES
    assert "profile_bin" in ATTRIBUTION_MODES
    assert len(ATTRIBUTION_MODES) == 5


def test_attribution_target_log_counts_channel() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        mode="log_counts",
        channel=1,
        bin_index=None,
        window_start=None,
        window_end=None,
    )

    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    out = target(x)
    assert torch.allclose(out, torch.tensor([1.0]))


def test_attribution_target_profile_bin_center_default() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        mode="profile_bin",
        channel=0,
        bin_index=None,
        window_start=None,
        window_end=None,
    )

    x = torch.tensor(
        [
            [
                [0.2, 0.7, 0.1],
                [0.3, 0.1, 0.6],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    out = target(x)
    assert torch.allclose(out, torch.tensor([0.7]))


def test_resolve_ism_span_validation() -> None:
    assert resolve_ism_span(10, None, None) == (0, 10)
    assert resolve_ism_span(10, 2, 5) == (2, 5)

    with pytest.raises(ValueError):
        resolve_ism_span(10, -1, 3)
    with pytest.raises(ValueError):
        resolve_ism_span(10, 6, 6)


def test_compute_ism_attributions_expected_deltas() -> None:
    model = _WeightedScalarTarget()
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0],  # C
                [0.0, 0.0, 1.0],  # G
                [0.0, 0.0, 0.0],  # T
            ]
        ],
        dtype=torch.float32,
    )

    attrs = compute_ism_attributions(model, inputs, ism_start=1, ism_end=3)
    assert attrs.shape == (1, 4, 3)

    expected = torch.zeros((1, 4, 3), dtype=torch.float32)
    # Position 1, reference base C (weight=2): raw deltas [-1, 0, 1, 2], mean=0.5.
    expected[0, :, 1] = torch.tensor([-1.0, -0.5, 1.0, 2.0])
    # Position 2, reference base G (weight=3): raw deltas [-2, -1, 0, 1], mean=-0.5.
    expected[0, :, 2] = torch.tensor([-2.0, -1.0, 0.5, 1.0])

    assert torch.allclose(attrs, expected)


# ---------------------------------------------------------------------------
# DifferentialAttributionTarget
# ---------------------------------------------------------------------------


class _MultiConditionModel(torch.nn.Module):
    """Toy multi-condition model: log_counts = sum of each input channel pair."""

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        # x: (B, 4, L); split into 2 mock conditions from first 2 input channels
        L = x.shape[-1]
        logits_a = x[:, :1, :].expand(-1, 1, L)    # (B, 1, L)
        logits_b = x[:, 1:2, :].expand(-1, 1, L)   # (B, 1, L)
        logits = torch.cat([logits_a, logits_b], dim=1)  # (B, 2, L)
        log_counts = logits.sum(dim=-1)              # (B, 2)
        return SimpleNamespace(logits=logits, log_counts=log_counts)


def test_differential_attribution_modes_constant() -> None:
    assert "delta_log_counts" in DIFFERENTIAL_ATTRIBUTION_MODES
    assert "delta_profile_window_sum" in DIFFERENTIAL_ATTRIBUTION_MODES
    assert len(DIFFERENTIAL_ATTRIBUTION_MODES) == 2


def test_differential_attribution_target_invalid_mode() -> None:
    model = _MultiConditionModel()
    with pytest.raises(ValueError, match="Unsupported mode"):
        DifferentialAttributionTarget(model=model, mode="log_counts")


def test_differential_attribution_target_same_idx_raises() -> None:
    model = _MultiConditionModel()
    with pytest.raises(ValueError, match="must differ"):
        DifferentialAttributionTarget(model=model, mode="delta_log_counts", cond_a_idx=1, cond_b_idx=1)


def test_differential_attribution_target_delta_log_counts() -> None:
    """delta_log_counts = log_counts_B − log_counts_A."""
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(model=model, mode="delta_log_counts", cond_a_idx=0, cond_b_idx=1)

    # x: channel 0 = all 1s, channel 1 = all 2s
    x = torch.zeros(2, 4, 5)
    x[:, 0, :] = 1.0
    x[:, 1, :] = 2.0

    out = target(x)
    assert out.shape == (2,)
    # log_counts_A = sum(1,1,1,1,1) = 5, log_counts_B = sum(2,...) = 10 → delta = 5
    assert torch.allclose(out, torch.tensor([5.0, 5.0]))


def test_differential_attribution_target_delta_profile_window_sum() -> None:
    """delta_profile_window_sum sums (logits_B - logits_A) in window."""
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(
        model=model,
        mode="delta_profile_window_sum",
        cond_a_idx=0,
        cond_b_idx=1,
        window_start=1,
        window_end=3,
    )

    x = torch.zeros(1, 4, 5)
    x[:, 0, :] = 1.0  # logits_A everywhere = 1
    x[:, 1, :] = 3.0  # logits_B everywhere = 3

    out = target(x)
    assert out.shape == (1,)
    # delta in window [1,3) = (3-1)*2 = 4
    assert torch.allclose(out, torch.tensor([4.0]))


def test_differential_attribution_target_invalid_window() -> None:
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(
        model=model, mode="delta_profile_window_sum", window_start=10, window_end=5
    )
    with pytest.raises(ValueError, match="Invalid window"):
        target(torch.zeros(1, 4, 8))


def test_differential_attribution_target_out_of_range_channel() -> None:
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(model=model, mode="delta_log_counts", cond_a_idx=0, cond_b_idx=5)
    with pytest.raises(ValueError, match="out of range"):
        target(torch.zeros(1, 4, 5))


def test_differential_attribution_target_ism_integration() -> None:
    """DifferentialAttributionTarget works as drop-in for compute_ism_attributions."""
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(model=model, mode="delta_log_counts")
    x = torch.zeros(1, 4, 8)
    x[0, 0, :] = 1.0
    attrs = compute_ism_attributions(target, x, ism_start=None, ism_end=None)
    assert attrs.shape == (1, 4, 8)


def test_differential_attribution_target_gradient_flows() -> None:
    """Gradient flows back to input from delta_log_counts target."""
    model = _MultiConditionModel()
    target = DifferentialAttributionTarget(model=model, mode="delta_log_counts")
    x = torch.zeros(1, 4, 5, requires_grad=True)
    out = target(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_apply_off_simplex_gradient_correction() -> None:
    attrs = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ],
        dtype=np.float32,
    )
    corrected = apply_off_simplex_gradient_correction(attrs)
    assert np.allclose(corrected.mean(axis=1), 0.0)

    with pytest.raises(ValueError):
        apply_off_simplex_gradient_correction(np.ones((4, 10), dtype=np.float32))
