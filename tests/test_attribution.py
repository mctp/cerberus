from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cerberus.attribution import (
    ATTRIBUTION_MODES,
    AttributionTarget,
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
