from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cerberus.attribution import (
    N_NUCLEOTIDES,
    TARGET_REDUCTIONS,
    AttributionTarget,
    compute_ism_attributions,
    mean_center_attributions,
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


def test_attribution_target_invalid_reduction_raises() -> None:
    model = _ToyCerberusModel()
    with pytest.raises(ValueError, match="Unsupported reduction"):
        AttributionTarget(
            model=model,
            reduction="bad_reduction",
            channel=0,
            bin_index=None,
            window_start=None,
            window_end=None,
        )


def test_target_reductions_constant() -> None:
    assert "log_counts" in TARGET_REDUCTIONS
    assert "profile_bin" in TARGET_REDUCTIONS
    assert len(TARGET_REDUCTIONS) == 5


def test_n_nucleotides_constant() -> None:
    assert N_NUCLEOTIDES == 4


def test_attribution_target_log_counts_channel() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="log_counts",
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
        reduction="profile_bin",
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
    assert resolve_ism_span(10, (None, None)) == (0, 10)
    assert resolve_ism_span(10, (2, 5)) == (2, 5)

    with pytest.raises(ValueError):
        resolve_ism_span(10, (-1, 3))
    with pytest.raises(ValueError):
        resolve_ism_span(10, (6, 6))


def test_resolve_ism_span_half_open() -> None:
    assert resolve_ism_span(10, (None, 5)) == (0, 5)
    assert resolve_ism_span(10, (3, None)) == (3, 10)


def test_attribution_target_profile_window_sum() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="profile_window_sum",
        channel=0,
        bin_index=None,
        window_start=0,
        window_end=2,
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
    # logits channel 0 = x[0, 0, :] = [0.2, 0.7, 0.1]; sum over [0:2) = 0.9.
    assert torch.allclose(target(x), torch.tensor([0.9]))


def test_attribution_target_pred_count_bin() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="pred_count_bin",
        channel=0,
        bin_index=1,
        window_start=None,
        window_end=None,
    )
    # Build logits [[0, 0, 0], [0, 0, 0]] via two rows of zeros; softmax → uniform 1/3.
    # log_counts = logits.sum(-1) = [0, 0]; exp(log_counts[:, 0]) = 1.
    # Expected pred_counts at bin 1 = 1/3.
    x = torch.zeros((1, 4, 3))
    out = target(x)
    assert torch.allclose(out, torch.tensor([1.0 / 3.0]))


def test_attribution_target_invalid_channel_raises() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="log_counts",
        channel=99,
        bin_index=None,
        window_start=None,
        window_end=None,
    )
    x = torch.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="Requested channel=99"):
        target(x)


def test_attribution_target_invalid_bin_index_raises() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="profile_bin",
        channel=0,
        bin_index=99,
        window_start=None,
        window_end=None,
    )
    x = torch.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="Invalid bin_index=99"):
        target(x)


def test_attribution_target_invalid_window_raises() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="profile_window_sum",
        channel=0,
        bin_index=None,
        window_start=5,
        window_end=2,
    )
    x = torch.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="Invalid window"):
        target(x)


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

    attrs = compute_ism_attributions(model, inputs, span=(1, 3))
    assert attrs.shape == (1, 4, 3)

    expected = torch.zeros((1, 4, 3), dtype=torch.float32)
    # Position 1, reference base C (weight=2): raw deltas [-1, 0, 1, 2], mean=0.5.
    expected[0, :, 1] = torch.tensor([-1.0, -0.5, 1.0, 2.0])
    # Position 2, reference base G (weight=3): raw deltas [-2, -1, 0, 1], mean=-0.5.
    expected[0, :, 2] = torch.tensor([-2.0, -1.0, 0.5, 1.0])

    assert torch.allclose(attrs, expected)


def test_compute_ism_attributions_multi_batch_distinct_refs() -> None:
    """Multi-batch with different ref bases at each (b, pos).

    Regression guard for the vectorized scatter: the old per-position loop
    wrote the ref override one batch at a time using advanced indexing; the
    new code uses `scatter_` with a (B, 1, span_len) index. A broadcasting
    or dim error would produce per-sample values that don't match.
    """
    model = _WeightedScalarTarget()
    # Sample 0: A, C, G   Sample 1: T, G, A
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0],  # C
                [0.0, 0.0, 1.0],  # G
                [0.0, 0.0, 0.0],  # T
            ],
            [
                [0.0, 0.0, 1.0],  # A
                [0.0, 0.0, 0.0],  # C
                [0.0, 1.0, 0.0],  # G
                [1.0, 0.0, 0.0],  # T
            ],
        ],
        dtype=torch.float32,
    )
    attrs = compute_ism_attributions(model, inputs, span=(0, 3))
    assert attrs.shape == (2, 4, 3)

    expected = torch.zeros((2, 4, 3), dtype=torch.float32)
    # Sample 0
    expected[0, :, 0] = torch.tensor([-1.5, 1.0, 2.0, 3.0])  # ref A
    expected[0, :, 1] = torch.tensor([-1.0, -0.5, 1.0, 2.0])  # ref C
    expected[0, :, 2] = torch.tensor([-2.0, -1.0, 0.5, 1.0])  # ref G
    # Sample 1
    expected[1, :, 0] = torch.tensor([-3.0, -2.0, -1.0, 1.5])  # ref T
    expected[1, :, 1] = torch.tensor([-2.0, -1.0, 0.5, 1.0])  # ref G
    expected[1, :, 2] = torch.tensor([-1.5, 1.0, 2.0, 3.0])  # ref A

    assert torch.allclose(attrs, expected)


def test_compute_ism_attributions_outside_span_is_zero() -> None:
    """Positions outside the resolved span remain zero.

    Regression guard for the hoist: before the refactor, the override ran
    inside the loop over span positions. The new code slices `attrs[:, :,
    span_start:span_end]` explicitly; a wrong slice dim or off-by-one would
    leak non-zero values outside the span.
    """
    model = _WeightedScalarTarget()
    inputs = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0]] * 4],  # 5 positions, all A (ignored at ref)
        dtype=torch.float32,
    )
    # Make at least one position distinguishable: set pos 2 to C.
    inputs[0, 0, 2] = 0.0
    inputs[0, 1, 2] = 1.0
    attrs = compute_ism_attributions(model, inputs, span=(1, 3))
    assert attrs.shape == (1, 4, 5)
    assert torch.all(attrs[:, :, 0] == 0.0)
    assert torch.all(attrs[:, :, 3:] == 0.0)
    # At least one in-span position has non-zero values.
    assert torch.any(attrs[:, :, 1:3] != 0.0)


def test_compute_ism_attributions_full_length_span() -> None:
    """`span=(None, None)` covers every position."""
    model = _WeightedScalarTarget()
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    attrs_full = compute_ism_attributions(model, inputs, span=(None, None))
    attrs_explicit = compute_ism_attributions(model, inputs, span=(0, 3))
    assert torch.allclose(attrs_full, attrs_explicit)


def test_compute_ism_attributions_half_open_span() -> None:
    """`(None, end)` equals `(0, end)`; `(start, None)` equals `(start, seq_len)`."""
    model = _WeightedScalarTarget()
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        compute_ism_attributions(model, inputs, span=(None, 2)),
        compute_ism_attributions(model, inputs, span=(0, 2)),
    )
    assert torch.allclose(
        compute_ism_attributions(model, inputs, span=(2, None)),
        compute_ism_attributions(model, inputs, span=(2, 4)),
    )


def test_compute_ism_attributions_extra_conditioning_channels() -> None:
    """Inputs with channels beyond the 4 DNA channels are ignored by ISM."""
    model = _WeightedScalarTarget()
    # 6 channels: first 4 are DNA (ACGT), last 2 are conditioning junk.
    inputs = torch.zeros((1, 6, 3), dtype=torch.float32)
    inputs[0, 1, 0] = 1.0  # C at pos 0
    inputs[0, 2, 1] = 1.0  # G at pos 1
    inputs[0, 0, 2] = 1.0  # A at pos 2
    inputs[0, 4, :] = 0.5  # conditioning noise
    inputs[0, 5, :] = -0.3  # conditioning noise

    attrs = compute_ism_attributions(model, inputs, span=(0, 3))
    assert attrs.shape == (1, 4, 3)
    # Conditioning channels must not leak into the DNA attributions.
    # Verify by comparing against the 4-channel subset.
    inputs_4ch = inputs[:, :4, :].clone()
    attrs_4ch = compute_ism_attributions(model, inputs_4ch, span=(0, 3))
    assert torch.allclose(attrs, attrs_4ch)


def test_compute_ism_attributions_rejects_too_few_channels() -> None:
    model = _WeightedScalarTarget()
    inputs = torch.zeros((1, 3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match=f">={N_NUCLEOTIDES} DNA channels"):
        compute_ism_attributions(model, inputs, span=(None, None))


def test_compute_ism_attributions_ref_channel_is_negative_span_mean() -> None:
    """At each in-span position, attrs[b, ref, l] == -mean_j raw_delta_j.

    Direct check of the TF-MoDISco convention, independent of the specific
    weight values used elsewhere.
    """
    model = _WeightedScalarTarget()
    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    attrs = compute_ism_attributions(model, inputs, span=(0, 3))
    # Reconstruct raw deltas by running ISM without the override and checking
    # that the observed-base channel stores -mean of the other three plus the
    # zero at ref itself (i.e. -(sum_j_nonref + 0) / 4).
    for pos in range(3):
        ref = int(inputs[0, :4, pos].argmax().item())
        nonref_deltas = [attrs[0, b, pos].item() for b in range(4) if b != ref]
        # Raw mean = (sum_nonref + 0) / 4.
        raw_mean = sum(nonref_deltas) / 4.0
        assert attrs[0, ref, pos].item() == pytest.approx(-raw_mean, abs=1e-6)


def test_mean_center_attributions() -> None:
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
    centered = mean_center_attributions(attrs)
    assert np.allclose(centered.mean(axis=1), 0.0)

    with pytest.raises(ValueError):
        mean_center_attributions(np.ones((4, 10), dtype=np.float32))
