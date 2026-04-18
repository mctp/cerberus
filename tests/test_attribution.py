from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cerberus.attribution import (
    N_NUCLEOTIDES,
    TARGET_REDUCTIONS,
    AttributionTarget,
    compute_ism_attributions,
    compute_taylor_ism_attributions,
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
            channels=0,
        )


def test_target_reductions_constant() -> None:
    assert "log_counts" in TARGET_REDUCTIONS
    assert "profile_bin" in TARGET_REDUCTIONS
    assert "delta_log_counts" in TARGET_REDUCTIONS
    assert "delta_profile_window_sum" in TARGET_REDUCTIONS
    assert len(TARGET_REDUCTIONS) == 7


def test_n_nucleotides_constant() -> None:
    assert N_NUCLEOTIDES == 4


def test_attribution_target_log_counts_channel() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="log_counts",
        channels=1,
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
        channels=0,
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
        channels=0,
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
        channels=0,
        bin_index=1,
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
        channels=99,
    )
    x = torch.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="Requested channel=99"):
        target(x)


def test_attribution_target_invalid_bin_index_raises() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="profile_bin",
        channels=0,
        bin_index=99,
    )
    x = torch.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="Invalid bin_index=99"):
        target(x)


def test_attribution_target_invalid_window_raises() -> None:
    model = _ToyCerberusModel()
    target = AttributionTarget(
        model=model,
        reduction="profile_window_sum",
        channels=0,
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


class _TinyNonlinearTarget(torch.nn.Module):
    """Small non-linear scalar target used to exercise the paper's identities.

    Deterministic (no dropout, no BN), so gradients are reproducible across
    runs without needing an explicit seed or eval() call.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(N_NUCLEOTIDES, 6, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.conv(x[:, :N_NUCLEOTIDES, :]))
        # Mean-pool across length, then linear head → (B,) scalar.
        return self.head(h.mean(dim=-1)).squeeze(-1)


def _make_onehot_sequence(
    indices: list[list[int]], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a ``(B, 4, L)`` one-hot tensor from per-sample base indices."""
    batch = len(indices)
    seq_len = len(indices[0])
    x = torch.zeros((batch, N_NUCLEOTIDES, seq_len), dtype=dtype)
    for b, seq in enumerate(indices):
        for pos, base in enumerate(seq):
            x[b, base, pos] = 1.0
    return x


def test_taylor_ism_linear_parity_single_batch() -> None:
    """On a linear model, TISM is bit-identical to exact ISM (paper Eq. 7).

    Core correctness gate: if the model's input-to-target map is linear, the
    first-order Taylor expansion is exact, so every span / formatting must
    match the brute-force ISM output to float tolerance.
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2, 3, 2]])

    for span in [(None, None), (0, 5), (1, 4), (2, 3), (None, 3), (2, None)]:
        ism = compute_ism_attributions(model, inputs, span=span)
        taylor = compute_taylor_ism_attributions(model, inputs, span=span)
        assert torch.allclose(taylor, ism, atol=1e-6), f"mismatch for span={span}"


def test_taylor_ism_linear_parity_multi_batch_distinct_refs() -> None:
    """Multi-batch with different ref bases per (b, pos): linear parity holds.

    Stronger than the single-batch test — catches any broadcasting error in
    the dot-product ``grad_ref`` computation or the span scatter.
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence(
        [
            [0, 1, 2, 3],  # A C G T
            [3, 2, 1, 0],  # T G C A
            [1, 1, 3, 0],  # C C T A
        ]
    )
    ism = compute_ism_attributions(model, inputs, span=(None, None))
    taylor = compute_taylor_ism_attributions(model, inputs, span=(None, None))
    assert torch.allclose(taylor, ism, atol=1e-6)


def test_taylor_ism_raw_mode_has_zero_at_reference() -> None:
    """``tf_modisco_format=False`` returns raw TISM: ref channel == 0 in span.

    This matches the TISM reference's ``output='tism'`` behavior.
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2, 3], [3, 2, 1, 0]])
    attrs = compute_taylor_ism_attributions(
        model, inputs, span=(0, 4), tf_modisco_format=False
    )
    for b in range(inputs.shape[0]):
        for pos in range(4):
            ref = int(inputs[b, :N_NUCLEOTIDES, pos].argmax().item())
            assert attrs[b, ref, pos].item() == 0.0, (
                f"raw mode should have 0 at reference (b={b}, pos={pos}, "
                f"ref={ref}), got {attrs[b, ref, pos].item()}"
            )


def test_taylor_ism_majdandzic_bridge_on_nonlinear_model() -> None:
    """``mean_center(raw_tism) == grads - grads.mean(dim=1)`` within the span.

    Paper Eq. 8 / 11 / 15 identity (Sasse et al. 2024 links Majdandzic 2023
    off-simplex correction to TISM via this equivalence). Must hold for any
    model — linear or not — to numerical precision.
    """
    model = _TinyNonlinearTarget().eval()
    torch.manual_seed(1)
    inputs = _make_onehot_sequence([[0, 1, 2, 3, 1, 2, 0, 3]])

    span_start, span_end = 1, 6
    raw_tism = compute_taylor_ism_attributions(
        model, inputs, span=(span_start, span_end), tf_modisco_format=False
    )

    # Reference: direct gradient minus mean across channels.
    x = inputs.detach().clone().requires_grad_(True)
    out = model(x).sum()
    (grads,) = torch.autograd.grad(out, x)
    grads_centered = grads[:, :N_NUCLEOTIDES, :] - grads[:, :N_NUCLEOTIDES, :].mean(
        dim=1, keepdim=True
    )

    centered_tism = mean_center_attributions(raw_tism.detach().numpy())

    np.testing.assert_allclose(
        centered_tism[:, :, span_start:span_end],
        grads_centered[:, :, span_start:span_end].detach().numpy(),
        atol=1e-5,
    )


def test_taylor_ism_outside_span_is_zero() -> None:
    """I/O contract: positions outside the resolved span are exactly zero.

    Matches :func:`compute_ism_attributions`.
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2, 0, 1]])
    attrs = compute_taylor_ism_attributions(model, inputs, span=(1, 3))
    assert torch.all(attrs[:, :, 0] == 0.0)
    assert torch.all(attrs[:, :, 3:] == 0.0)
    assert torch.any(attrs[:, :, 1:3] != 0.0)


def test_taylor_ism_rejects_too_few_channels() -> None:
    """I/O contract: same channel-count validation as exact ISM."""
    model = _WeightedScalarTarget()
    inputs = torch.zeros((1, 3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match=f">={N_NUCLEOTIDES} DNA channels"):
        compute_taylor_ism_attributions(model, inputs, span=(None, None))


def test_taylor_ism_preserves_dtype_and_shape() -> None:
    """I/O contract: output dtype follows input dtype, shape is (B, 4, L)."""
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2], [3, 0, 1]], dtype=torch.float32)
    attrs = compute_taylor_ism_attributions(model, inputs, span=(None, None))
    assert attrs.shape == (2, N_NUCLEOTIDES, 3)
    assert attrs.dtype == torch.float32


def test_taylor_ism_works_inside_no_grad_context() -> None:
    """Callers nested in ``@torch.no_grad()`` (eval loops) must still work.

    Without the :func:`torch.enable_grad` wrap, the gradient call would fail
    with an obscure error inside ``torch.no_grad()``. This guard keeps the
    function usable from the export tool and other inference-mode paths.
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2]])
    with torch.no_grad():
        attrs = compute_taylor_ism_attributions(model, inputs, span=(None, None))
    assert attrs.shape == (1, N_NUCLEOTIDES, 3)


def test_taylor_ism_default_output_matches_ism_shape_and_format() -> None:
    """Default (``tf_modisco_format=True``) returns the same shape and the
    same TF-MoDISco ref-channel formatting as :func:`compute_ism_attributions`.

    Linear model → values also match (redundant with the parity test, but
    asserts the full contract in one place).
    """
    model = _WeightedScalarTarget()
    inputs = _make_onehot_sequence([[0, 1, 2, 3]])
    ism = compute_ism_attributions(model, inputs, span=(None, None))
    taylor = compute_taylor_ism_attributions(model, inputs, span=(None, None))
    assert taylor.shape == ism.shape
    assert taylor.dtype == ism.dtype
    # Ref channel at each position must equal -mean_j raw_delta_j (both funcs).
    for pos in range(4):
        ref = int(inputs[0, :N_NUCLEOTIDES, pos].argmax().item())
        assert taylor[0, ref, pos].item() == pytest.approx(ism[0, ref, pos].item(), abs=1e-6)


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


# ---------------------------------------------------------------------------
# AttributionTarget — delta reductions
# ---------------------------------------------------------------------------


class _MultiConditionModel(torch.nn.Module):
    """Toy multi-condition model: logits = first 2 DNA channels, log_counts = logits.sum(-1)."""

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        logits = x[:, :2, :]
        log_counts = logits.sum(dim=-1)
        return SimpleNamespace(logits=logits, log_counts=log_counts)


def test_attribution_target_delta_reductions_in_target_reductions() -> None:
    assert {"delta_log_counts", "delta_profile_window_sum"}.issubset(TARGET_REDUCTIONS)


def test_attribution_target_delta_requires_tuple_channels() -> None:
    """Arity guard: delta reduction with ``channels: int`` must raise."""
    model = _MultiConditionModel()
    with pytest.raises(TypeError, match="channels=\\(cond_a_idx, cond_b_idx\\)"):
        AttributionTarget(model=model, reduction="delta_log_counts", channels=0)


def test_attribution_target_single_rejects_tuple_channels() -> None:
    """Arity guard: single-channel reduction with tuple channels must raise."""
    model = _MultiConditionModel()
    with pytest.raises(TypeError, match="channels: int"):
        AttributionTarget(model=model, reduction="log_counts", channels=(0, 1))


def test_attribution_target_delta_same_idx_raises() -> None:
    model = _MultiConditionModel()
    with pytest.raises(ValueError, match="must differ"):
        AttributionTarget(
            model=model, reduction="delta_log_counts", channels=(1, 1)
        )


def test_attribution_target_delta_log_counts() -> None:
    model = _MultiConditionModel()
    target = AttributionTarget(
        model=model, reduction="delta_log_counts", channels=(0, 1)
    )
    # Build x such that channel 0 sums to 3 and channel 1 sums to 7 in each batch element.
    x = torch.zeros(2, 4, 5)
    x[:, 0, :3] = 1.0  # ch0 sum = 3
    x[:, 1, :] = 7.0 / 5.0  # ch1 sum = 7
    out = target(x)
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.full((2,), 4.0))


def test_attribution_target_delta_profile_window_sum_full_window() -> None:
    model = _MultiConditionModel()
    target = AttributionTarget(
        model=model,
        reduction="delta_profile_window_sum",
        channels=(0, 1),
    )
    x = torch.zeros(1, 4, 6)
    x[:, 0, :] = 2.0
    x[:, 1, :] = 5.0
    # delta = (5 - 2) * 6 = 18
    torch.testing.assert_close(target(x), torch.tensor([18.0]))


def test_attribution_target_delta_profile_window_sum_partial_window() -> None:
    model = _MultiConditionModel()
    target = AttributionTarget(
        model=model,
        reduction="delta_profile_window_sum",
        channels=(0, 1),
        window_start=2,
        window_end=5,
    )
    x = torch.zeros(1, 4, 6)
    x[:, 0, :] = 1.0
    x[:, 1, :] = 4.0
    # delta over [2,5) = (4-1)*3 = 9
    torch.testing.assert_close(target(x), torch.tensor([9.0]))


def test_attribution_target_delta_invalid_window_raises() -> None:
    model = _MultiConditionModel()
    target = AttributionTarget(
        model=model,
        reduction="delta_profile_window_sum",
        channels=(0, 1),
        window_start=10,
        window_end=5,
    )
    x = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError, match="Invalid window"):
        target(x)


@pytest.mark.parametrize(
    "cond_a_idx,cond_b_idx",
    [(-1, 1), (0, 5), (2, 0)],
)
def test_attribution_target_delta_out_of_range_channel_raises(
    cond_a_idx: int, cond_b_idx: int
) -> None:
    model = _MultiConditionModel()  # has 2 condition channels
    target = AttributionTarget(
        model=model,
        reduction="delta_log_counts",
        channels=(cond_a_idx, cond_b_idx),
    )
    x = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError, match="out of range"):
        target(x)


def test_attribution_target_delta_gradient_flows() -> None:
    """Attribution-relevant check: delta target must be differentiable wrt input."""
    model = _MultiConditionModel()
    target = AttributionTarget(
        model=model, reduction="delta_log_counts", channels=(0, 1)
    )
    x = torch.randn(2, 4, 8, requires_grad=True)
    out = target(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_attribution_target_delta_ism_matches_manual_delta() -> None:
    """For linear models, ISM on delta target == manual ISM(B) - ISM(A)."""
    model = _MultiConditionModel()
    delta_target = AttributionTarget(
        model=model, reduction="delta_log_counts", channels=(0, 1)
    )
    # Per-channel targets for manual difference
    target_a = AttributionTarget(
        model=model,
        reduction="log_counts",
        channels=0,
    )
    target_b = AttributionTarget(
        model=model,
        reduction="log_counts",
        channels=1,
    )

    x = torch.zeros(1, 4, 8)
    x[:, 0, :3] = 1.0  # A,C,G,T one-hot preamble
    x[:, 1, 3:6] = 1.0
    x[:, 2, 6:] = 1.0

    attrs_delta = compute_ism_attributions(delta_target, x, span=(2, 6))
    attrs_a = compute_ism_attributions(target_a, x, span=(2, 6))
    attrs_b = compute_ism_attributions(target_b, x, span=(2, 6))

    # For a linear model and ISM, attr(f_B - f_A) == attr(f_B) - attr(f_A).
    # Both sides will have had _apply_tf_modisco_ref_override applied, so
    # compare the span region after removing the ref-channel override (which
    # depends on non-linear selection of mean). Easiest: compare raw delta by
    # summing across the nucleotide axis — the mean-override keeps sum zero.
    torch.testing.assert_close(
        attrs_delta.sum(dim=1), (attrs_b - attrs_a).sum(dim=1), atol=1e-5, rtol=1e-5
    )
