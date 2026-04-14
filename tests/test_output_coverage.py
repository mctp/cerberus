"""Coverage tests for cerberus.output — untested code paths."""

import pytest
import torch

from cerberus.interval import Interval
from cerberus.output import (
    ModelOutput,
    ProfileCountOutput,
    ProfileLogits,
    ProfileLogRates,
    aggregate_models,
    compute_channel_log_counts,
    compute_profile_probs,
    compute_signal,
    compute_total_log_counts,
    unbatch_modeloutput,
)

# ---------------------------------------------------------------------------
# compute_total_log_counts
# ---------------------------------------------------------------------------


class TestComputeTotalLogCountsProfileLogRates:
    """Tests for compute_total_log_counts with ProfileLogRates input."""

    def test_single_channel(self):
        """Single-channel ProfileLogRates: logsumexp over length."""
        torch.manual_seed(0)
        log_rates = torch.randn(2, 1, 10)
        output = ProfileLogRates(log_rates=log_rates)
        result = compute_total_log_counts(output)
        assert result.shape == (2,)
        # For 1 channel: logsumexp over (channel=1, length=10) dims -> scalar per batch
        expected = torch.logsumexp(log_rates.float(), dim=(1, 2)).flatten()
        torch.testing.assert_close(result, expected)

    def test_multi_channel(self):
        """Multi-channel ProfileLogRates: flatten channels+length then logsumexp."""
        torch.manual_seed(1)
        log_rates = torch.randn(3, 4, 8)
        output = ProfileLogRates(log_rates=log_rates)
        result = compute_total_log_counts(output)
        assert result.shape == (3,)
        expected = torch.logsumexp(log_rates.float().flatten(start_dim=1), dim=-1)
        torch.testing.assert_close(result, expected)


class TestComputeTotalLogCountsPseudocount:
    """Tests for compute_total_log_counts with log_counts_include_pseudocount."""

    def test_multi_channel_with_pseudocount(self):
        """Multi-channel ProfileCountOutput with log_counts_include_pseudocount=True."""
        torch.manual_seed(2)
        logits = torch.randn(2, 3, 10)
        # Simulate log_counts in log(count + 1) space per channel
        counts_per_channel = torch.tensor([[100.0, 200.0, 50.0], [30.0, 60.0, 10.0]])
        log_counts = torch.log(counts_per_channel + 1.0)
        output = ProfileCountOutput(logits=logits, log_counts=log_counts)

        result = compute_total_log_counts(
            output, log_counts_include_pseudocount=True, pseudocount=1.0
        )
        assert result.shape == (2,)

        # Manual: undo log(x+1) -> x, sum channels, reapply log(total+1)
        inverted = (torch.exp(log_counts.float()) - 1.0).clamp_min(0.0)
        total = inverted.sum(dim=1)
        expected = torch.log(total + 1.0)
        torch.testing.assert_close(result, expected)

    def test_single_channel_returns_flattened(self):
        """Single-channel ProfileCountOutput returns flattened regardless of pseudocount flag."""
        log_counts = torch.tensor([[5.0]])
        output = ProfileCountOutput(logits=torch.randn(1, 1, 10), log_counts=log_counts)
        result = compute_total_log_counts(output, log_counts_include_pseudocount=True)
        assert result.shape == (1,)
        assert result.item() == pytest.approx(5.0)

    def test_multi_channel_without_pseudocount_uses_logsumexp(self):
        """Multi-channel ProfileCountOutput without pseudocount uses logsumexp."""
        torch.manual_seed(3)
        log_counts = torch.randn(2, 3)
        output = ProfileCountOutput(logits=torch.randn(2, 3, 10), log_counts=log_counts)
        result = compute_total_log_counts(output, log_counts_include_pseudocount=False)
        expected = torch.logsumexp(log_counts.float(), dim=1)
        torch.testing.assert_close(result, expected)


class TestComputeTotalLogCountsUnsupported:
    """compute_total_log_counts should raise ValueError for unsupported types."""

    def test_plain_model_output_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            compute_total_log_counts(ProfileLogits(logits=torch.randn(1, 1, 10)))


# ---------------------------------------------------------------------------
# aggregate_models
# ---------------------------------------------------------------------------


class TestAggregateModels:
    def test_median_method(self):
        """aggregate_models with 'median' computes element-wise median."""
        torch.manual_seed(4)
        outputs = [
            ProfileLogits(logits=torch.tensor([[[1.0, 2.0, 3.0]]])),
            ProfileLogits(logits=torch.tensor([[[4.0, 5.0, 6.0]]])),
            ProfileLogits(logits=torch.tensor([[[7.0, 8.0, 9.0]]])),
        ]
        result = aggregate_models(outputs, method="median")
        assert isinstance(result, ProfileLogits)
        # Median of [1,4,7]=4, [2,5,8]=5, [3,6,9]=6
        expected = torch.tensor([[[4.0, 5.0, 6.0]]])
        torch.testing.assert_close(result.logits, expected)

    def test_mean_method(self):
        """aggregate_models with 'mean' computes element-wise mean."""
        outputs = [
            ProfileLogits(logits=torch.tensor([[[1.0, 2.0]]])),
            ProfileLogits(logits=torch.tensor([[[3.0, 4.0]]])),
        ]
        result = aggregate_models(outputs, method="mean")
        assert isinstance(result, ProfileLogits)
        expected = torch.tensor([[[2.0, 3.0]]])
        torch.testing.assert_close(result.logits, expected)

    def test_invalid_method_raises(self):
        """aggregate_models with unknown method raises ValueError."""
        outputs = [ProfileLogits(logits=torch.randn(1, 1, 5))]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_models(outputs, method="bogus")

    def test_preserves_out_interval(self):
        """aggregate_models preserves out_interval from the first output."""
        iv = Interval("chr1", 100, 200)
        outputs = [
            ProfileLogits(logits=torch.randn(1, 1, 5), out_interval=iv),
            ProfileLogits(logits=torch.randn(1, 1, 5), out_interval=None),
        ]
        result = aggregate_models(outputs, method="mean")
        assert isinstance(result.out_interval, Interval)
        assert result.out_interval == iv

    # -- Masked aggregation ---------------------------------------------------

    def test_masked_disjoint_single_field(self):
        """Disjoint masks with ProfileLogits: each sample gets its own model."""
        out0 = ProfileLogits(logits=torch.tensor([[[10.0, 10.0]], [[0.0, 0.0]]]))
        out1 = ProfileLogits(logits=torch.tensor([[[0.0, 0.0]], [[20.0, 20.0]]]))
        masks = [torch.tensor([True, False]), torch.tensor([False, True])]

        result = aggregate_models([out0, out1], method="mean", masks=masks)
        assert isinstance(result, ProfileLogits)
        torch.testing.assert_close(
            result.logits,
            torch.tensor([[[10.0, 10.0]], [[20.0, 20.0]]]),
        )

    def test_masked_multi_field_output(self):
        """Masks work correctly across fields with different shapes.

        ProfileCountOutput has logits (B, C, L) and log_counts (B, C) —
        the broadcast must adapt per-field.
        """
        out0 = ProfileCountOutput(
            logits=torch.tensor([[[5.0, 5.0]], [[0.0, 0.0]]]),
            log_counts=torch.tensor([[3.0], [0.0]]),
        )
        out1 = ProfileCountOutput(
            logits=torch.tensor([[[0.0, 0.0]], [[7.0, 7.0]]]),
            log_counts=torch.tensor([[0.0], [4.0]]),
        )
        masks = [torch.tensor([True, False]), torch.tensor([False, True])]

        result = aggregate_models([out0, out1], method="mean", masks=masks)
        assert isinstance(result, ProfileCountOutput)
        # logits: sample 0 from model 0, sample 1 from model 1
        torch.testing.assert_close(
            result.logits,
            torch.tensor([[[5.0, 5.0]], [[7.0, 7.0]]]),
        )
        # log_counts: same routing
        torch.testing.assert_close(
            result.log_counts,
            torch.tensor([[3.0], [4.0]]),
        )

    def test_masked_overlapping(self):
        """When both models contribute, result is the mean."""
        out0 = ProfileLogits(logits=torch.tensor([[[10.0]]]))
        out1 = ProfileLogits(logits=torch.tensor([[[30.0]]]))
        masks = [torch.tensor([True]), torch.tensor([True])]

        result = aggregate_models([out0, out1], method="mean", masks=masks)
        assert isinstance(result, ProfileLogits)
        torch.testing.assert_close(result.logits, torch.tensor([[[20.0]]]))

    def test_masked_partial_overlap(self):
        """Sample 0 seen by both models, sample 1 by model 1 only."""
        out0 = ProfileLogits(logits=torch.tensor([[[10.0]], [[0.0]]]))
        out1 = ProfileLogits(logits=torch.tensor([[[20.0]], [[30.0]]]))
        masks = [torch.tensor([True, False]), torch.tensor([True, True])]

        result = aggregate_models([out0, out1], method="mean", masks=masks)
        assert isinstance(result, ProfileLogits)
        # Sample 0: mean(10, 20) = 15; Sample 1: 30/1 = 30
        torch.testing.assert_close(result.logits, torch.tensor([[[15.0]], [[30.0]]]))

    def test_masked_none_is_identity(self):
        """masks=None gives identical results to omitting the parameter."""
        outputs = [
            ProfileLogits(logits=torch.tensor([[[1.0, 2.0]]])),
            ProfileLogits(logits=torch.tensor([[[3.0, 4.0]]])),
        ]
        with_none = aggregate_models(outputs, method="mean", masks=None)
        without = aggregate_models(outputs, method="mean")
        assert isinstance(with_none, ProfileLogits)
        assert isinstance(without, ProfileLogits)
        torch.testing.assert_close(with_none.logits, without.logits)

    def test_masked_median_raises(self):
        """Masked aggregation rejects method='median'."""
        outputs = [ProfileLogits(logits=torch.randn(1, 1, 5))]
        masks = [torch.tensor([True])]
        with pytest.raises(ValueError, match="Masked aggregation only supports 'mean'"):
            aggregate_models(outputs, method="median", masks=masks)

    def test_masked_3_models(self):
        """Three models with non-uniform coverage per sample."""
        # 3 samples, 3 models
        # Sample 0: models 0, 1      → mean(10, 20) = 15
        # Sample 1: model 2 only     → 60
        # Sample 2: models 0, 1, 2   → mean(10, 20, 60) = 30
        out0 = ProfileLogits(logits=torch.tensor([[[10.0]], [[0.0]], [[10.0]]]))
        out1 = ProfileLogits(logits=torch.tensor([[[20.0]], [[0.0]], [[20.0]]]))
        out2 = ProfileLogits(logits=torch.tensor([[[0.0]], [[60.0]], [[60.0]]]))
        masks = [
            torch.tensor([True, False, True]),
            torch.tensor([True, False, True]),
            torch.tensor([False, True, True]),
        ]

        result = aggregate_models([out0, out1, out2], method="mean", masks=masks)
        assert isinstance(result, ProfileLogits)
        expected = torch.tensor([[[15.0]], [[60.0]], [[30.0]]])
        torch.testing.assert_close(result.logits, expected)


# ---------------------------------------------------------------------------
# unbatch_modeloutput
# ---------------------------------------------------------------------------


class TestUnbatchModelOutput:
    def test_metadata_replication(self):
        """out_interval (non-tensor) is replicated across all unbatched items."""
        iv = Interval("chr1", 0, 1000)
        batched = ProfileLogits(logits=torch.randn(3, 2, 10), out_interval=iv)
        items = unbatch_modeloutput(batched, batch_size=3)
        assert len(items) == 3
        for item in items:
            assert item["logits"].shape == (2, 10)
            assert isinstance(item["out_interval"], Interval)
            assert item["out_interval"].chrom == "chr1"
            assert item["out_interval"].start == 0
            assert item["out_interval"].end == 1000

    def test_preserves_interval_type(self):
        """unbatch_modeloutput must return Interval objects, not dicts."""
        iv = Interval("chr7", 500, 1500, "-")
        batched = ProfileCountOutput(
            logits=torch.randn(2, 1, 10),
            log_counts=torch.randn(2, 1),
            out_interval=iv,
        )
        items = unbatch_modeloutput(batched, batch_size=2)
        for item in items:
            out_iv = item["out_interval"]
            assert isinstance(out_iv, Interval), (
                f"Expected Interval, got {type(out_iv).__name__}"
            )
            assert out_iv.chrom == "chr7"
            assert out_iv.strand == "-"

    def test_no_tensor_deepcopy(self):
        """unbatch_modeloutput should not deep-copy tensors (shares storage)."""
        logits = torch.randn(2, 1, 10)
        batched = ProfileLogits(logits=logits, out_interval=None)
        items = unbatch_modeloutput(batched, batch_size=2)
        # unbind creates views that share storage with the original
        assert (
            items[0]["logits"].untyped_storage().data_ptr()
            == logits.untyped_storage().data_ptr()
        )

    def test_none_interval(self):
        """out_interval=None is replicated as None."""
        batched = ProfileCountOutput(
            logits=torch.randn(2, 1, 5),
            log_counts=torch.randn(2, 1),
        )
        items = unbatch_modeloutput(batched, batch_size=2)
        assert len(items) == 2
        for item in items:
            assert item["out_interval"] is None
            assert "logits" in item
            assert "log_counts" in item


# ---------------------------------------------------------------------------
# ModelOutput.detach() and subclasses
# ---------------------------------------------------------------------------


class TestDetach:
    def test_base_model_output_detach_raises(self):
        """ModelOutput.detach() raises NotImplementedError."""
        base = ModelOutput()
        with pytest.raises(NotImplementedError):
            base.detach()

    def test_profile_logits_detach(self):
        logits = torch.randn(2, 1, 10, requires_grad=True)
        out = ProfileLogits(logits=logits)
        detached = out.detach()
        assert not detached.logits.requires_grad

    def test_profile_log_rates_detach(self):
        log_rates = torch.randn(2, 1, 10, requires_grad=True)
        out = ProfileLogRates(log_rates=log_rates)
        detached = out.detach()
        assert not detached.log_rates.requires_grad
        assert isinstance(detached, ProfileLogRates)

    def test_profile_count_output_detach(self):
        logits = torch.randn(2, 1, 10, requires_grad=True)
        log_counts = torch.randn(2, 1, requires_grad=True)
        out = ProfileCountOutput(logits=logits, log_counts=log_counts)
        detached = out.detach()
        assert not detached.logits.requires_grad
        assert not detached.log_counts.requires_grad
        assert isinstance(detached, ProfileCountOutput)

    def test_detach_preserves_interval(self):
        iv = Interval("chr1", 100, 200)
        out = ProfileLogRates(log_rates=torch.randn(1, 1, 5), out_interval=iv)
        detached = out.detach()
        assert detached.out_interval == iv


# ---------------------------------------------------------------------------
# compute_signal
# ---------------------------------------------------------------------------


class TestComputePredictedSignal:
    def test_profile_count_batched_shape(self):
        """ProfileCountOutput batched: returns (B, C, L)."""
        out = ProfileCountOutput(
            logits=torch.randn(2, 3, 100),
            log_counts=torch.ones(2, 3),
        )
        signal = compute_signal(out)
        assert signal.shape == (2, 3, 100)

    def test_profile_count_unbatched_shape(self):
        """ProfileCountOutput unbatched: returns (C, L)."""
        out = ProfileCountOutput(
            logits=torch.randn(3, 100),
            log_counts=torch.ones(3),
        )
        signal = compute_signal(out)
        assert signal.shape == (3, 100)

    def test_profile_count_non_negative(self):
        """Reconstructed signal is non-negative."""
        out = ProfileCountOutput(
            logits=torch.randn(2, 1, 50),
            log_counts=torch.tensor([[3.0]]),
        )
        signal = compute_signal(out)
        assert (signal >= 0).all()

    def test_profile_count_sums_to_total(self):
        """Signal sums approximately to exp(log_counts) per channel."""
        log_c = torch.tensor([[4.0]])
        out = ProfileCountOutput(
            logits=torch.randn(1, 1, 100),
            log_counts=log_c,
        )
        signal = compute_signal(out)
        expected_total = torch.exp(log_c).item()
        assert signal.sum().item() == pytest.approx(expected_total, rel=1e-4)

    def test_profile_count_with_pseudocount(self):
        """With pseudocount, total is exp(log_counts) - pseudocount."""
        log_c = torch.tensor([[4.0]])
        out = ProfileCountOutput(
            logits=torch.randn(1, 1, 100),
            log_counts=log_c,
        )
        signal = compute_signal(
            out, log_counts_include_pseudocount=True, pseudocount=1.0
        )
        expected_total = (torch.exp(log_c) - 1.0).item()
        assert signal.sum().item() == pytest.approx(expected_total, rel=1e-4)

    def test_profile_count_pseudocount_clamped(self):
        """Pseudocount larger than exp(log_counts) clamps to zero."""
        log_c = torch.tensor([[0.0]])  # exp(0) = 1.0
        out = ProfileCountOutput(
            logits=torch.randn(1, 1, 50),
            log_counts=log_c,
        )
        signal = compute_signal(
            out, log_counts_include_pseudocount=True, pseudocount=2.0
        )
        assert signal.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_profile_log_rates_shape(self):
        """ProfileLogRates: returns exp(log_rates)."""
        log_rates = torch.randn(2, 1, 50)
        out = ProfileLogRates(log_rates=log_rates)
        signal = compute_signal(out)
        assert signal.shape == (2, 1, 50)
        assert torch.allclose(signal, torch.exp(log_rates))

    def test_profile_logits_fallback(self):
        """ProfileLogits fallback: returns raw logits with warning."""
        logits = torch.randn(2, 1, 50)
        out = ProfileLogits(logits=logits)
        signal = compute_signal(out)
        assert torch.allclose(signal, logits)

    def test_global_log_counts_multichannel(self):
        """Global log_counts (B, 1) with multi-channel logits (B, C, L)."""
        out = ProfileCountOutput(
            logits=torch.randn(1, 4, 100),
            log_counts=torch.tensor([[5.0]]),
        )
        signal = compute_signal(out)
        assert signal.shape == (1, 4, 100)
        # Total should be exp(5.0) distributed across 4 channels
        expected_per_channel = torch.exp(torch.tensor(5.0)).item() / 4
        for c in range(4):
            assert signal[0, c].sum().item() == pytest.approx(
                expected_per_channel, rel=1e-4
            )


# ---------------------------------------------------------------------------
# compute_profile_probs
# ---------------------------------------------------------------------------


class TestComputeProfileProbs:
    def test_profile_count_shape(self):
        out = ProfileCountOutput(
            logits=torch.randn(2, 3, 100), log_counts=torch.ones(2, 3)
        )
        probs = compute_profile_probs(out)
        assert probs.shape == (2, 3, 100)

    def test_sums_to_one(self):
        out = ProfileCountOutput(
            logits=torch.randn(4, 2, 50), log_counts=torch.ones(4, 2)
        )
        probs = compute_profile_probs(out)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_non_negative(self):
        out = ProfileCountOutput(
            logits=torch.randn(2, 1, 100), log_counts=torch.ones(2, 1)
        )
        probs = compute_profile_probs(out)
        assert (probs >= 0).all()

    def test_unbatched(self):
        out = ProfileCountOutput(logits=torch.randn(3, 100), log_counts=torch.ones(3))
        probs = compute_profile_probs(out)
        assert probs.shape == (3, 100)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_profile_logits(self):
        """ProfileLogits also works (parent class of ProfileCountOutput)."""
        out = ProfileLogits(logits=torch.randn(2, 1, 50))
        probs = compute_profile_probs(out)
        assert probs.shape == (2, 1, 50)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 1), atol=1e-5)

    def test_profile_log_rates(self):
        """ProfileLogRates: normalized exp(log_rates)."""
        log_rates = torch.randn(2, 1, 50)
        out = ProfileLogRates(log_rates=log_rates)
        probs = compute_profile_probs(out)
        assert probs.shape == (2, 1, 50)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 1), atol=1e-5)

    def test_uniform_logits(self):
        """Uniform logits → uniform probabilities."""
        out = ProfileCountOutput(
            logits=torch.zeros(1, 1, 10), log_counts=torch.ones(1, 1)
        )
        probs = compute_profile_probs(out)
        assert torch.allclose(probs, torch.full_like(probs, 0.1), atol=1e-5)

    def test_spike_logits(self):
        """Large logit at one position → probability concentrated there."""
        logits = torch.full((1, 1, 100), -100.0)
        logits[0, 0, 42] = 0.0
        out = ProfileCountOutput(logits=logits, log_counts=torch.ones(1, 1))
        probs = compute_profile_probs(out)
        assert probs[0, 0, 42].item() > 0.99


# ---------------------------------------------------------------------------
# compute_channel_log_counts
# ---------------------------------------------------------------------------


class TestComputeChannelLogCounts:
    def test_profile_count_passthrough(self):
        """Per-channel log_counts are returned directly."""
        log_c = torch.tensor([[2.0, 3.0, 4.0]])
        out = ProfileCountOutput(logits=torch.randn(1, 3, 50), log_counts=log_c)
        result = compute_channel_log_counts(out)
        assert torch.allclose(result, log_c)

    def test_shape_batched(self):
        out = ProfileCountOutput(
            logits=torch.randn(4, 3, 50), log_counts=torch.randn(4, 3)
        )
        result = compute_channel_log_counts(out)
        assert result.shape == (4, 3)

    def test_shape_unbatched(self):
        out = ProfileCountOutput(logits=torch.randn(3, 50), log_counts=torch.randn(3))
        result = compute_channel_log_counts(out)
        assert result.shape == (3,)

    def test_global_log_counts_distributed(self):
        """Global (B,1) log_counts with C>1 channels → log(total/C) per channel."""
        import math

        log_c = torch.tensor([[6.0]])  # total = exp(6)
        out = ProfileCountOutput(logits=torch.randn(1, 4, 50), log_counts=log_c)
        result = compute_channel_log_counts(out)
        assert result.shape == (1, 4)
        expected = 6.0 - math.log(4)
        assert result[0, 0].item() == pytest.approx(expected, rel=1e-5)
        assert result[0, 3].item() == pytest.approx(expected, rel=1e-5)

    def test_single_channel_unchanged(self):
        """Single channel: no distribution needed."""
        log_c = torch.tensor([[5.0]])
        out = ProfileCountOutput(logits=torch.randn(1, 1, 50), log_counts=log_c)
        result = compute_channel_log_counts(out)
        assert result.item() == pytest.approx(5.0, rel=1e-5)

    def test_profile_log_rates(self):
        """ProfileLogRates: logsumexp along length axis."""
        log_rates = torch.zeros(1, 2, 100)  # rates = 1.0 everywhere
        out = ProfileLogRates(log_rates=log_rates)
        result = compute_channel_log_counts(out)
        assert result.shape == (1, 2)
        # logsumexp of 100 zeros = log(100)
        import math

        assert result[0, 0].item() == pytest.approx(math.log(100), rel=1e-4)

    def test_consistency_with_total(self):
        """Sum of exp(channel_log_counts) ≈ exp(total_log_counts)."""
        out = ProfileCountOutput(
            logits=torch.randn(2, 3, 50),
            log_counts=torch.randn(2, 3),
        )
        channel = compute_channel_log_counts(out)
        total = compute_total_log_counts(out)
        channel_sum = torch.exp(channel).sum(dim=-1)
        total_linear = torch.exp(total)
        assert torch.allclose(channel_sum, total_linear, rtol=1e-4)
