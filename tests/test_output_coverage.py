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
        counts_per_channel = torch.tensor([[100.0, 200.0, 50.0],
                                            [30.0, 60.0, 10.0]])
        log_counts = torch.log(counts_per_channel + 1.0)
        output = ProfileCountOutput(logits=logits, log_counts=log_counts)

        result = compute_total_log_counts(output, log_counts_include_pseudocount=True, pseudocount=1.0)
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
        assert items[0]["logits"].untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()

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
