"""Tests for cerberus.predict_bigwig — BigWig prediction generation."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ProfileCountOutput, ProfileLogRates
from cerberus.predict_bigwig import (
    _process_island,
    _reconstruct_linear_signal,
    predict_to_bigwig,
)

# ---------------------------------------------------------------------------
# _reconstruct_linear_signal
# ---------------------------------------------------------------------------


class TestReconstructLinearSignal:
    """Unit tests for per-window signal reconstruction."""

    def test_profile_count_output_uniform(self):
        """Uniform logits + known log_counts → uniform signal at total count level."""
        L = 100
        total_count = 200.0
        pseudocount = 1.0
        logits = torch.zeros(1, L)  # uniform → each bin gets 1/L of count
        log_counts = torch.tensor([math.log(total_count + pseudocount)])

        output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        signal = _reconstruct_linear_signal(output, count_pseudocount=pseudocount)

        assert signal.shape == (1, L)
        expected_per_bin = total_count / L
        assert torch.allclose(signal, torch.full((1, L), expected_per_bin), atol=1e-4)

    def test_profile_count_output_zero_pseudocount(self):
        """With pseudocount=0, no offset is subtracted."""
        L = 10
        total_count = 50.0
        logits = torch.zeros(1, L)
        log_counts = torch.tensor([math.log(total_count)])

        output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        signal = _reconstruct_linear_signal(output, count_pseudocount=0.0)

        expected = total_count / L
        assert torch.allclose(signal, torch.full((1, L), expected), atol=1e-4)

    def test_profile_count_output_sums_to_total(self):
        """Signal bins should sum to total_count (within float tolerance)."""
        L = 50
        total_count = 300.0
        pseudocount = 1.0
        logits = torch.randn(2, L)  # non-uniform
        log_counts = torch.tensor([
            math.log(total_count + pseudocount),
            math.log(total_count + pseudocount),
        ])

        output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        signal = _reconstruct_linear_signal(output, count_pseudocount=pseudocount)

        assert signal.shape == (2, L)
        for c in range(2):
            assert abs(signal[c].sum().item() - total_count) < 0.5

    def test_profile_count_output_clamps_negative_counts(self):
        """If exp(log_counts) - pseudocount < 0, clamp to 0."""
        logits = torch.zeros(1, 10)
        # log_counts such that exp(lc) < pseudocount
        log_counts = torch.tensor([math.log(0.5)])  # exp = 0.5, minus 1.0 = -0.5

        output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        signal = _reconstruct_linear_signal(output, count_pseudocount=1.0)

        assert (signal >= 0).all()
        assert signal.sum().item() == 0.0

    def test_profile_log_rates(self):
        """ProfileLogRates → exp(log_rates)."""
        L = 20
        log_rates = torch.full((1, L), math.log(5.0))

        output = ProfileLogRates(log_rates=log_rates)
        signal = _reconstruct_linear_signal(output, count_pseudocount=0.0)

        assert signal.shape == (1, L)
        assert torch.allclose(signal, torch.full((1, L), 5.0), atol=1e-4)

    def test_fallback_logits_only(self):
        """Output with logits but no log_counts or log_rates → raw logits."""
        from dataclasses import dataclass

        from cerberus.output import ModelOutput

        @dataclass
        class LogitsOnly(ModelOutput):
            logits: torch.Tensor
            def detach(self):
                return self

        output = LogitsOnly(logits=torch.ones(1, 10))
        signal = _reconstruct_linear_signal(output, count_pseudocount=0.0)
        assert torch.equal(signal, torch.ones(1, 10))

    def test_unsupported_output_raises(self):
        from dataclasses import dataclass

        from cerberus.output import ModelOutput

        @dataclass
        class Weird(ModelOutput):
            data: torch.Tensor
            def detach(self):
                return self

        output = Weird(data=torch.zeros(1, 10))
        with pytest.raises(ValueError, match="Cannot extract profile track"):
            _reconstruct_linear_signal(output, count_pseudocount=0.0)

    def test_multichannel_profile_count(self):
        """Multi-channel output reconstructs each channel independently."""
        L = 20
        C = 3
        logits = torch.zeros(C, L)
        counts = torch.tensor([math.log(100.0), math.log(200.0), math.log(300.0)])

        output = ProfileCountOutput(logits=logits, log_counts=counts)
        signal = _reconstruct_linear_signal(output, count_pseudocount=0.0)

        assert signal.shape == (C, L)
        assert abs(signal[0].sum().item() - 100.0) < 0.5
        assert abs(signal[1].sum().item() - 200.0) < 0.5
        assert abs(signal[2].sum().item() - 300.0) < 0.5


# ---------------------------------------------------------------------------
# _process_island — island processing with mock ensemble
# ---------------------------------------------------------------------------


def _make_mock_ensemble_and_dataset(
    input_len=100,
    output_len=50,
    output_bin_size=1,
    target_scale=1.0,
    count_pseudocount=1.0,
):
    """Create lightweight mocks of ModelEnsemble and CerberusDataset."""
    data_config = DataConfig.model_construct(
        inputs={},
        targets={"signal": Path("signal.bw")},
        input_len=input_len,
        output_len=output_len,
        output_bin_size=output_bin_size,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=target_scale,
        use_sequence=True,
    )

    genome_config = GenomeConfig.model_construct(
        name="test",
        fasta_path=Path("test.fa"),
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 10000},
        fold_type="chrom_partition",
        fold_args={"k": 2},
    )

    dataset = MagicMock(spec=CerberusDataset)
    dataset.data_config = data_config
    dataset.genome_config = genome_config
    dataset.exclude_intervals = {}

    model_config = ModelConfig.model_construct(
        name="test",
        model_cls="test",
        loss_cls="cerberus.loss.MSEMultinomialLoss",
        loss_args={"count_pseudocount": count_pseudocount},
        metrics_cls="test",
        metrics_args={},
        model_args={},
        pretrained=[],
        count_pseudocount=count_pseudocount,
    )

    ensemble = MagicMock(spec=ModelEnsemble)
    cerberus_config = CerberusConfig.model_construct(
        genome_config=genome_config,
        data_config=data_config,
        model_config_=model_config,
        sampler_config=SamplerConfig.model_construct(
            sampler_type="interval", padded_size=input_len, sampler_args={},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1, max_epochs=1, learning_rate=1e-3,
            weight_decay=0.0, patience=1, optimizer="adam",
            scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=False,
            reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8, gradient_clip_val=None,
        ),
    )
    ensemble.cerberus_config = cerberus_config

    return dataset, ensemble


class TestProcessIsland:
    """Tests for the _process_island helper."""

    def test_empty_island_returns_nothing(self):
        dataset, ensemble = _make_mock_ensemble_and_dataset()
        result = list(_process_island([], dataset, ensemble, ["test"], 64, 1.0))
        assert result == []

    def test_single_window_yields_correct_bins(self):
        """A single window should produce output_len / output_bin_size bins."""
        input_len = 100
        output_len = 50
        output_bin_size = 1
        n_bins = output_len // output_bin_size
        count_pseudocount = 1.0
        total_count = 100.0

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            count_pseudocount=count_pseudocount,
        )

        # Mock predict_intervals_batched to yield a single batch
        logits = torch.zeros(1, 1, n_bins)  # uniform profile
        log_counts = torch.full((1, 1), math.log(total_count + count_pseudocount))
        batch_output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        interval = Interval("chr1", 500, 600, "+")
        ensemble.predict_intervals_batched.return_value = iter(
            [(batch_output, [interval])]
        )

        result = list(
            _process_island(
                [interval], dataset, ensemble, ["test"], 64, count_pseudocount,
            )
        )

        assert len(result) == n_bins
        # Each bin should be (chrom, start, end, value)
        for chrom, start, end, _val in result:
            assert chrom == "chr1"
            assert end - start == output_bin_size

        # Values should be positive (signal was reconstructed)
        values = [v for _, _, _, v in result]
        assert all(v >= 0 for v in values)

    def test_target_scale_undo(self):
        """Output values should be divided by target_scale * output_bin_size."""
        input_len = 100
        output_len = 50
        output_bin_size = 5
        n_bins = output_len // output_bin_size  # 10
        target_scale = 2.0
        count_pseudocount = 0.0
        total_count = 100.0

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            target_scale=target_scale,
            count_pseudocount=count_pseudocount,
        )

        logits = torch.zeros(1, 1, n_bins)
        log_counts = torch.full((1, 1), math.log(total_count))
        batch_output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        interval = Interval("chr1", 500, 600, "+")
        ensemble.predict_intervals_batched.return_value = iter(
            [(batch_output, [interval])]
        )

        result = list(
            _process_island(
                [interval], dataset, ensemble, ["test"], 64, count_pseudocount,
            )
        )

        # Uniform profile: each bin gets total_count / n_bins
        # Then divided by target_scale * output_bin_size
        raw_per_bin = total_count / n_bins
        expected_per_bin = raw_per_bin / target_scale / output_bin_size
        actual_values = [v for _, _, _, v in result]
        assert all(abs(v - expected_per_bin) < 0.1 for v in actual_values)

    def test_overlapping_windows_averaged(self):
        """Two overlapping windows should have their overlap averaged."""
        input_len = 100
        output_len = 50
        output_bin_size = 1
        n_bins = 50
        count_pseudocount = 0.0
        total_count = 50.0  # 1.0 per bin for uniform

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            count_pseudocount=count_pseudocount,
        )

        # Two windows, stride 25 → overlap of 25 bins
        iv1 = Interval("chr1", 500, 600, "+")
        iv2 = Interval("chr1", 525, 625, "+")

        logits1 = torch.zeros(1, 1, n_bins)  # uniform
        lc1 = torch.full((1, 1), math.log(total_count))
        logits2 = torch.zeros(1, 1, n_bins)
        lc2 = torch.full((1, 1), math.log(total_count))

        # Yield two batches (one per window)
        batch1 = ProfileCountOutput(logits=logits1, log_counts=lc1)
        batch2 = ProfileCountOutput(logits=logits2, log_counts=lc2)
        ensemble.predict_intervals_batched.return_value = iter([
            (batch1, [iv1]),
            (batch2, [iv2]),
        ])

        result = list(
            _process_island(
                [iv1, iv2], dataset, ensemble, ["test"], 64, count_pseudocount,
            )
        )

        # Total span: output of iv1 starts at 525 (500+25), output of iv2 at 550 (525+25)
        # iv1 output: 525-575, iv2 output: 550-600
        # Merged: 525-600 = 75 bins
        assert len(result) == 75

        # Non-overlap regions should have value = 1.0 per bin (50 count / 50 bins)
        # Overlap region (25 bins) averaged: (1.0 + 1.0) / 2 = 1.0
        # All values should be ~1.0 before target_scale/bin_size division
        # With target_scale=1.0, bin_size=1: value = 1.0
        values = [v for _, _, _, v in result]
        assert all(abs(v - 1.0) < 0.1 for v in values)

    def test_multichannel_uses_first_channel(self):
        """Only channel 0 should appear in the BigWig output tuples."""
        input_len = 100
        output_len = 50
        output_bin_size = 1
        n_bins = 50
        count_pseudocount = 0.0

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            count_pseudocount=count_pseudocount,
        )

        # 2 channels, channel 0 = 10, channel 1 = 999
        logits = torch.zeros(1, 2, n_bins)
        log_counts = torch.tensor([[math.log(10.0 * n_bins), math.log(999.0 * n_bins)]])
        batch_output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        interval = Interval("chr1", 500, 600, "+")
        ensemble.predict_intervals_batched.return_value = iter(
            [(batch_output, [interval])]
        )

        result = list(
            _process_island(
                [interval], dataset, ensemble, ["test"], 64, count_pseudocount,
            )
        )

        values = [v for _, _, _, v in result]
        # Channel 0 has ~10.0 per bin; channel 1 (999) should not appear
        assert all(abs(v - 10.0) < 0.5 for v in values)


# ---------------------------------------------------------------------------
# predict_to_bigwig — end-to-end with mock BigWig write
# ---------------------------------------------------------------------------


class TestPredictToBigwig:
    """Integration tests for predict_to_bigwig with mocked I/O."""

    def test_writes_bigwig_file(self, tmp_path):
        """Verify predict_to_bigwig calls pybigtools.open and writes data."""
        input_len = 100
        output_len = 50
        n_bins = 50
        count_pseudocount = 1.0

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            count_pseudocount=count_pseudocount,
        )

        # predict_intervals_batched returns one batch per chrom
        logits = torch.zeros(1, 1, n_bins)
        log_counts = torch.full((1, 1), math.log(100.0 + count_pseudocount))
        batch_output = ProfileCountOutput(logits=logits, log_counts=log_counts)
        interval = Interval("chr1", 0, input_len, "+")
        ensemble.predict_intervals_batched.return_value = iter(
            [(batch_output, [interval])]
        )

        output_path = tmp_path / "pred.bw"

        # Mock pybigtools to capture written data
        mock_bw = MagicMock()
        written_records = []

        def capture_write(chrom_sizes, generator):
            for record in generator:
                written_records.append(record)

        mock_bw.write.side_effect = capture_write

        with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt, \
             patch("cerberus.predict_bigwig.SlidingWindowSampler") as mock_sw:
            mock_pbt.open.return_value = mock_bw

            # SlidingWindowSampler yields one window
            mock_sw.return_value = iter([interval])

            predict_to_bigwig(
                output_path=output_path,
                dataset=dataset,
                model_ensemble=ensemble,
                batch_size=1,
            )

        # Verify BigWig was opened for writing
        mock_pbt.open.assert_called_once_with(str(output_path), "w")
        mock_bw.close.assert_called_once()

        # Verify records were written
        assert len(written_records) == n_bins
        for chrom, _start, _end, val in written_records:
            assert chrom == "chr1"
            assert isinstance(val, float)
            assert val >= 0

    def test_regions_mode(self, tmp_path):
        """When regions are provided, only those regions are predicted."""
        input_len = 100
        output_len = 50
        n_bins = 50
        count_pseudocount = 0.0

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
            count_pseudocount=count_pseudocount,
        )

        logits = torch.zeros(1, 1, n_bins)
        log_counts = torch.full((1, 1), math.log(50.0))
        batch_output = ProfileCountOutput(logits=logits, log_counts=log_counts)

        # The interval that will be tiled from the region
        called_intervals = []

        def mock_predict(*args, **kwargs):
            intervals = list(args[0]) if args else list(kwargs.get("intervals", []))
            called_intervals.extend(intervals)
            for iv in intervals:
                yield batch_output, [iv]

        ensemble.predict_intervals_batched.side_effect = mock_predict

        mock_bw = MagicMock()
        written_records = []

        def capture_write(chrom_sizes, generator):
            for record in generator:
                written_records.append(record)

        mock_bw.write.side_effect = capture_write

        region = Interval("chr1", 1000, 1200, "+")

        with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt:
            mock_pbt.open.return_value = mock_bw
            predict_to_bigwig(
                output_path=tmp_path / "pred.bw",
                dataset=dataset,
                model_ensemble=ensemble,
                regions=[region],
                batch_size=1,
            )

        # Verify that written records are within the region's output area
        assert len(written_records) > 0
        for chrom, _start, _end, _val in written_records:
            assert chrom == "chr1"

    def test_stride_defaults_to_half_output_len(self, tmp_path):
        """When stride is None, it defaults to output_len // 2."""
        input_len = 100
        output_len = 50

        dataset, ensemble = _make_mock_ensemble_and_dataset(
            input_len=input_len,
            output_len=output_len,
        )

        # We don't need the full pipeline — just check that SlidingWindowSampler
        # is called with the correct stride
        ensemble.predict_intervals_batched.return_value = iter([])

        mock_bw = MagicMock()
        mock_bw.write.side_effect = lambda cs, gen: list(gen)

        with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt, \
             patch("cerberus.predict_bigwig.SlidingWindowSampler") as mock_sw:
            mock_pbt.open.return_value = mock_bw
            mock_sw.return_value = iter([])

            predict_to_bigwig(
                output_path=tmp_path / "pred.bw",
                dataset=dataset,
                model_ensemble=ensemble,
                stride=None,
            )

            # SlidingWindowSampler should have been called with stride = 25
            call_kwargs = mock_sw.call_args.kwargs
            assert call_kwargs["stride"] == output_len // 2

    def test_custom_stride(self, tmp_path):
        """Explicit stride is forwarded to SlidingWindowSampler."""
        dataset, ensemble = _make_mock_ensemble_and_dataset()

        ensemble.predict_intervals_batched.return_value = iter([])

        mock_bw = MagicMock()
        mock_bw.write.side_effect = lambda cs, gen: list(gen)

        with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt, \
             patch("cerberus.predict_bigwig.SlidingWindowSampler") as mock_sw:
            mock_pbt.open.return_value = mock_bw
            mock_sw.return_value = iter([])

            predict_to_bigwig(
                output_path=tmp_path / "pred.bw",
                dataset=dataset,
                model_ensemble=ensemble,
                stride=10,
            )

            call_kwargs = mock_sw.call_args.kwargs
            assert call_kwargs["stride"] == 10

    def test_bigwig_closed_on_error(self, tmp_path):
        """BigWig file handle is closed even if an error occurs during writing."""
        dataset, ensemble = _make_mock_ensemble_and_dataset()

        mock_bw = MagicMock()

        def raise_during_write(chrom_sizes, generator):
            raise RuntimeError("write error")

        mock_bw.write.side_effect = raise_during_write

        ensemble.predict_intervals_batched.return_value = iter([])

        with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt:
            mock_pbt.open.return_value = mock_bw

            with pytest.raises(RuntimeError, match="write error"):
                predict_to_bigwig(
                    output_path=tmp_path / "pred.bw",
                    dataset=dataset,
                    model_ensemble=ensemble,
                )

            # close() should still be called via finally block
            mock_bw.close.assert_called_once()
