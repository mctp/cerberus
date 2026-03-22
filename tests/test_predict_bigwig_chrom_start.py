"""Test for window generation when region is near chromosome start.

When region.start < offset (= (input_len - output_len) // 2), the model
cannot predict positions [region.start, offset) because it needs `offset`
bp of left context. The fix clamps `pos` before the loop starts (not
mid-loop via max(0, pos)), preserving stride alignment and emitting a
warning about the coverage gap.
"""

import logging
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from cerberus.output import ProfileCountOutput
from cerberus.predict_bigwig import predict_to_bigwig


def _make_mocks(
    input_len=2048, output_len=1024, output_bin_size=1, chrom_size=1_000_000
):
    """Lightweight mocks of ModelEnsemble and CerberusDataset."""
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
        target_scale=1.0,
        use_sequence=True,
    )
    genome_config = GenomeConfig.model_construct(
        name="test",
        fasta_path=Path("test.fa"),
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": chrom_size},
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
        loss_cls="cerberus.loss.PoissonMultinomialLoss",
        loss_args={},
        metrics_cls="test",
        metrics_args={},
        model_args={},
        pretrained=[],
        count_pseudocount=0.0,
    )
    ensemble = MagicMock(spec=ModelEnsemble)
    ensemble.cerberus_config = CerberusConfig.model_construct(
        genome_config=genome_config,
        data_config=data_config,
        model_config_=model_config,
        sampler_config=SamplerConfig.model_construct(
            sampler_type="interval",
            padded_size=input_len,
            sampler_args={},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1,
            max_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            patience=1,
            optimizer="adam",
            scheduler_type="default",
            scheduler_args={},
            filter_bias_and_bn=False,
            reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8,
            gradient_clip_val=None,
        ),
    )
    return dataset, ensemble


def _make_batch_output(n_bins, total_count=100.0):
    """Create a uniform ProfileCountOutput for a single window."""
    logits = torch.zeros(1, 1, n_bins)
    log_counts = torch.full((1, 1), math.log(total_count))
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


def _run_predict_regions(dataset, ensemble, region, tmp_path):
    """Run predict_to_bigwig in regions mode, return called intervals."""
    output_len = dataset.data_config.output_len
    n_bins = output_len // dataset.data_config.output_bin_size
    batch_output = _make_batch_output(n_bins, total_count=float(n_bins))

    called_intervals: list[Interval] = []

    def mock_predict(intervals, *args, **kwargs):
        for iv in intervals:
            called_intervals.append(iv)
            yield batch_output, [iv]

    ensemble.predict_intervals_batched.side_effect = mock_predict

    mock_bw = MagicMock()
    mock_bw.write.side_effect = lambda cs, gen: list(gen)

    with patch("cerberus.predict_bigwig.pybigtools") as mock_pbt:
        mock_pbt.open.return_value = mock_bw
        predict_to_bigwig(
            output_path=tmp_path / "pred.bw",
            dataset=dataset,
            model_ensemble=ensemble,
            regions=[region],
            batch_size=64,
        )

    return called_intervals


class TestRegionNearChromStart:
    """Window generation for regions where region.start < offset."""

    def test_stride_alignment_preserved_near_chrom_start(self, tmp_path):
        """All windows must be on a regular stride grid, even when the
        region starts near chromosome position 0.

        With input_len=200, output_len=100, offset=50, stride=50:
        Region [10, 500) — ideal first pos=-40, clamped to 0.
        All window starts should be spaced by stride=50.
        """
        input_len = 200
        output_len = 100
        offset = (input_len - output_len) // 2  # 50
        stride = output_len // 2  # 50

        dataset, ensemble = _make_mocks(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=1,
            chrom_size=100_000,
        )

        region = Interval("chr1", 10, 500, "+")
        called_intervals = _run_predict_regions(dataset, ensemble, region, tmp_path)

        assert len(called_intervals) >= 2

        output_starts = [iv.start + offset for iv in called_intervals]

        for i in range(1, len(output_starts)):
            gap = output_starts[i] - output_starts[i - 1]
            assert gap == stride, (
                f"Output start gap between window {i - 1} and {i} is {gap}, "
                f"expected stride={stride}. "
                f"Output starts: {output_starts[:5]}..."
            )

    def test_warning_emitted_near_chrom_start(self, tmp_path):
        """A warning should be logged when region.start < offset,
        indicating the coverage gap.
        """
        input_len = 200
        output_len = 100

        dataset, ensemble = _make_mocks(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=1,
            chrom_size=100_000,
        )

        region = Interval("chr1", 10, 500, "+")

        logger = logging.getLogger("cerberus.predict_bigwig")
        with patch.object(logger, "warning", wraps=logger.warning) as mock_warn:
            _run_predict_regions(dataset, ensemble, region, tmp_path)

        warning_messages = [str(call) for call in mock_warn.call_args_list]
        assert any("starts within" in msg for msg in warning_messages), (
            f"Expected a warning about region near chromosome start, "
            f"got: {warning_messages}"
        )

    def test_no_warning_for_normal_region(self, tmp_path):
        """No warning should be emitted for regions far from chromosome start."""
        input_len = 200
        output_len = 100

        dataset, ensemble = _make_mocks(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=1,
            chrom_size=100_000,
        )

        region = Interval("chr1", 5000, 6000, "+")

        logger = logging.getLogger("cerberus.predict_bigwig")
        with patch.object(logger, "warning", wraps=logger.warning) as mock_warn:
            _run_predict_regions(dataset, ensemble, region, tmp_path)

        boundary_warnings = [
            str(call)
            for call in mock_warn.call_args_list
            if "starts within" in str(call)
        ]
        assert len(boundary_warnings) == 0

    def test_normal_region_stride_aligned(self, tmp_path):
        """A region far from chromosome start should have stride-aligned
        windows starting at region.start.
        """
        input_len = 200
        output_len = 100
        offset = 50
        stride = 50

        dataset, ensemble = _make_mocks(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=1,
            chrom_size=100_000,
        )

        region = Interval("chr1", 5000, 6000, "+")
        called_intervals = _run_predict_regions(dataset, ensemble, region, tmp_path)

        assert len(called_intervals) >= 2

        output_starts = [iv.start + offset for iv in called_intervals]

        # First output should start at region.start
        assert output_starts[0] == region.start

        # All gaps should be stride
        for i in range(1, len(output_starts)):
            gap = output_starts[i] - output_starts[i - 1]
            assert gap == stride
