from unittest.mock import MagicMock, patch

import pytest
from interlap import InterLap

from cerberus.interval import Interval
from cerberus.samplers import (
    ComplexityMatchedSampler,
    IntervalSampler,
    NegativePeakSampler,
    RandomSampler,
    create_sampler,
)


@pytest.fixture
def mock_dependencies():
    with patch("cerberus.samplers.IntervalSampler") as mock_interval_sampler, \
         patch("cerberus.samplers.RandomSampler") as mock_random_sampler, \
         patch("cerberus.samplers.ComplexityMatchedSampler") as mock_complexity_matched_sampler:

        mock_interval_instance = MagicMock(spec=IntervalSampler)
        mock_interval_instance.__len__.return_value = 100
        intervals = [Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(100)]
        mock_interval_instance.__iter__.return_value = iter(intervals)
        mock_interval_instance.__getitem__.side_effect = lambda i: intervals[i]
        mock_interval_sampler.return_value = mock_interval_instance

        mock_random_instance = MagicMock(spec=RandomSampler)
        mock_random_sampler.return_value = mock_random_instance

        mock_complexity_instance = MagicMock(spec=ComplexityMatchedSampler)
        mock_complexity_matched_sampler.return_value = mock_complexity_instance

        yield {
            "interval_sampler": mock_interval_sampler,
            "random_sampler": mock_random_sampler,
            "complexity_matched_sampler": mock_complexity_matched_sampler,
        }

def test_negative_peak_sampler_init(mock_dependencies):
    chrom_sizes = {"chr1": 10000}
    exclude_intervals = {"chr1": InterLap()}

    sampler = NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        exclude_intervals=exclude_intervals,
        background_ratio=1.0,
    )

    # IntervalSampler created for peak reference
    mock_dependencies["interval_sampler"].assert_called_once()

    # RandomSampler created for candidates
    mock_dependencies["random_sampler"].assert_called_once()
    call_args = mock_dependencies["random_sampler"].call_args
    assert call_args.kwargs["num_intervals"] == 10000
    assert call_args.kwargs.get("generate_on_init") is False

    # ComplexityMatchedSampler created for negatives
    mock_dependencies["complexity_matched_sampler"].assert_called_once()
    assert mock_dependencies["complexity_matched_sampler"].call_args.kwargs.get("generate_on_init") is False
    assert mock_dependencies["complexity_matched_sampler"].call_args.kwargs.get("metrics") == ["gc", "dust", "cpg"]

    # Peaks excluded from background candidates
    random_excludes = call_args.kwargs["exclude_intervals"]
    assert "chr1" in random_excludes
    assert len(list(random_excludes["chr1"])) == 100

    # Only one sub-sampler (negatives), not two
    assert len(sampler.samplers) == 1

def test_negative_peak_sampler_interval_source_always_complexity_matched(mock_dependencies):
    chrom_sizes = {"chr1": 10000}

    sampler = NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=1.0,
    )

    for i in range(len(sampler)):
        assert sampler.get_interval_source(i) == "ComplexityMatchedSampler"

def test_negative_peak_sampler_no_peaks_in_training(mock_dependencies):
    """Verify that peaks are NOT included as a sub-sampler."""
    chrom_sizes = {"chr1": 10000}

    sampler = NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=1.0,
    )

    assert sampler.samplers[0] is sampler.negatives

def test_negative_peak_sampler_resample(mock_dependencies):
    """Verify that resample propagates to negatives."""
    chrom_sizes = {"chr1": 10000}

    sampler = NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=1.0,
    )

    sampler.resample(seed=42)

    # Negatives resample should have been called
    call_args = sampler.negatives.resample.call_args  # type: ignore[union-attr]
    assert isinstance(call_args.args[0], int)

def test_negative_peak_sampler_custom_parameters(mock_dependencies):
    """Verify custom min_candidates and candidate_oversample_factor."""
    chrom_sizes = {"chr1": 10000}

    NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=1.0,
        min_candidates=5000,
        candidate_oversample_factor=5.0,
    )

    call_args = mock_dependencies["random_sampler"].call_args
    # max(5000, 100 * 1.0 * 5.0) = 5000
    assert call_args.kwargs["num_intervals"] == 5000

    # Test where oversample factor dominates
    NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=2.0,
        min_candidates=100,
        candidate_oversample_factor=100.0,
    )

    call_args = mock_dependencies["random_sampler"].call_args
    # max(100, 100 * 2.0 * 100.0) = 20000
    assert call_args.kwargs["num_intervals"] == 20000

def test_negative_peak_sampler_background_ratio_scales_candidates(mock_dependencies):
    """Verify background_ratio is passed to ComplexityMatchedSampler as candidate_ratio."""
    chrom_sizes = {"chr1": 10000}

    NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=3.0,
    )

    cm_args = mock_dependencies["complexity_matched_sampler"].call_args
    assert cm_args.kwargs["candidate_ratio"] == 3.0

def test_negative_peak_sampler_none_exclude_intervals(mock_dependencies):
    """Verify that None exclude_intervals is handled (peaks still excluded from candidates)."""
    chrom_sizes = {"chr1": 10000}

    sampler = NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        exclude_intervals=None,
        background_ratio=1.0,
    )

    # Should still work — peaks are added to neg_excludes from empty dict
    random_call_args = mock_dependencies["random_sampler"].call_args
    random_excludes = random_call_args.kwargs["exclude_intervals"]
    assert "chr1" in random_excludes
    assert len(list(random_excludes["chr1"])) == 100
    assert len(sampler.samplers) == 1

def test_negative_peak_sampler_multi_chrom(mock_dependencies):
    """Verify peaks on multiple chromosomes are all excluded."""
    chrom_sizes = {"chr1": 10000, "chr2": 10000}

    # Override mock to return intervals on two chroms
    intervals = (
        [Interval("chr1", i * 100, i * 100 + 50, "+") for i in range(50)]
        + [Interval("chr2", i * 100, i * 100 + 50, "+") for i in range(50)]
    )
    mock_interval_instance = mock_dependencies["interval_sampler"].return_value
    mock_interval_instance.__len__.return_value = 100
    mock_interval_instance.__iter__.return_value = iter(intervals)

    NegativePeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        background_ratio=1.0,
    )

    random_call_args = mock_dependencies["random_sampler"].call_args
    random_excludes = random_call_args.kwargs["exclude_intervals"]
    assert "chr1" in random_excludes
    assert "chr2" in random_excludes
    assert len(list(random_excludes["chr1"])) == 50
    assert len(list(random_excludes["chr2"])) == 50

def test_create_sampler_negative_peak():
    """Verify create_sampler dispatches to NegativePeakSampler."""
    from pathlib import Path

    from cerberus.config import SamplerConfig

    config = SamplerConfig.model_construct(
        sampler_type="negative_peak",
        padded_size=50,
        sampler_args={"intervals_path": Path("peaks.bed"), "background_ratio": 1.0, "complexity_center_size": None},
    )

    with patch("cerberus.samplers.NegativePeakSampler") as MockNPS:
        create_sampler(
            config=config,
            chrom_sizes={"chr1": 10000},
            folds=[],
            exclude_intervals={"chr1": InterLap()},
            fasta_path="genome.fa",
        )

        MockNPS.assert_called_once()
        call_args = MockNPS.call_args
        assert call_args.kwargs["intervals_path"] == Path("peaks.bed")
        assert call_args.kwargs["background_ratio"] == 1.0
        assert call_args.kwargs["fasta_path"] == "genome.fa"

def test_create_sampler_negative_peak_requires_fasta():
    """Verify create_sampler raises if fasta_path is None for negative_peak."""
    from pathlib import Path

    from cerberus.config import SamplerConfig

    config = SamplerConfig.model_construct(
        sampler_type="negative_peak",
        padded_size=50,
        sampler_args={"intervals_path": Path("peaks.bed"), "background_ratio": 1.0, "complexity_center_size": None},
    )

    with pytest.raises(ValueError, match="fasta_path"):
        create_sampler(
            config=config,
            chrom_sizes={"chr1": 10000},
            folds=[],
            exclude_intervals={"chr1": InterLap()},
            fasta_path=None,
        )
