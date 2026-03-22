"""Test for off-by-one bug in PeakSampler / NegativePeakSampler peak exclusion zones.

InterLap stores closed intervals [start, end]. Cerberus intervals are half-open
[start, end). When adding peak intervals to the exclusion InterLap, the code must
convert to closed form: (start, end - 1). The buggy code uses (start, end), which
extends the exclusion zone one base pair too far on the right side.

This test verifies:
- A query starting exactly at peak.end (exclusive) should NOT be excluded.
- A query ending exactly at peak.start should NOT be excluded.
- A query overlapping the peak interior SHOULD be excluded.
"""

from unittest.mock import MagicMock, patch

from cerberus.exclude import is_excluded
from cerberus.interval import Interval
from cerberus.samplers import (
    ComplexityMatchedSampler,
    IntervalSampler,
    PeakSampler,
    RandomSampler,
)


@patch("cerberus.samplers.ComplexityMatchedSampler")
@patch("cerberus.samplers.RandomSampler")
@patch("cerberus.samplers.IntervalSampler")
@patch("cerberus.samplers.compute_intervals_complexity")
def test_peak_sampler_exclusion_zone_boundary(
    mock_compute, mock_interval_cls, mock_random_cls, mock_complexity_cls
):
    """An interval starting at exactly peak.end should not be excluded.

    Peak is [100, 200) half-open. The exclusion zone stored in InterLap should
    be the closed interval [100, 199]. A candidate at [200, 250) should NOT
    overlap this exclusion.
    """
    peak = Interval("chr1", 100, 200, "+")

    # Mock IntervalSampler to return our single peak
    mock_positives = MagicMock(spec=IntervalSampler)
    mock_positives.__len__.return_value = 1
    mock_positives.__iter__.return_value = iter([peak])
    mock_positives.__getitem__.side_effect = lambda i: peak
    mock_interval_cls.return_value = mock_positives

    mock_random_cls.return_value = MagicMock(spec=RandomSampler)
    mock_complexity_cls.return_value = MagicMock(spec=ComplexityMatchedSampler)

    PeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes={"chr1": 10000},
        padded_size=100,
        background_ratio=1.0,
    )

    # Capture the exclude_intervals passed to RandomSampler
    neg_excludes = mock_random_cls.call_args.kwargs["exclude_intervals"]

    # The exclusion zone should cover the peak [100, 200) and nothing more.
    # An interval starting at exactly position 200 should NOT be excluded
    # because 200 is the exclusive end of the peak.
    assert not is_excluded(neg_excludes, "chr1", 200, 250), (
        "Interval [200, 250) starts at the exclusive end of peak [100, 200) "
        "and should NOT be excluded, but it is — off-by-one in exclusion zone"
    )

    # Sanity: interval overlapping the peak interior SHOULD be excluded
    assert is_excluded(neg_excludes, "chr1", 150, 250)

    # Sanity: interval just before the peak should not be excluded
    assert not is_excluded(neg_excludes, "chr1", 50, 100)

    # An interval that ends exactly at peak.start (exclusive end = 100,
    # so closed = [50, 99]) should NOT overlap [100, 199]
    assert not is_excluded(neg_excludes, "chr1", 50, 100)

    # An interval touching the last base of the peak SHOULD be excluded
    # [199, 200) overlaps [100, 199] in closed form
    assert is_excluded(neg_excludes, "chr1", 199, 200)


@patch("cerberus.samplers.ComplexityMatchedSampler")
@patch("cerberus.samplers.RandomSampler")
@patch("cerberus.samplers.IntervalSampler")
@patch("cerberus.samplers.compute_intervals_complexity")
def test_peak_sampler_exclusion_zone_single_base_gap(
    mock_compute, mock_interval_cls, mock_random_cls, mock_complexity_cls
):
    """Two adjacent peaks should not create a gap exclusion between them.

    Peak A: [100, 200), Peak B: [200, 300). Together they cover [100, 300).
    There should be no gap at position 200. With the off-by-one bug, peak A's
    exclusion extends to [100, 200] closed = [100, 201) half-open, which masks
    the bug for adjacent peaks but still incorrectly excludes position 200
    even if peak B did not exist.
    """
    peak_a = Interval("chr1", 100, 200, "+")
    peak_b = Interval("chr1", 300, 400, "+")  # gap between 200 and 300

    mock_positives = MagicMock(spec=IntervalSampler)
    mock_positives.__len__.return_value = 2
    mock_positives.__iter__.return_value = iter([peak_a, peak_b])
    mock_positives.__getitem__.side_effect = lambda i: [peak_a, peak_b][i]
    mock_interval_cls.return_value = mock_positives

    mock_random_cls.return_value = MagicMock(spec=RandomSampler)
    mock_complexity_cls.return_value = MagicMock(spec=ComplexityMatchedSampler)

    PeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes={"chr1": 10000},
        padded_size=100,
        background_ratio=1.0,
    )

    neg_excludes = mock_random_cls.call_args.kwargs["exclude_intervals"]

    # Position 200 is in the gap between peaks — should NOT be excluded
    assert not is_excluded(neg_excludes, "chr1", 200, 201), (
        "Single-base interval [200, 201) is in the gap between peak A [100, 200) "
        "and peak B [300, 400) — should NOT be excluded"
    )

    # Position 299 is also in the gap — should NOT be excluded
    assert not is_excluded(neg_excludes, "chr1", 299, 300)

    # Position 300 is the start of peak B — SHOULD be excluded
    assert is_excluded(neg_excludes, "chr1", 300, 301)
