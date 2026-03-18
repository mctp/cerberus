import logging

import pytest
import torch

from cerberus.interval import Interval
from cerberus.output import ProfileCountOutput, ProfileLogRates
from cerberus.predict_bigwig import (
    _load_peak_nonpeak_intervals_for_chrom,
    _process_island,
    _reconstruct_profile_counts,
    _stitch_tracks_midpoint_between_summits,
)


class _DummyDataset:
    data_config = {"output_len": 4, "output_bin_size": 1}


class _DummyEnsemble:
    def __init__(self, batches):
        self._batches = batches

    def predict_intervals_batched(
        self,
        intervals,
        dataset,
        use_folds=None,
        aggregation="model",
        batch_size=64,
    ):
        del intervals, dataset, use_folds, aggregation, batch_size
        for item in self._batches:
            yield item


def test_reconstruct_profile_counts_matches_chrombpnet_formula():
    logits = torch.zeros((1, 1, 4), dtype=torch.float32)
    log_counts = torch.log(torch.tensor([[10.0]], dtype=torch.float32))
    reconstructed = _reconstruct_profile_counts(logits, log_counts)
    expected = torch.full((1, 1, 4), 2.5, dtype=torch.float32)
    assert torch.allclose(reconstructed, expected)


def test_process_island_uses_reconstructed_profile_counts():
    interval = Interval("chr1", 0, 6)
    output = ProfileCountOutput(
        logits=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_counts=torch.log(torch.tensor([[10.0]], dtype=torch.float32)),
        out_interval=None,
    )
    ensemble = _DummyEnsemble([(output, [interval])])
    rows = list(
        _process_island(
            island_intervals=[interval],
            dataset=_DummyDataset(),
            model_ensemble=ensemble,
            use_folds=["test"],
            aggregation="model",
            batch_size=1,
        )
    )

    assert len(rows) == 4
    assert rows[0][:3] == ("chr1", 1, 2)
    assert rows[-1][:3] == ("chr1", 4, 5)
    for _, _, _, value in rows:
        assert value == pytest.approx(2.5)


def test_process_island_warns_and_uses_channel_zero(caplog):
    interval = Interval("chr1", 0, 6)
    output = ProfileCountOutput(
        logits=torch.zeros((1, 2, 4), dtype=torch.float32),
        log_counts=torch.log(torch.tensor([[10.0, 20.0]], dtype=torch.float32)),
        out_interval=None,
    )
    ensemble = _DummyEnsemble([(output, [interval])])

    with caplog.at_level(logging.WARNING):
        rows = list(
            _process_island(
                island_intervals=[interval],
                dataset=_DummyDataset(),
                model_ensemble=ensemble,
                use_folds=["test"],
                aggregation="model",
                batch_size=1,
            )
        )

    assert any("only channel 0 will be exported" in r.message for r in caplog.records)
    for _, _, _, value in rows:
        assert value == pytest.approx(2.5)


def test_process_island_preserves_profile_log_rates_behavior():
    interval = Interval("chr1", 0, 6)
    output = ProfileLogRates(
        log_rates=torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32),
        out_interval=None,
    )
    ensemble = _DummyEnsemble([(output, [interval])])

    rows = list(
        _process_island(
            island_intervals=[interval],
            dataset=_DummyDataset(),
            model_ensemble=ensemble,
            use_folds=["test"],
            aggregation="model",
            batch_size=1,
        )
    )
    assert [row[3] for row in rows] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_midpoint_stitch_splits_overlap_by_adjacent_summit_midpoint():
    tracks = [
        torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float32),
    ]
    intervals = [
        Interval("chr1", 100, 104),
        Interval("chr1", 102, 106),
    ]
    summit_positions = [101, 105]

    rows = list(
        _stitch_tracks_midpoint_between_summits(
            tracks=tracks,
            intervals=intervals,
            summit_positions=summit_positions,
            output_bin_size=1,
        )
    )

    starts = [row[1] for row in rows]
    values = [row[3] for row in rows]
    assert starts == [100, 101, 102, 103, 104, 105]
    assert values == pytest.approx([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def test_process_island_supports_midpoint_between_summits_overlap_resolution():
    intervals = [Interval("chr1", 0, 6), Interval("chr1", 2, 8)]
    output = ProfileLogRates(
        log_rates=torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]], dtype=torch.float32
        ),
        out_interval=None,
    )
    ensemble = _DummyEnsemble([(output, intervals)])

    rows = list(
        _process_island(
            island_intervals=intervals,
            dataset=_DummyDataset(),
            model_ensemble=ensemble,
            use_folds=["test"],
            aggregation="model",
            batch_size=2,
            overlap_resolution="midpoint_between_summits",
        )
    )

    starts = [row[1] for row in rows]
    values = [row[3] for row in rows]
    assert starts == [1, 2, 3, 4, 5, 6]
    assert values == pytest.approx([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def test_load_peak_nonpeak_intervals_for_chrom_filters_and_combines(tmp_path):
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t140\nchr2\t200\t240\n")

    nonpeaks = tmp_path / "nonpeaks.bed"
    nonpeaks.write_text("chr1\t300\t340\n")

    intervals = _load_peak_nonpeak_intervals_for_chrom(
        peaks_path=peaks,
        nonpeaks_path=nonpeaks,
        chrom="chr1",
        chrom_sizes={"chr1": 1000, "chr2": 1000},
        padded_size=40,
        exclude_intervals={},
    )

    assert len(intervals) == 2
    assert all(iv.chrom == "chr1" for iv in intervals)
    assert [iv.start for iv in intervals] == sorted(iv.start for iv in intervals)
