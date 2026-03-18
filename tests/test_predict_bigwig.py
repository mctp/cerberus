import logging

import pytest
import torch

from cerberus.interval import Interval
from cerberus.output import ProfileCountOutput, ProfileLogRates
from cerberus.predict_bigwig import _process_island, _reconstruct_profile_counts


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
