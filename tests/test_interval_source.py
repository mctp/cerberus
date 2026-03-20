"""Tests for interval_source labelling in MultiSampler and CerberusDataset."""
from unittest.mock import MagicMock
from cerberus.interval import Interval
from cerberus.samplers import MultiSampler


# ---------------------------------------------------------------------------
# Helper: a minimal sampler stub
# ---------------------------------------------------------------------------

class _StubSampler:
    """Minimal sampler that returns a fixed list of intervals."""
    def __init__(self, intervals: list[Interval]):
        self._intervals = intervals

    def __len__(self) -> int:
        return len(self._intervals)

    def __iter__(self):  # type: ignore[override]
        return iter(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def resample(self, seed: int | None = None) -> None:
        pass

    def split_folds(self, test_fold: int | None = None, val_fold: int | None = None):  # type: ignore[override]
        return self, self, self

    def get_interval_source(self, idx: int) -> str:
        return type(self).__name__


class _PeakStub(_StubSampler):
    """Stub representing peak intervals."""
    pass


class _BgStub(_StubSampler):
    """Stub representing background intervals."""
    pass


_PEAK = Interval("chr1", 100, 200)
_BG1  = Interval("chr2", 300, 400)
_BG2  = Interval("chr2", 500, 600)


# ---------------------------------------------------------------------------
# MultiSampler.get_interval_source
# ---------------------------------------------------------------------------

def test_get_interval_source_single_sampler() -> None:
    """With one sub-sampler all intervals report that sampler's class name."""
    peaks = _StubSampler([_PEAK])
    ms = MultiSampler([peaks], seed=0)

    for i in range(len(ms)):
        assert ms.get_interval_source(i) == "_StubSampler"


def test_get_interval_source_two_same_type() -> None:
    """Two sub-samplers of the same type both report the same class name."""
    peaks = _StubSampler([_PEAK])
    bg    = _StubSampler([_BG1, _BG2])
    ms = MultiSampler([peaks, bg], seed=42)

    sources = [ms.get_interval_source(i) for i in range(len(ms))]
    assert all(s == "_StubSampler" for s in sources)
    assert len(sources) == 3


def test_get_interval_source_distinguishes_sampler_types() -> None:
    """Different sub-sampler classes produce different source strings."""
    peaks = _PeakStub([_PEAK])
    bg = _BgStub([_BG1, _BG2])
    ms = MultiSampler([peaks, bg], seed=42)

    sources = [ms.get_interval_source(i) for i in range(len(ms))]
    assert sources.count("_PeakStub") == 1
    assert sources.count("_BgStub") == 2


def test_get_interval_source_consistent_with_getitem() -> None:
    """
    For every idx, the interval from __getitem__ and the source from
    get_interval_source are derived from the same _indices entry, so they
    are always in sync -- even after resample() reshuffles the index table.
    """
    peaks = _PeakStub([_PEAK])
    bg    = _BgStub([_BG1, _BG2])
    ms = MultiSampler([peaks, bg], seed=0)

    for trial_seed in (0, 1, 2):
        ms.resample(seed=trial_seed)
        for idx in range(len(ms)):
            interval = ms[idx]
            source = ms.get_interval_source(idx)
            if interval == _PEAK:
                assert source == "_PeakStub", f"seed={trial_seed}, idx={idx}: peak interval has source {source}"
            else:
                assert source == "_BgStub", f"seed={trial_seed}, idx={idx}: bg interval has source {source}"


def test_get_interval_source_preserved_after_split_folds() -> None:
    """split_folds returns MultiSampler instances that also support get_interval_source."""
    peaks = _StubSampler([_PEAK])
    bg    = _StubSampler([_BG1])
    ms = MultiSampler([peaks, bg], seed=0)

    train, val, test = ms.split_folds()

    assert hasattr(train, "get_interval_source")
    assert hasattr(val,   "get_interval_source")
    assert hasattr(test,  "get_interval_source")


# ---------------------------------------------------------------------------
# CerberusDataset.__getitem__ -- interval_source field
# ---------------------------------------------------------------------------

def test_dataset_getitem_includes_interval_source_from_sampler() -> None:
    """`interval_source` in batch equals what the sampler's get_interval_source returns."""
    from cerberus.dataset import CerberusDataset

    mock_sampler = MagicMock()
    mock_sampler.__len__ = MagicMock(return_value=2)
    mock_sampler.__getitem__ = MagicMock(return_value=_PEAK)
    mock_sampler.get_interval_source = MagicMock(return_value="ComplexityMatchedSampler")

    dataset = MagicMock(spec=CerberusDataset)
    dataset.sampler = mock_sampler
    dataset._get_interval = MagicMock(return_value={
        "inputs": MagicMock(), "targets": MagicMock(), "intervals": str(_PEAK)
    })
    result = CerberusDataset.__getitem__(dataset, 0)

    assert result["interval_source"] == "ComplexityMatchedSampler"
    mock_sampler.get_interval_source.assert_called_once_with(0)
