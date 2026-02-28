"""Tests for peak_status labelling in MultiSampler and CerberusDataset."""
from unittest.mock import MagicMock
from cerberus.interval import Interval
from cerberus.samplers import MultiSampler


# ---------------------------------------------------------------------------
# Helper: a minimal sampler stub
# ---------------------------------------------------------------------------

class _StubSampler:
    """Minimal sampler that returns a fixed list of intervals."""
    def __init__(self, intervals):
        self._intervals = intervals

    def __len__(self):
        return len(self._intervals)

    def __iter__(self):
        return iter(self._intervals)

    def __getitem__(self, idx):
        return self._intervals[idx]

    def resample(self, seed=None):
        pass

    def split_folds(self, test_fold=None, val_fold=None):
        return self, self, self


_PEAK = Interval("chr1", 100, 200)
_BG1  = Interval("chr2", 300, 400)
_BG2  = Interval("chr2", 500, 600)


# ---------------------------------------------------------------------------
# MultiSampler.get_peak_status
# ---------------------------------------------------------------------------

def test_get_peak_status_single_sampler():
    """With one sub-sampler all intervals are labelled as peaks (sampler_idx == 0)."""
    peaks = _StubSampler([_PEAK])
    ms = MultiSampler([peaks], seed=0)

    for i in range(len(ms)):
        assert ms.get_peak_status(i) == 1


def test_get_peak_status_two_samplers():
    """First sub-sampler → 1 (peaks); second → 0 (background)."""
    peaks = _StubSampler([_PEAK])
    bg    = _StubSampler([_BG1, _BG2])
    ms = MultiSampler([peaks, bg], seed=42)

    statuses = [ms.get_peak_status(i) for i in range(len(ms))]
    # 1 peak + 2 background intervals
    assert sorted(statuses) == [0, 0, 1]


def test_get_peak_status_consistent_with_getitem():
    """
    For every idx, the interval from __getitem__ and the label from get_peak_status
    are derived from the same _indices entry, so they are always in sync — even after
    resample() reshuffles the index table.
    """
    peaks = _StubSampler([_PEAK])
    bg    = _StubSampler([_BG1, _BG2])
    ms = MultiSampler([peaks, bg], seed=0)

    for trial_seed in (0, 1, 2):
        ms.resample(seed=trial_seed)
        for idx in range(len(ms)):
            interval    = ms[idx]
            peak_status = ms.get_peak_status(idx)
            # If the interval came from peaks (samplers[0]), status must be 1
            if interval == _PEAK:
                assert peak_status == 1, f"seed={trial_seed}, idx={idx}: peak interval labelled {peak_status}"
            else:
                assert peak_status == 0, f"seed={trial_seed}, idx={idx}: bg interval labelled {peak_status}"


def test_get_peak_status_preserved_after_split_folds():
    """split_folds returns MultiSampler instances that also support get_peak_status."""
    peaks = _StubSampler([_PEAK])
    bg    = _StubSampler([_BG1])
    ms = MultiSampler([peaks, bg], seed=0)

    train, val, test = ms.split_folds()

    assert hasattr(train, "get_peak_status")
    assert hasattr(val,   "get_peak_status")
    assert hasattr(test,  "get_peak_status")


# ---------------------------------------------------------------------------
# CerberusDataset.__getitem__ — peak_status field
# ---------------------------------------------------------------------------

def test_dataset_getitem_includes_peak_status_from_sampler():
    """`peak_status` in batch equals what the sampler's get_peak_status returns."""
    from cerberus.dataset import CerberusDataset

    mock_sampler = MagicMock()
    mock_sampler.__len__ = MagicMock(return_value=2)
    mock_sampler.__getitem__ = MagicMock(return_value=_PEAK)
    mock_sampler.get_peak_status = MagicMock(return_value=0)  # background

    dataset = MagicMock(spec=CerberusDataset)
    dataset.sampler = mock_sampler
    dataset._get_interval = MagicMock(return_value={
        "inputs": MagicMock(), "targets": MagicMock(), "intervals": str(_PEAK)
    })
    # Call the real __getitem__ via the unbound method
    result = CerberusDataset.__getitem__(dataset, 0)

    assert result["peak_status"] == 0
    mock_sampler.get_peak_status.assert_called_once_with(0)


def test_dataset_getitem_defaults_peak_status_when_not_supported():
    """`peak_status` defaults to 1 when the sampler has no get_peak_status."""
    from cerberus.dataset import CerberusDataset

    mock_sampler = MagicMock(spec=[  # spec without get_peak_status
        "__len__", "__getitem__", "resample", "split_folds"
    ])
    mock_sampler.__getitem__ = MagicMock(return_value=_PEAK)

    dataset = MagicMock(spec=CerberusDataset)
    dataset.sampler = mock_sampler
    dataset._get_interval = MagicMock(return_value={
        "inputs": MagicMock(), "targets": MagicMock(), "intervals": str(_PEAK)
    })
    result = CerberusDataset.__getitem__(dataset, 0)

    assert result["peak_status"] == 1
