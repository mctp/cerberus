import os

import pytest
import torch

from cerberus.interval import Interval
from cerberus.signal import InMemorySignalExtractor, SignalExtractor


@pytest.fixture
def bigwig_path(mdapca2b_ar_dataset):
    return mdapca2b_ar_dataset["bigwig"]

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_signal_extractor_basic(bigwig_path):
    extractor = SignalExtractor({"test": bigwig_path})

    # 50M on chr1, length 200
    interval = Interval("chr1", 50000000, 50000200) # 200bp interval

    signal = extractor.extract(interval)

    # Raw extraction of full interval
    assert signal.shape == (1, 200)
    assert isinstance(signal, torch.Tensor)

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_signal_extractor_multiple_tracks(bigwig_path):
    # Use same file twice as two tracks
    paths = {"track1": bigwig_path, "track2": bigwig_path}
    extractor = SignalExtractor(paths)

    interval = Interval("chr1", 50000000, 50000200)
    signal = extractor.extract(interval)

    assert signal.shape == (2, 200)
    # Both tracks should be identical
    assert torch.allclose(signal[0], signal[1])

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow in-memory signal tests")
def test_in_memory_signal_extractor(bigwig_path):
    # Load only chr21 to avoid reading the full genome
    extractor = InMemorySignalExtractor({"test": bigwig_path}, chroms=["chr21"])

    interval = Interval("chr21", 0, 200)

    signal = extractor.extract(interval)

    assert signal.shape == (1, 200)
    assert isinstance(signal, torch.Tensor)

    # Verify values match disk extractor
    disk_extractor = SignalExtractor({"test": bigwig_path})
    disk_signal = disk_extractor.extract(interval)

    assert torch.allclose(signal, disk_signal)

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow in-memory signal tests")
def test_in_memory_signal_extractor_missing_chrom(bigwig_path):
    # Test gracefully handling missing chrom (returns zeros)
    extractor = InMemorySignalExtractor({"test": bigwig_path}, chroms=["chr21"])
    # 'chrFake' not in file or in the loaded chroms
    interval = Interval("chrFake", 0, 100)
    signal = extractor.extract(interval)
    assert signal.shape == (1, 100)
    assert torch.all(signal == 0)

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow in-memory signal tests")
def test_in_memory_signal_extractor_chroms_filter(bigwig_path):
    # Verify that only the requested chromosome is loaded into cache
    extractor = InMemorySignalExtractor({"test": bigwig_path}, chroms=["chr21"])
    assert "chr21" in extractor._cache["test"]
    assert "chr1" not in extractor._cache["test"]

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow in-memory signal tests")
def test_in_memory_signal_extractor_chroms_requested_missing(bigwig_path):
    # Requesting a chrom not present in the file should not raise; cache stays empty
    extractor = InMemorySignalExtractor({"test": bigwig_path}, chroms=["chrNotInFile"])
    assert extractor._cache["test"] == {}
    # Extraction returns zeros for the absent chrom
    signal = extractor.extract(Interval("chrNotInFile", 0, 50))
    assert signal.shape == (1, 50)
    assert torch.all(signal == 0)
