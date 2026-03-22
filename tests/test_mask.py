import pybigtools
import pytest
import torch

from cerberus.interval import Interval
from cerberus.mask import BigBedMaskExtractor, InMemoryBigBedMaskExtractor


@pytest.fixture
def bigbed_file(tmp_path):
    path = tmp_path / "test.bb"
    chroms = {"chr1": 1000}
    # (chrom, start, end, rest)
    # 100-200, 300-400
    entries = [("chr1", 100, 200, "."), ("chr1", 300, 400, ".")]

    bb = pybigtools.open(str(path), "w")  # type: ignore
    bb.write(chroms, entries)

    return path


def test_mask_extractor_basic(bigbed_file):
    extractor = BigBedMaskExtractor({"test": bigbed_file})

    # Case 1: Overlap with first interval 100-200
    # Query: 150-250.
    # Expect: 150-200 -> 1, 200-250 -> 0.
    interval = Interval("chr1", 150, 250, "+")
    mask = extractor.extract(interval)

    assert mask.shape == (1, 100)
    assert torch.all(mask[0, 0:50] == 1.0)
    assert torch.all(mask[0, 50:100] == 0.0)

    # Case 2: No overlap
    interval = Interval("chr1", 500, 600, "+")
    mask = extractor.extract(interval)
    assert torch.all(mask == 0.0)

    # Case 3: Full overlap inside 300-400
    interval = Interval("chr1", 320, 350, "+")
    mask = extractor.extract(interval)
    assert torch.all(mask == 1.0)


def test_mask_extractor_multi_channel(bigbed_file):
    # Use same file for 2 channels
    extractor = BigBedMaskExtractor({"A": bigbed_file, "B": bigbed_file})
    interval = Interval("chr1", 150, 250, "+")
    mask = extractor.extract(interval)

    assert mask.shape == (2, 100)
    assert torch.equal(mask[0], mask[1])
    assert torch.all(mask[0, 0:50] == 1.0)


def test_in_memory_mask_extractor(bigbed_file):
    extractor = InMemoryBigBedMaskExtractor({"test": bigbed_file})

    # Check cache population
    assert "test" in extractor._cache
    assert "chr1" in extractor._cache["test"]
    cached_mask = extractor._cache["test"]["chr1"]

    assert cached_mask.shape == (1000,)
    assert torch.all(cached_mask[100:200] == 1.0)
    assert torch.all(cached_mask[300:400] == 1.0)
    assert torch.all(cached_mask[0:100] == 0.0)
    assert torch.all(cached_mask[200:300] == 0.0)

    # Check extraction
    interval = Interval("chr1", 150, 250, "+")
    mask = extractor.extract(interval)

    assert mask.shape == (1, 100)
    assert torch.all(mask[0, 0:50] == 1.0)
    assert torch.all(mask[0, 50:100] == 0.0)


def test_mask_extractor_pickle(bigbed_file):
    import pickle

    extractor = BigBedMaskExtractor({"test": bigbed_file})
    extractor.extract(Interval("chr1", 0, 10, "+"))  # Trigger load

    assert extractor._bigbed_files is not None

    dumped = pickle.dumps(extractor)
    loaded = pickle.loads(dumped)

    assert loaded._bigbed_files is None
    # Check it works after reload
    mask = loaded.extract(Interval("chr1", 100, 110, "+"))
    assert torch.all(mask == 1.0)
