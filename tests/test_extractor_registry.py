from pathlib import Path

import pytest
import torch

from cerberus.interval import Interval
from cerberus.mask import (
    BedMaskExtractor,
    BigBedMaskExtractor,
    InMemoryBigBedMaskExtractor,
)
from cerberus.signal import (
    _EXTRACTOR_REGISTRY,
    InMemorySignalExtractor,
    SignalExtractor,
    UniversalExtractor,
    _resolve_extractor_cls,
    register_extractor,
)


class TestResolveExtractorCls:
    """Tests for the _resolve_extractor_cls helper."""

    def test_bigwig_default(self):
        assert (
            _resolve_extractor_cls(Path("track.bw"), in_memory=False) is SignalExtractor
        )

    def test_bigwig_in_memory(self):
        assert (
            _resolve_extractor_cls(Path("track.bw"), in_memory=True)
            is InMemorySignalExtractor
        )

    def test_bigwig_alternate_extension(self):
        assert (
            _resolve_extractor_cls(Path("track.bigwig"), in_memory=False)
            is SignalExtractor
        )

    def test_bigbed_default(self):
        assert (
            _resolve_extractor_cls(Path("mask.bb"), in_memory=False)
            is BigBedMaskExtractor
        )

    def test_bigbed_in_memory(self):
        assert (
            _resolve_extractor_cls(Path("mask.bb"), in_memory=True)
            is InMemoryBigBedMaskExtractor
        )

    def test_bigbed_alternate_extension(self):
        assert (
            _resolve_extractor_cls(Path("mask.bigbed"), in_memory=False)
            is BigBedMaskExtractor
        )

    def test_bed_default(self):
        assert (
            _resolve_extractor_cls(Path("peaks.bed"), in_memory=False)
            is BedMaskExtractor
        )

    def test_bed_in_memory_still_returns_default(self):
        # BedMaskExtractor has no in-memory variant (it's always in-memory)
        assert (
            _resolve_extractor_cls(Path("peaks.bed"), in_memory=True)
            is BedMaskExtractor
        )

    def test_bed_gz_compound_extension(self):
        assert (
            _resolve_extractor_cls(Path("peaks.bed.gz"), in_memory=False)
            is BedMaskExtractor
        )

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="No extractor registered"):
            _resolve_extractor_cls(Path("data.hdf5"), in_memory=False)


class TestRegisterExtractor:
    """Tests for the register_extractor function."""

    def test_register_new_extension(self):
        # Use a dummy class
        class DummyExtractor:
            pass

        register_extractor(".dummy", DummyExtractor)
        try:
            assert (
                _resolve_extractor_cls(Path("file.dummy"), in_memory=False)
                is DummyExtractor
            )
        finally:
            # Clean up registry
            _EXTRACTOR_REGISTRY.pop(".dummy", None)

    def test_register_with_in_memory_variant(self):
        class DummyExtractor:
            pass

        class DummyInMemoryExtractor:
            pass

        register_extractor(".dummy2", DummyExtractor, DummyInMemoryExtractor)
        try:
            assert (
                _resolve_extractor_cls(Path("f.dummy2"), in_memory=False)
                is DummyExtractor
            )
            assert (
                _resolve_extractor_cls(Path("f.dummy2"), in_memory=True)
                is DummyInMemoryExtractor
            )
        finally:
            _EXTRACTOR_REGISTRY.pop(".dummy2", None)

    def test_register_case_insensitive(self):
        class DummyExtractor:
            pass

        register_extractor(".DuMmY3", DummyExtractor)
        try:
            assert (
                _resolve_extractor_cls(Path("file.dummy3"), in_memory=False)
                is DummyExtractor
            )
        finally:
            _EXTRACTOR_REGISTRY.pop(".dummy3", None)


class TestUniversalExtractorRegistry:
    """Tests that UniversalExtractor correctly uses the registry."""

    def test_groups_same_type_channels(self, tmp_path):
        """Channels with same extension should share one extractor instance."""
        # Create two BED files
        bed1 = tmp_path / "peaks1.bed"
        bed2 = tmp_path / "peaks2.bed"
        bed1.write_text("chr1\t100\t200\n")
        bed2.write_text("chr1\t300\t400\n")

        extractor = UniversalExtractor(
            paths={"peak_a": bed1, "peak_b": bed2},
            in_memory=False,
        )

        # Both should resolve to BedMaskExtractor — one group
        assert len(extractor._extractors) == 1
        assert BedMaskExtractor in extractor._extractors

    def test_mixed_types_creates_separate_groups(self, tmp_path):
        """Different extensions should create separate extractor groups."""
        bed = tmp_path / "peaks.bed"
        bed.write_text("chr1\t100\t200\n")

        # Create a minimal bigwig
        import pybigtools

        bw_path = tmp_path / "signal.bw"
        bw = pybigtools.open(str(bw_path), "w")  # type: ignore[attr-defined]
        bw.write({"chr1": 1000}, [("chr1", 0, 1000, [0.0] * 1000)])

        extractor = UniversalExtractor(
            paths={"my_signal": bw_path, "my_peaks": bed},
            in_memory=False,
        )

        assert len(extractor._extractors) == 2

    def test_extract_correct_shape(self, tmp_path):
        """Extract should return (C, L) tensor with correct channel count."""
        bed1 = tmp_path / "a.bed"
        bed2 = tmp_path / "b.bed"
        bed1.write_text("chr1\t100\t200\n")
        bed2.write_text("chr1\t150\t250\n")

        extractor = UniversalExtractor(
            paths={"chan_a": bed1, "chan_b": bed2},
            in_memory=False,
        )

        result = extractor.extract(Interval("chr1", 100, 200, "+"))
        assert result.shape == (2, 100)

    def test_extract_correct_values(self, tmp_path):
        """Verify extracted values are correct for BED channels."""
        bed1 = tmp_path / "a.bed"
        bed2 = tmp_path / "b.bed"
        # bed1 covers 100-200 fully
        bed1.write_text("chr1\t100\t200\n")
        # bed2 covers 150-250 (only latter half of query)
        bed2.write_text("chr1\t150\t250\n")

        extractor = UniversalExtractor(
            paths={"chan_a": bed1, "chan_b": bed2},
            in_memory=False,
        )

        result = extractor.extract(Interval("chr1", 100, 200, "+"))
        # chan_a: fully covered -> all 1s
        assert torch.all(result[0] == 1.0)
        # chan_b: first 50 bases uncovered, last 50 covered
        assert torch.all(result[1, :50] == 0.0)
        assert torch.all(result[1, 50:] == 1.0)

    def test_unknown_extension_raises_in_constructor(self, tmp_path):
        """UniversalExtractor should raise ValueError for unregistered extensions."""
        fake = tmp_path / "data.xyz"
        fake.write_text("something")

        with pytest.raises(ValueError, match="No extractor registered"):
            UniversalExtractor(paths={"ch": fake})
