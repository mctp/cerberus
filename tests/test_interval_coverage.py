"""Coverage tests for cerberus.interval — untested code paths."""
import pytest
import tempfile
from pathlib import Path
from typing import cast
from cerberus.config import GenomeConfig
from cerberus.interval import Interval, resolve_interval, merge_intervals, parse_intervals


# ---------------------------------------------------------------------------
# Interval methods
# ---------------------------------------------------------------------------

class TestIntervalMethods:

    def test_center(self):
        iv = Interval("chr1", 100, 200)
        centered = iv.center(50)
        assert centered.chrom == "chr1"
        assert centered.start == 125
        assert centered.end == 175
        assert len(centered) == 50

    def test_center_preserves_strand(self):
        iv = Interval("chr1", 0, 1000, "-")
        centered = iv.center(200)
        assert centered.strand == "-"

    def test_len(self):
        iv = Interval("chr1", 100, 300)
        assert len(iv) == 200

    def test_str(self):
        iv = Interval("chr2", 500, 1500, "-")
        assert str(iv) == "chr2:500-1500(-)"

    def test_str_default_strand(self):
        iv = Interval("chrX", 0, 100)
        assert str(iv) == "chrX:0-100(+)"


# ---------------------------------------------------------------------------
# resolve_interval edge cases
# ---------------------------------------------------------------------------

class TestResolveInterval:

    def test_tuple_length_less_than_3_raises(self):
        with pytest.raises(ValueError, match="Invalid interval tuple"):
            resolve_interval(("chr1", 100))

    def test_list_length_less_than_3_raises(self):
        with pytest.raises(ValueError, match="Invalid interval tuple"):
            resolve_interval(["chr1"])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported interval query type"):
            resolve_interval(12345)  # type: ignore[arg-type]

    def test_invalid_string_format_raises(self):
        with pytest.raises(ValueError, match="Invalid interval string format"):
            resolve_interval("chr1_100_200")

    def test_interval_passthrough(self):
        iv = Interval("chr1", 100, 200)
        assert resolve_interval(iv) is iv

    def test_string_format(self):
        iv = resolve_interval("chr1:100-200")
        assert iv.chrom == "chr1"
        assert iv.start == 100
        assert iv.end == 200

    def test_tuple_format(self):
        iv = resolve_interval(("chr2", 50, 150))
        assert iv.chrom == "chr2"
        assert iv.start == 50
        assert iv.end == 150


# ---------------------------------------------------------------------------
# merge_intervals
# ---------------------------------------------------------------------------

class TestMergeIntervals:

    def test_empty_list(self):
        result = merge_intervals([])
        assert result == []

    def test_non_overlapping_same_chrom(self):
        intervals = [
            Interval("chr1", 100, 200),
            Interval("chr1", 300, 400),
        ]
        result = merge_intervals(intervals)
        assert len(result) == 2

    def test_non_overlapping_different_chroms(self):
        intervals = [
            Interval("chr1", 100, 200),
            Interval("chr2", 100, 200),
        ]
        result = merge_intervals(intervals)
        assert len(result) == 2
        chroms = {r.chrom for r in result}
        assert chroms == {"chr1", "chr2"}

    def test_overlapping_merged(self):
        intervals = [
            Interval("chr1", 100, 300),
            Interval("chr1", 200, 400),
        ]
        result = merge_intervals(intervals)
        assert len(result) == 1
        assert result[0].start == 100
        assert result[0].end == 400

    def test_adjacent_merged(self):
        intervals = [
            Interval("chr1", 100, 200),
            Interval("chr1", 200, 300),
        ]
        result = merge_intervals(intervals)
        assert len(result) == 1
        assert result[0].start == 100
        assert result[0].end == 300


# ---------------------------------------------------------------------------
# parse_intervals
# ---------------------------------------------------------------------------

class TestParseIntervals:

    def _make_genome_config(self) -> GenomeConfig:
        return GenomeConfig.model_construct(
            chrom_sizes={"chr1": 1000000, "chr2": 500000},
            allowed_chroms=["chr1", "chr2"],
        )

    def test_bed_file_parsing(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\nchr2\t300\t400\n")
        gc = self._make_genome_config()
        result = parse_intervals([], [bed], gc)
        assert len(result) == 2
        assert result[0].chrom == "chr1"
        assert result[0].start == 100
        assert result[0].end == 200
        assert result[1].chrom == "chr2"

    def test_default_whole_genome(self):
        gc = self._make_genome_config()
        result = parse_intervals([], [], gc)
        assert len(result) == 2  # chr1 and chr2

    def test_string_chrom_only(self):
        gc = self._make_genome_config()
        result = parse_intervals(["chr1"], [], gc)
        assert len(result) == 1
        assert result[0].chrom == "chr1"
        assert result[0].start == 0
        assert result[0].end == 1000000

    def test_string_with_coords(self):
        gc = self._make_genome_config()
        result = parse_intervals(["chr1:500-1000"], [], gc)
        assert len(result) == 1
        assert result[0].start == 500
        assert result[0].end == 1000
