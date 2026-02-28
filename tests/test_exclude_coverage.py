"""Coverage tests for cerberus.exclude — untested code paths."""
import pytest
from pathlib import Path
from cerberus.exclude import get_exclude_intervals, is_excluded


# ---------------------------------------------------------------------------
# get_exclude_intervals
# ---------------------------------------------------------------------------

class TestGetExcludeIntervals:

    def test_comment_and_header_lines_skipped(self, tmp_path):
        """Lines starting with #, track, or browser should be skipped."""
        bed = tmp_path / "exclude.bed"
        bed.write_text(
            "# comment line\n"
            "track name=test\n"
            "browser position chr1:100-200\n"
            "chr1\t100\t200\n"
            "chr1\t300\t400\n"
        )
        result = get_exclude_intervals({"test": bed})
        assert "chr1" in result
        intervals = list(result["chr1"])
        assert len(intervals) == 2

    def test_end_lte_start_skipped(self, tmp_path):
        """Lines where end <= start should be skipped."""
        bed = tmp_path / "bad.bed"
        bed.write_text(
            "chr1\t100\t100\n"  # end == start
            "chr1\t200\t150\n"  # end < start
            "chr1\t300\t400\n"  # valid
        )
        result = get_exclude_intervals({"bad": bed})
        assert "chr1" in result
        intervals = list(result["chr1"])
        assert len(intervals) == 1

    def test_empty_file(self, tmp_path):
        bed = tmp_path / "empty.bed"
        bed.write_text("")
        result = get_exclude_intervals({"empty": bed})
        assert len(result) == 0

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_exclude_intervals({"missing": tmp_path / "nonexistent.bed"})

    def test_multiple_files_merged(self, tmp_path):
        bed1 = tmp_path / "a.bed"
        bed1.write_text("chr1\t100\t200\n")
        bed2 = tmp_path / "b.bed"
        bed2.write_text("chr1\t300\t400\nchr2\t500\t600\n")
        result = get_exclude_intervals({"a": bed1, "b": bed2})
        assert "chr1" in result
        assert "chr2" in result
        chr1_intervals = list(result["chr1"])
        assert len(chr1_intervals) == 2

    def test_malformed_coordinates_skipped(self, tmp_path):
        """Lines with non-integer coordinates should be skipped."""
        bed = tmp_path / "malformed.bed"
        bed.write_text("chr1\tABC\tDEF\nchr1\t100\t200\n")
        result = get_exclude_intervals({"mal": bed})
        assert "chr1" in result
        intervals = list(result["chr1"])
        assert len(intervals) == 1


# ---------------------------------------------------------------------------
# is_excluded
# ---------------------------------------------------------------------------

class TestIsExcluded:

    def test_chrom_not_in_intervals(self, tmp_path):
        """Chromosome not in intervals dict should return False."""
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\n")
        intervals = get_exclude_intervals({"t": bed})
        assert is_excluded(intervals, "chr2", 100, 200) is False

    def test_empty_intervals(self):
        assert is_excluded({}, "chr1", 0, 100) is False

    def test_overlapping_region(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\n")
        intervals = get_exclude_intervals({"t": bed})
        assert is_excluded(intervals, "chr1", 150, 250) is True

    def test_non_overlapping_region(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\n")
        intervals = get_exclude_intervals({"t": bed})
        assert is_excluded(intervals, "chr1", 200, 300) is False
