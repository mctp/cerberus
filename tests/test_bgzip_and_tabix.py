"""Tests for _bgzip_and_tabix and _outputs_exist from tools/scatac_pseudobulk.py."""

import gzip
import importlib
import sys
from pathlib import Path

import pysam
import pytest


@pytest.fixture()
def _pseudobulk_mod():
    """Import the scatac_pseudobulk module."""
    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    sys.path.insert(0, str(tools_dir))
    mod = importlib.import_module("scatac_pseudobulk")
    sys.path.pop(0)
    return mod


@pytest.fixture()
def _import_pseudobulk(_pseudobulk_mod):
    """Import the _bgzip_and_tabix function from tools/scatac_pseudobulk.py."""
    return _pseudobulk_mod._bgzip_and_tabix


# Sample BED lines (unsorted on purpose)
UNSORTED_BED = (
    "chr2\t500\t600\tpeak3\t200\t.\t5.0\t3.0\t2.0\t50\n"
    "chr1\t300\t400\tpeak2\t150\t.\t4.0\t2.5\t1.5\t30\n"
    "chr1\t100\t200\tpeak1\t100\t.\t3.0\t2.0\t1.0\t10\n"
)

SORTED_CHROMS = ["chr1", "chr1", "chr2"]
SORTED_STARTS = [100, 300, 500]


class TestBgzipAndTabix:
    """Tests for the _bgzip_and_tabix helper."""

    def test_returns_gz_path(self, tmp_path: Path, _import_pseudobulk):
        """Returned path ends with .bed.gz."""
        bed = tmp_path / "test.narrowPeak.bed"
        bed.write_text(UNSORTED_BED)
        result = _import_pseudobulk(bed)
        assert result == tmp_path / "test.narrowPeak.bed.gz"

    def test_gz_file_created(self, tmp_path: Path, _import_pseudobulk):
        """The .bed.gz file exists on disk."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        gz = _import_pseudobulk(bed)
        assert gz.exists()

    def test_original_removed(self, tmp_path: Path, _import_pseudobulk):
        """The original uncompressed BED file is deleted."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        _import_pseudobulk(bed)
        assert not bed.exists()

    def test_tabix_index_created(self, tmp_path: Path, _import_pseudobulk):
        """A .tbi tabix index file is created alongside the .bed.gz."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        gz = _import_pseudobulk(bed)
        assert Path(str(gz) + ".tbi").exists()

    def test_content_sorted(self, tmp_path: Path, _import_pseudobulk):
        """Output is sorted by chrom then start position."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        gz = _import_pseudobulk(bed)
        with gzip.open(gz, "rt") as f:
            lines = [l.strip() for l in f if l.strip()]
        chroms = [l.split("\t")[0] for l in lines]
        starts = [int(l.split("\t")[1]) for l in lines]
        assert chroms == SORTED_CHROMS
        assert starts == SORTED_STARTS

    def test_content_preserved(self, tmp_path: Path, _import_pseudobulk):
        """All rows are preserved (no data loss)."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        gz = _import_pseudobulk(bed)
        with gzip.open(gz, "rt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_is_valid_bgzf(self, tmp_path: Path, _import_pseudobulk):
        """Output file is valid BGZF (readable by pysam.TabixFile)."""
        bed = tmp_path / "test.bed"
        bed.write_text(UNSORTED_BED)
        gz = _import_pseudobulk(bed)
        tbx = pysam.TabixFile(str(gz))
        assert "chr1" in tbx.contigs
        tbx.close()

    def test_empty_file(self, tmp_path: Path, _import_pseudobulk):
        """An empty BED file produces an empty .bed.gz without error."""
        bed = tmp_path / "empty.bed"
        bed.write_text("")
        gz = _import_pseudobulk(bed)
        assert gz.exists()
        assert not bed.exists()


class TestOutputsExist:
    """Tests for the _outputs_exist skip-check helper."""

    def test_all_exist_returns_true(self, tmp_path: Path, _pseudobulk_mod):
        """Returns True when all expected files exist and are non-empty."""
        paths = [tmp_path / "a.bw", tmp_path / "b.bw"]
        for p in paths:
            p.write_text("data")
        assert _pseudobulk_mod._outputs_exist(paths, "test stage") is True

    def test_none_exist_returns_false(self, tmp_path: Path, _pseudobulk_mod):
        """Returns False when no files exist."""
        paths = [tmp_path / "a.bw", tmp_path / "b.bw"]
        assert _pseudobulk_mod._outputs_exist(paths, "test stage") is False

    def test_partial_exist_returns_false(self, tmp_path: Path, _pseudobulk_mod):
        """Returns False when only some files exist."""
        paths = [tmp_path / "a.bw", tmp_path / "b.bw"]
        paths[0].write_text("data")
        assert _pseudobulk_mod._outputs_exist(paths, "test stage") is False

    def test_empty_list_returns_false(self, _pseudobulk_mod):
        """Returns False for an empty path list."""
        assert _pseudobulk_mod._outputs_exist([], "test stage") is False

    def test_empty_file_returns_false(self, tmp_path: Path, _pseudobulk_mod):
        """Returns False when files exist but are empty (truncated/corrupt)."""
        paths = [tmp_path / "a.bw", tmp_path / "b.bw"]
        for p in paths:
            p.write_text("")
        assert _pseudobulk_mod._outputs_exist(paths, "test stage") is False
