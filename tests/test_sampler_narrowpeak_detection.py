"""Tests for case-insensitive narrowPeak detection in IntervalSampler."""

import gzip
from pathlib import Path

from cerberus.samplers import IntervalSampler

# 10-column narrowPeak content (tab-separated).
# Coordinates are large enough so that centering with padded_size=1000 stays valid.
NARROWPEAK_CONTENT = (
    "chr1\t10000\t10500\tpeak1\t500\t.\t10.5\t5.2\t3.1\t250\n"
    "chr1\t20000\t20500\tpeak2\t300\t+\t8.0\t4.0\t2.0\t200\n"
)

CHROM_SIZES = {f"chr{i}": 10**9 for i in range(1, 23)}


class TestIsNarrowpeak:
    """Tests for IntervalSampler._is_narrowpeak static method."""

    def test_plain_narrowpeak(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.narrowPeak"))

    def test_narrowpeak_gz(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.narrowPeak.gz"))

    def test_narrowpeak_bed(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.narrowPeak.bed"))

    def test_narrowpeak_bed_gz(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.narrowPeak.bed.gz"))

    def test_narrowpeak_uppercase(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.NARROWPEAK"))

    def test_narrowpeak_mixed_case(self):
        assert IntervalSampler._is_narrowpeak(Path("peaks.NarrowPeak.bed.gz"))

    def test_plain_bed_not_narrowpeak(self):
        assert not IntervalSampler._is_narrowpeak(Path("peaks.bed"))

    def test_bed_gz_not_narrowpeak(self):
        assert not IntervalSampler._is_narrowpeak(Path("peaks.bed.gz"))

    def test_bigbed_not_narrowpeak(self):
        assert not IntervalSampler._is_narrowpeak(Path("peaks.bb"))


class TestIntervalSamplerNarrowPeakLoading:
    """Tests that IntervalSampler uses _load_narrowPeak for all narrowPeak variants."""

    def _write_narrowpeak(self, path: Path, compress: bool = False) -> None:
        if compress:
            with gzip.open(path, "wt") as f:
                f.write(NARROWPEAK_CONTENT)
        else:
            path.write_text(NARROWPEAK_CONTENT)

    def _make_sampler(self, path: Path) -> IntervalSampler:
        return IntervalSampler(
            file_path=path,
            chrom_sizes=CHROM_SIZES,
            padded_size=1000,
        )

    def test_plain_narrowpeak(self, tmp_path):
        p = tmp_path / "peaks.narrowPeak"
        self._write_narrowpeak(p)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_narrowpeak_gz(self, tmp_path):
        p = tmp_path / "peaks.narrowPeak.gz"
        self._write_narrowpeak(p, compress=True)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_narrowpeak_bed(self, tmp_path):
        p = tmp_path / "peaks.narrowPeak.bed"
        self._write_narrowpeak(p)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_narrowpeak_bed_gz(self, tmp_path):
        p = tmp_path / "peaks.narrowPeak.bed.gz"
        self._write_narrowpeak(p, compress=True)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_narrowpeak_uppercase(self, tmp_path):
        p = tmp_path / "peaks.NARROWPEAK"
        self._write_narrowpeak(p)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_narrowpeak_mixed_case_gz(self, tmp_path):
        p = tmp_path / "peaks.NarrowPeak.gz"
        self._write_narrowpeak(p, compress=True)
        sampler = self._make_sampler(p)
        assert len(sampler) == 2

    def test_summit_used_for_centering(self, tmp_path):
        """Verify summit offset (col 10) is used when loading narrowPeak."""
        # Summit at offset 250 from start=10000 → center=10250
        # padded_size=200 → start=10250-100=10150, end=10250+100=10350
        p = tmp_path / "peaks.narrowPeak.bed.gz"
        content = "chr1\t10000\t10500\tpeak1\t500\t.\t10.5\t5.2\t3.1\t250\n"
        with gzip.open(p, "wt") as f:
            f.write(content)

        sampler = IntervalSampler(
            file_path=p,
            chrom_sizes=CHROM_SIZES,
            padded_size=200,
        )
        assert len(sampler) == 1
        interval = sampler._intervals[0]
        # Center should be at start + summit_offset = 10000 + 250 = 10250
        expected_start = 10250 - 100  # 10150
        expected_end = 10250 + 100  # 10350
        assert interval.start == expected_start
        assert interval.end == expected_end
