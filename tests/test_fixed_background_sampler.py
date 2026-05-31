"""Tests for FixedBackgroundSampler and PeakFixedBackgroundSampler.

These verify that Cerberus can train on a fixed, externally-supplied negative
set (e.g. a GC-matched ``negatives.bed``) — matching reference
chrombpnet-pytorch — while keeping peak vs background distinguishable after
fold splitting.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from interlap import InterLap

from cerberus.config import SamplerConfig
from cerberus.samplers import (
    PEAK_INTERVAL_SOURCES,
    FixedBackgroundSampler,
    MultiSampler,
    PeakFixedBackgroundSampler,
    create_sampler,
)

CHROM_SIZES = {"chr1": 100_000, "chr2": 100_000, "chr3": 100_000}
PADDED = 2114


def _build_folds() -> list[dict[str, InterLap]]:
    """One fold per chromosome: fold 0 -> chr1, fold 1 -> chr2, fold 2 -> chr3."""
    folds = []
    for chrom in ("chr1", "chr2", "chr3"):
        tree = InterLap()
        tree.add((0, CHROM_SIZES[chrom] - 1))
        folds.append({chrom: tree})
    return folds


def _write_bed(path: Path, rows: list[tuple[str, int, int, str]]) -> None:
    with open(path, "w") as fh:
        for chrom, start, end, name in rows:
            # 10-col narrowPeak-like; summit centered (matches negatives.bed layout)
            summit = (end - start) // 2
            fh.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\t0\t0\t0\t{summit}\n")


class TestFixedBackgroundSampler(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.peaks_path = tmp / "peaks.bed"
        self.neg_path = tmp / "negatives.bed"

        # 2 peaks + 2 negatives per chromosome, well inside bounds.
        peak_rows, neg_rows = [], []
        for ci, chrom in enumerate(("chr1", "chr2", "chr3")):
            for k in range(2):
                c = 20_000 + ci * 1000 + k * 5000
                peak_rows.append((chrom, c - 250, c + 250, f"peak_{chrom}_{k}"))
                cn = 60_000 + ci * 1000 + k * 5000
                neg_rows.append((chrom, cn - 250, cn + 250, f"background_{chrom}_{k}"))
        _write_bed(self.peaks_path, peak_rows)
        _write_bed(self.neg_path, neg_rows)
        self.folds = _build_folds()

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self) -> PeakFixedBackgroundSampler:
        return PeakFixedBackgroundSampler(
            intervals_path=self.peaks_path,
            background_intervals_path=self.neg_path,
            chrom_sizes=CHROM_SIZES,
            padded_size=PADDED,
            folds=self.folds,
            exclude_intervals={},
            seed=42,
        )

    def test_mixes_peaks_and_fixed_negatives(self):
        s = self._make()
        self.assertEqual(len(s.positives), 6)
        self.assertEqual(len(s.negatives), 6)
        self.assertEqual(len(s), 12)
        self.assertIsInstance(s.negatives, FixedBackgroundSampler)

    def test_source_labels_distinguish_peak_from_background(self):
        s = self._make()
        sources = {s.get_interval_source(i) for i in range(len(s))}
        self.assertEqual(sources, {"IntervalSampler", "FixedBackgroundSampler"})

    def test_split_folds_preserves_distinct_background_label(self):
        """After split, peaks -> 'ListSampler', negatives -> 'FixedBackgroundSampler'.

        This is the critical property: the base ListSampler.split_folds would
        return plain ListSampler for both, collapsing the label and breaking
        peak/background separation in evaluation.
        """
        s = self._make()
        _train, _val, test = s.split_folds(test_fold=0, val_fold=1)
        self.assertIsInstance(test, MultiSampler)

        sources = [test.get_interval_source(i) for i in range(len(test))]
        # test fold = chr1 -> 2 peaks + 2 negatives
        self.assertEqual(sources.count("ListSampler"), 2)
        self.assertEqual(sources.count("FixedBackgroundSampler"), 2)

        # Every interval in the test split is on chr1
        self.assertTrue(all(test[i].chrom == "chr1" for i in range(len(test))))

        # Background intervals must NOT be mislabeled as peaks ("ListSampler")
        bg = [
            test[i]
            for i in range(len(test))
            if test.get_interval_source(i) == "FixedBackgroundSampler"
        ]
        self.assertEqual(len(bg), 2)

    def test_peak_interval_sources_separates_post_split(self):
        """PEAK_INTERVAL_SOURCES is the shared peak/background predicate.

        After split_folds, peaks report "ListSampler" and fixed negatives
        report "FixedBackgroundSampler"; the shared constant must classify the
        former as peak and the latter as background (the predicate used by the
        training loss and evaluation interval selection).
        """
        _train, _val, test = self._make().split_folds(test_fold=0, val_fold=1)
        sources = [test.get_interval_source(i) for i in range(len(test))]
        peaks = [s for s in sources if s in PEAK_INTERVAL_SOURCES]
        background = [s for s in sources if s not in PEAK_INTERVAL_SOURCES]
        self.assertEqual(len(peaks), 2)
        self.assertEqual(set(peaks), {"ListSampler"})
        self.assertEqual(set(background), {"FixedBackgroundSampler"})

    def test_resample_is_noop_fixed_set(self):
        """Negatives are static: resample never changes the set (vs ComplexityMatched)."""
        s = self._make()
        before = sorted((iv.chrom, iv.start, iv.end) for iv in s.negatives)
        s.resample(seed=999)
        after = sorted((iv.chrom, iv.start, iv.end) for iv in s.negatives)
        self.assertEqual(before, after)

    def test_create_sampler_peak_fixed_background(self):
        config = SamplerConfig(
            sampler_type="peak_fixed_background",
            padded_size=PADDED,
            sampler_args={
                "intervals_path": self.peaks_path,
                "background_intervals_path": self.neg_path,
            },
        )
        sampler = create_sampler(
            config=config,
            chrom_sizes=CHROM_SIZES,
            folds=self.folds,
            exclude_intervals={},
        )
        self.assertIsInstance(sampler, PeakFixedBackgroundSampler)
        self.assertEqual(len(sampler), 12)


if __name__ == "__main__":
    unittest.main()
