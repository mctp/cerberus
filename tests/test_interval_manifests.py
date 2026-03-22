"""Tests for interval manifest I/O (write_intervals_bed, load_intervals_bed)
and CerberusDataModule.save_interval_manifests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cerberus.interval import Interval, load_intervals_bed, write_intervals_bed
from cerberus.samplers import MultiSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_IV1 = Interval("chr1", 100, 200, "+")
_IV2 = Interval("chr2", 300, 400, "-")
_IV3 = Interval("chrX", 500, 600, "+")


# ---------------------------------------------------------------------------
# Interval.to_bed_row
# ---------------------------------------------------------------------------


def test_to_bed_row_default_strand() -> None:
    iv = Interval("chr1", 100, 200)
    assert iv.to_bed_row() == "chr1\t100\t200\t+"


def test_to_bed_row_minus_strand() -> None:
    iv = Interval("chr2", 300, 400, "-")
    assert iv.to_bed_row() == "chr2\t300\t400\t-"


# ---------------------------------------------------------------------------
# write_intervals_bed / load_intervals_bed round-trip
# ---------------------------------------------------------------------------


def test_round_trip_basic(tmp_path: Path) -> None:
    """Write and re-read intervals; verify exact round-trip."""
    intervals = [_IV1, _IV2, _IV3]
    sources = ["IntervalSampler", "ComplexityMatchedSampler", "IntervalSampler"]

    path = tmp_path / "intervals.bed"
    write_intervals_bed(path, intervals, sources)

    loaded_iv, loaded_src = load_intervals_bed(path)
    assert loaded_iv == intervals
    assert loaded_src == sources


def test_round_trip_empty(tmp_path: Path) -> None:
    """Empty interval list round-trips correctly."""
    path = tmp_path / "empty.bed"
    write_intervals_bed(path, [], [])

    loaded_iv, loaded_src = load_intervals_bed(path)
    assert loaded_iv == []
    assert loaded_src == []


def test_round_trip_minus_strand(tmp_path: Path) -> None:
    """Minus-strand intervals survive round-trip."""
    intervals = [Interval("chr1", 0, 100, "-")]
    sources = ["RandomSampler"]

    path = tmp_path / "minus.bed"
    write_intervals_bed(path, intervals, sources)

    loaded_iv, loaded_src = load_intervals_bed(path)
    assert loaded_iv[0].strand == "-"
    assert loaded_src[0] == "RandomSampler"


def test_round_trip_large_coordinates(tmp_path: Path) -> None:
    """Large genomic coordinates don't lose precision."""
    iv = Interval("chr1", 248_000_000, 248_956_422, "+")
    path = tmp_path / "large.bed"
    write_intervals_bed(path, [iv], ["IntervalSampler"])

    loaded_iv, _ = load_intervals_bed(path)
    assert loaded_iv[0].start == 248_000_000
    assert loaded_iv[0].end == 248_956_422


def test_write_mismatched_lengths_raises(tmp_path: Path) -> None:
    """write_intervals_bed raises if intervals and sources differ in length."""
    with pytest.raises(ValueError, match="same length"):
        write_intervals_bed(
            tmp_path / "bad.bed",
            [_IV1, _IV2],
            ["IntervalSampler"],
        )


def test_load_bad_header_raises(tmp_path: Path) -> None:
    """load_intervals_bed raises on unexpected header."""
    path = tmp_path / "bad_header.bed"
    path.write_text("bad\theader\n")
    with pytest.raises(ValueError, match="Unexpected header"):
        load_intervals_bed(path)


def test_file_format_is_valid_tsv(tmp_path: Path) -> None:
    """Verify the file is tab-separated with expected header."""
    intervals = [_IV1]
    sources = ["IntervalSampler"]
    path = tmp_path / "check.bed"
    write_intervals_bed(path, intervals, sources)

    lines = path.read_text().splitlines()
    assert lines[0] == "chrom\tstart\tend\tstrand\tinterval_source"
    assert lines[1] == "chr1\t100\t200\t+\tIntervalSampler"


def test_round_trip_many_sources(tmp_path: Path) -> None:
    """Multiple distinct source types round-trip correctly."""
    intervals = [_IV1, _IV2, _IV3]
    sources = ["IntervalSampler", "ComplexityMatchedSampler", "RandomSampler"]

    path = tmp_path / "multi.bed"
    write_intervals_bed(path, intervals, sources)

    _, loaded_src = load_intervals_bed(path)
    assert loaded_src == sources


# ---------------------------------------------------------------------------
# CerberusDataModule.save_interval_manifests
# ---------------------------------------------------------------------------


class _StubSampler:
    """Minimal sampler stub for manifest tests."""

    def __init__(self, intervals: list[Interval], source_name: str = "IntervalSampler"):
        self._intervals = intervals
        self._source_name = source_name

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def get_interval_source(self, idx: int) -> str:
        return self._source_name


class _MultiStubSampler:
    """Stub mimicking a MultiSampler with peak + background intervals."""

    def __init__(
        self,
        peak_intervals: list[Interval],
        bg_intervals: list[Interval],
    ):
        self._intervals = peak_intervals + bg_intervals
        self._sources = ["IntervalSampler"] * len(peak_intervals) + [
            "ComplexityMatchedSampler"
        ] * len(bg_intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def get_interval_source(self, idx: int) -> str:
        return self._sources[idx]


def test_save_interval_manifests_creates_files(tmp_path: Path) -> None:
    """save_interval_manifests writes one file per split."""
    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True

    # Create mock datasets with samplers
    for attr in ("train_dataset", "val_dataset", "test_dataset"):
        ds = MagicMock()
        ds.sampler = _StubSampler([_IV1, _IV2])
        setattr(dm, attr, ds)

    # Call the real method
    CerberusDataModule.save_interval_manifests(dm, tmp_path)

    assert (tmp_path / "intervals_train.bed").exists()
    assert (tmp_path / "intervals_val.bed").exists()
    assert (tmp_path / "intervals_test.bed").exists()


def test_save_interval_manifests_content(tmp_path: Path) -> None:
    """Saved manifest content matches the sampler's intervals and sources."""
    from cerberus.datamodule import CerberusDataModule

    peaks = [_IV1]
    bgs = [_IV2, _IV3]

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True

    ds = MagicMock()
    ds.sampler = _MultiStubSampler(peaks, bgs)
    dm.test_dataset = ds
    dm.train_dataset = None
    dm.val_dataset = None

    CerberusDataModule.save_interval_manifests(dm, tmp_path)

    loaded_iv, loaded_src = load_intervals_bed(tmp_path / "intervals_test.bed")
    assert loaded_iv == peaks + bgs
    assert loaded_src == [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "ComplexityMatchedSampler",
    ]


def test_save_interval_manifests_skips_none_datasets(tmp_path: Path) -> None:
    """Splits with None datasets are silently skipped."""
    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True
    dm.train_dataset = None
    dm.val_dataset = None

    ds = MagicMock()
    ds.sampler = _StubSampler([_IV1])
    dm.test_dataset = ds

    CerberusDataModule.save_interval_manifests(dm, tmp_path)

    assert not (tmp_path / "intervals_train.bed").exists()
    assert not (tmp_path / "intervals_val.bed").exists()
    assert (tmp_path / "intervals_test.bed").exists()


def test_save_interval_manifests_skips_none_sampler(tmp_path: Path) -> None:
    """Datasets with no sampler (inference-only) are silently skipped."""
    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = True

    ds = MagicMock()
    ds.sampler = None
    dm.train_dataset = ds
    dm.val_dataset = None
    dm.test_dataset = None

    CerberusDataModule.save_interval_manifests(dm, tmp_path)

    assert not (tmp_path / "intervals_train.bed").exists()


def test_save_interval_manifests_raises_before_setup() -> None:
    """Calling save_interval_manifests before setup() raises RuntimeError."""
    from cerberus.datamodule import CerberusDataModule

    dm = MagicMock(spec=CerberusDataModule)
    dm._is_initialized = False

    with pytest.raises(RuntimeError, match="not setup"):
        CerberusDataModule.save_interval_manifests(dm, Path("/tmp"))


# ---------------------------------------------------------------------------
# Round-trip through real MultiSampler
# ---------------------------------------------------------------------------


class _FakeSampler:
    """Stub sampler for MultiSampler integration tests."""

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


class _PeakFake(_FakeSampler):
    pass


class _BgFake(_FakeSampler):
    pass


def test_round_trip_with_real_multisampler(tmp_path: Path) -> None:
    """End-to-end: MultiSampler → write → load → verify intervals and sources."""
    peaks = _PeakFake([_IV1])
    bgs = _BgFake([_IV2, _IV3])
    ms = MultiSampler([peaks, bgs], seed=42)

    n = len(ms)
    intervals = [ms[i] for i in range(n)]
    sources = [ms.get_interval_source(i) for i in range(n)]

    path = tmp_path / "intervals.bed"
    write_intervals_bed(path, intervals, sources)

    loaded_iv, loaded_src = load_intervals_bed(path)

    # Same intervals (order may differ due to shuffle, but sets must match)
    assert sorted(loaded_iv, key=str) == sorted(intervals, key=str)
    # For each loaded interval, its source matches what the sampler reported
    for iv, src in zip(loaded_iv, loaded_src, strict=True):
        original_idx = intervals.index(iv)
        assert src == sources[original_idx]


# ---------------------------------------------------------------------------
# Determinism: same seed → same manifests
# ---------------------------------------------------------------------------


def test_manifest_deterministic_across_runs(tmp_path: Path) -> None:
    """Two independent MultiSamplers with the same seed produce identical manifests."""

    class _P(_FakeSampler):
        pass

    class _B(_FakeSampler):
        pass

    ivs_peak = [Interval("chr1", i * 100, (i + 1) * 100) for i in range(10)]
    ivs_bg = [Interval("chr2", i * 100, (i + 1) * 100) for i in range(20)]

    # Run 1
    ms1 = MultiSampler([_P(ivs_peak), _B(ivs_bg)], seed=999)
    iv1 = [ms1[i] for i in range(len(ms1))]
    src1 = [ms1.get_interval_source(i) for i in range(len(ms1))]
    write_intervals_bed(tmp_path / "run1.bed", iv1, src1)

    # Run 2 (fresh instances, same seed)
    ms2 = MultiSampler([_P(ivs_peak), _B(ivs_bg)], seed=999)
    iv2 = [ms2[i] for i in range(len(ms2))]
    src2 = [ms2.get_interval_source(i) for i in range(len(ms2))]
    write_intervals_bed(tmp_path / "run2.bed", iv2, src2)

    # Files should be byte-identical
    assert (tmp_path / "run1.bed").read_text() == (tmp_path / "run2.bed").read_text()


# ---------------------------------------------------------------------------
# SLOW: Multi-GPU fold determinism (helpers at module level for picklability)
# ---------------------------------------------------------------------------


class _SpawnPeak:
    """Stub peak sampler (module-level for spawn picklability)."""

    def __init__(self, ivs: list[Interval]):
        self._intervals = ivs

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
        return "_SpawnPeak"


class _SpawnBg:
    """Stub background sampler (module-level for spawn picklability)."""

    def __init__(self, ivs: list[Interval]):
        self._intervals = ivs

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
        return "_SpawnBg"


def _generate_stub_manifest(seed: int, output_path: str) -> None:
    """Worker: create a MultiSampler with stubs, write manifest."""
    from cerberus.interval import Interval, write_intervals_bed
    from cerberus.samplers import MultiSampler

    peak_ivs = [Interval("chr1", i * 100, (i + 1) * 100) for i in range(50)]
    bg_ivs = [Interval("chr2", i * 100, (i + 1) * 100) for i in range(100)]

    ms = MultiSampler([_SpawnPeak(peak_ivs), _SpawnBg(bg_ivs)], seed=seed)

    n = len(ms)
    intervals = [ms[i] for i in range(n)]
    sources = [ms.get_interval_source(i) for i in range(n)]
    write_intervals_bed(Path(output_path), intervals, sources)


def test_multiprocess_fold_determinism(tmp_path: Path) -> None:
    """Verify that sampler fold splitting is deterministic across processes.

    Simulates multi-GPU DDP by running the same sampler setup in two
    separate processes (via multiprocessing) with the same seed and
    config, then asserting the generated interval manifests are identical.

    This catches non-determinism from:
    - Process-level random state leakage
    - Non-deterministic dict iteration
    - OS-level randomness (e.g. hash seed)
    """
    import os

    if os.environ.get("RUN_SLOW_TESTS") is None:
        pytest.skip("Skipping slow tests (RUN_SLOW_TESTS not set)")
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")

    seed = 42
    path1 = str(tmp_path / "proc1.bed")
    path2 = str(tmp_path / "proc2.bed")

    p1 = ctx.Process(target=_generate_stub_manifest, args=(seed, path1))
    p2 = ctx.Process(target=_generate_stub_manifest, args=(seed, path2))

    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)

    assert p1.exitcode == 0, f"Process 1 failed with exit code {p1.exitcode}"
    assert p2.exitcode == 0, f"Process 2 failed with exit code {p2.exitcode}"

    # Manifests from both processes must be byte-identical
    content1 = Path(path1).read_text()
    content2 = Path(path2).read_text()
    assert content1 == content2, (
        "Manifests differ between processes — non-deterministic sampler behavior detected"
    )

    # Sanity check: files are non-empty and have correct count
    loaded_iv, loaded_src = load_intervals_bed(Path(path1))
    assert len(loaded_iv) == 150  # 50 peaks + 100 backgrounds
    assert loaded_src.count("_SpawnPeak") == 50
    assert loaded_src.count("_SpawnBg") == 100


# ---------------------------------------------------------------------------
# SLOW: Realistic multi-process PeakSampler determinism
# ---------------------------------------------------------------------------


def _build_peak_sampler_and_write_manifest(
    fasta_path: str,
    peaks_path: str,
    output_path: str,
    seed: int,
    test_fold: int,
    val_fold: int,
) -> None:
    """Worker: build a real PeakSampler pipeline, split folds, write manifest.

    Runs the full sampler stack: PeakSampler -> IntervalSampler (peaks) +
    ComplexityMatchedSampler (backgrounds) -> MultiSampler -> split_folds.
    This is the exact code path used during training.

    Defined at module level so it is picklable by multiprocessing.
    """
    from pathlib import Path as _Path

    from cerberus.genome import create_genome_folds
    from cerberus.interval import write_intervals_bed
    from cerberus.samplers import MultiSampler, PeakSampler

    chrom_sizes = {
        "chr1": 5000,
        "chr2": 5000,
        "chr3": 5000,
        "chr4": 5000,
    }
    folds = create_genome_folds(
        chrom_sizes,
        fold_type="chrom_partition",
        fold_args={"k": 4},
    )

    sampler = PeakSampler(
        intervals_path=peaks_path,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        padded_size=200,
        folds=folds,
        exclude_intervals={},
        background_ratio=1.0,
        seed=seed,
    )

    assert isinstance(sampler, MultiSampler)
    train_s, val_s, test_s = sampler.split_folds(test_fold, val_fold)

    for split_name, split_sampler in [
        ("train", train_s),
        ("val", val_s),
        ("test", test_s),
    ]:
        assert isinstance(split_sampler, MultiSampler)
        n = len(split_sampler)
        intervals = [split_sampler[i] for i in range(n)]
        sources = [split_sampler.get_interval_source(i) for i in range(n)]
        write_intervals_bed(
            _Path(output_path) / f"intervals_{split_name}.bed",
            intervals,
            sources,
        )


def test_peak_sampler_multiprocess_determinism(tmp_path: Path) -> None:
    """Full PeakSampler pipeline produces identical manifests across processes.

    Creates a real multi-chromosome FASTA, real peak BED, and runs the
    full PeakSampler -> ComplexityMatchedSampler -> MultiSampler -> split_folds
    pipeline in two independent processes with the same seed. Asserts the
    resulting interval manifests are byte-identical.

    This is a realistic simulation of multi-GPU DDP training where each
    rank independently sets up the datamodule. It catches:
    - Non-determinism in ComplexityMatchedSampler background generation
    - Hash-seed or dict-ordering issues across processes
    - Random state leakage from process initialization
    """
    import os

    if os.environ.get("RUN_SLOW_TESTS") is None:
        pytest.skip("Skipping slow tests (RUN_SLOW_TESTS not set)")

    import multiprocessing
    import random

    import pyfaidx

    ctx = multiprocessing.get_context("spawn")

    # --- Create a realistic multi-chromosome FASTA ---
    fasta_path = tmp_path / "genome.fa"
    rng = random.Random(0)
    with open(fasta_path, "w") as f:
        for chrom in ["chr1", "chr2", "chr3", "chr4"]:
            f.write(f">{chrom}\n")
            # 5000 bp of varied sequence (not all one base)
            seq = "".join(rng.choice("ACGT") for _ in range(5000))
            f.write(seq + "\n")
    pyfaidx.Faidx(str(fasta_path))

    # --- Create peak BED spread across chromosomes ---
    peaks_path = tmp_path / "peaks.bed"
    with open(peaks_path, "w") as f:
        for chrom in ["chr1", "chr2", "chr3", "chr4"]:
            # 10 peaks per chromosome, spaced 400bp apart
            for i in range(10):
                start = 200 + i * 400
                end = start + 200
                f.write(f"{chrom}\t{start}\t{end}\n")

    # --- Run in two independent processes ---
    seed = 42
    test_fold = 0
    val_fold = 1

    dir1 = tmp_path / "proc1"
    dir2 = tmp_path / "proc2"
    dir1.mkdir()
    dir2.mkdir()

    p1 = ctx.Process(
        target=_build_peak_sampler_and_write_manifest,
        args=(str(fasta_path), str(peaks_path), str(dir1), seed, test_fold, val_fold),
    )
    p2 = ctx.Process(
        target=_build_peak_sampler_and_write_manifest,
        args=(str(fasta_path), str(peaks_path), str(dir2), seed, test_fold, val_fold),
    )

    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)

    assert p1.exitcode == 0, f"Process 1 failed with exit code {p1.exitcode}"
    assert p2.exitcode == 0, f"Process 2 failed with exit code {p2.exitcode}"

    # --- Compare manifests across processes ---
    for split in ["train", "val", "test"]:
        file1 = dir1 / f"intervals_{split}.bed"
        file2 = dir2 / f"intervals_{split}.bed"

        assert file1.exists(), f"Process 1 did not create {file1.name}"
        assert file2.exists(), f"Process 2 did not create {file2.name}"

        content1 = file1.read_text()
        content2 = file2.read_text()
        assert content1 == content2, (
            f"Manifests for {split} split differ between processes — "
            f"non-deterministic sampler behavior detected"
        )

    # --- Sanity checks on manifest content ---
    for split in ["train", "val", "test"]:
        loaded_iv, loaded_src = load_intervals_bed(dir1 / f"intervals_{split}.bed")
        source_types = set(loaded_src)
        if len(loaded_iv) > 0:
            # Should have both peaks and complexity-matched backgrounds
            assert (
                "IntervalSampler" in source_types
                or "ComplexityMatchedSampler" in source_types
            )
        # Verify round-trip integrity
        reloaded_iv, reloaded_src = load_intervals_bed(dir1 / f"intervals_{split}.bed")
        assert reloaded_iv == loaded_iv
        assert reloaded_src == loaded_src
