"""Tests for region-level ``bed_partition`` folds.

Covers BED parsing / coordinate conversion, fold-id validation, cross-fold
disjointness, centre-based single-owner assignment, and the corner cases that
region folds introduce but chromosome folds cannot (boundary-straddling
intervals, inter-fold buffer zones).  Also checks that the packaged Borzoi
fold definitions load cleanly.
"""

import gzip

import pytest

from cerberus.genome import (
    _parse_fold_id,
    create_genome_folds,
    fold_bed_path,
)
from cerberus.interval import Interval
from cerberus.samplers import owning_fold, partition_intervals_by_fold


def _write_bed(path, rows):
    with open(path, "w") as f:
        for chrom, start, end, fold in rows:
            f.write(f"{chrom}\t{start}\t{end}\t{fold}\n")
    return path


# ---------------------------------------------------------------------------
# _parse_fold_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [("0", 0), ("3", 3), ("fold0", 0), ("fold7", 7), ("fold_2", 2), ("FOLD5", 5)],
)
def test_parse_fold_id_accepts_int_and_labels(token, expected):
    assert _parse_fold_id(token) == expected


def test_parse_fold_id_rejects_garbage():
    with pytest.raises(ValueError, match="Invalid fold id"):
        _parse_fold_id("xyz")


# ---------------------------------------------------------------------------
# _create_folds_bed_partition (via create_genome_folds dispatch)
# ---------------------------------------------------------------------------


def test_bed_partition_basic_parse_and_coords(tmp_path):
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(
        tmp_path / "folds.bed",
        [("chr1", 0, 300, 0), ("chr1", 400, 700, 1), ("chr1", 800, 1000, 2)],
    )
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 3, "path": str(bed)}
    )
    assert len(folds) == 3
    # Half-open [0,300) stored as closed [0,299]; query the interior.
    assert (150, 150) in folds[0]["chr1"]
    assert (299, 299) in folds[0]["chr1"]
    # 300 is the exclusive end -> not owned by fold 0.
    assert (300, 300) not in folds[0]["chr1"]
    assert (500, 500) in folds[1]["chr1"]
    assert (900, 900) in folds[2]["chr1"]


def test_bed_partition_gzip(tmp_path):
    chrom_sizes = {"chr1": 1000}
    bed = tmp_path / "folds.bed.gz"
    with gzip.open(bed, "wt") as f:
        f.write("chr1\t0\t500\tfold0\n")
        f.write("chr1\t500\t1000\tfold1\n")
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )
    assert (250, 250) in folds[0]["chr1"]
    assert (750, 750) in folds[1]["chr1"]


def test_bed_partition_empty_folds_allowed(tmp_path):
    """k may exceed the number of fold ids present; missing folds are empty."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chr1", 500, 1000, 2)])
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 5, "path": str(bed)}
    )
    assert len(folds) == 5
    assert folds[0] and folds[2]
    assert folds[1] == {} and folds[3] == {} and folds[4] == {}


def test_bed_partition_fold_id_out_of_range_raises(tmp_path):
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chr1", 500, 1000, 5)])
    with pytest.raises(ValueError, match="out of range"):
        create_genome_folds(chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)})


def test_bed_partition_too_few_columns_raises(tmp_path):
    chrom_sizes = {"chr1": 1000}
    bed = tmp_path / "f.bed"
    with open(bed, "w") as f:
        f.write("chr1\t0\t500\n")  # missing fold id
    with pytest.raises(ValueError, match=">=4 columns"):
        create_genome_folds(chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)})


def test_bed_partition_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Fold BED not found"):
        create_genome_folds(
            {"chr1": 1000},
            "bed_partition",
            {"k": 2, "path": str(tmp_path / "nope.bed")},
        )


def test_bed_partition_cross_fold_overlap_raises(tmp_path):
    """Two different folds claiming the same bp is an error (ambiguous owner)."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(
        tmp_path / "f.bed", [("chr1", 0, 600, 0), ("chr1", 500, 1000, 1)]
    )  # overlap [500,600)
    with pytest.raises(ValueError, match="must be disjoint"):
        create_genome_folds(chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)})


def test_bed_partition_within_fold_overlap_allowed(tmp_path):
    """Overlapping rows of the SAME fold (e.g. sliding windows) are fine."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(
        tmp_path / "f.bed",
        [("chr1", 0, 600, 0), ("chr1", 400, 1000, 0)],  # same fold, overlapping
    )
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 1, "path": str(bed)}
    )
    assert (500, 500) in folds[0]["chr1"]


def test_bed_partition_adjacent_different_folds_ok(tmp_path):
    """Half-open-adjacent regions of different folds do not 'overlap'."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(
        tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chr1", 500, 1000, 1)]
    )  # touch at 500, no shared bp
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )
    assert (499, 499) in folds[0]["chr1"]
    assert (500, 500) in folds[1]["chr1"]


def test_bed_partition_skips_unknown_chrom(tmp_path, caplog):
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chrZ", 0, 500, 1)])
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )
    assert "chr1" in folds[0]
    assert folds[1] == {}  # chrZ row skipped


# ---------------------------------------------------------------------------
# owning_fold — centre-based single ownership
# ---------------------------------------------------------------------------


def test_owning_fold_straddling_interval_single_owner(tmp_path):
    """A peak straddling the fold0/fold1 boundary is owned by exactly the fold
    containing its centre — never double-assigned (the headline region case)."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chr1", 500, 1000, 1)])
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )

    # Interval [450, 550): centre 500 -> owned by fold 1 only.
    assert owning_fold(folds, "chr1", 450, 550) == 1
    # Interval [400, 480): centre 440 -> fold 0.
    assert owning_fold(folds, "chr1", 400, 480) == 0


def test_owning_fold_none_when_uncovered(tmp_path):
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 300, 0), ("chr1", 700, 1000, 1)])
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )
    # Centre 500 falls in the gap [300,700) -> no owner.
    assert owning_fold(folds, "chr1", 450, 550) is None
    # Chromosome absent from all folds -> no owner.
    assert owning_fold(folds, "chrX", 0, 100) is None


# ---------------------------------------------------------------------------
# partition_intervals_by_fold — routing + drop semantics
# ---------------------------------------------------------------------------


def test_partition_drops_buffer_zone_intervals(tmp_path):
    """An interval centred in an inter-fold buffer is dropped, NOT dumped into
    train (the key difference from the old overlap-based rule)."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(
        tmp_path / "f.bed", [("chr1", 0, 300, 0), ("chr1", 700, 1000, 1)]
    )  # buffer = [300,700)
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )

    in_fold0 = Interval("chr1", 100, 200, "+")  # centre 150 -> fold 0
    in_buffer = Interval("chr1", 450, 550, "+")  # centre 500 -> dropped
    in_fold1 = Interval("chr1", 800, 900, "+")  # centre 850 -> fold 1

    train, val, test = partition_intervals_by_fold(
        [in_fold0, in_buffer, in_fold1], folds, test_fold=1, val_fold=None
    )
    assert train == [in_fold0]
    assert test == [in_fold1]
    assert val == []
    # buffer interval appears nowhere
    assert in_buffer not in train and in_buffer not in val and in_buffer not in test


def test_partition_test_equals_val_duplicates(tmp_path):
    """test_fold == val_fold sends owned intervals into both (parity with the
    historical chrom-fold overlap behaviour)."""
    chrom_sizes = {"chr1": 1000}
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 500, 0), ("chr1", 500, 1000, 1)])
    folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )

    iv = Interval("chr1", 100, 200, "+")  # fold 0
    train, val, test = partition_intervals_by_fold([iv], folds, test_fold=0, val_fold=0)
    assert test == [iv] and val == [iv]
    assert train == []


def test_partition_equivalent_to_chrom_partition_for_full_coverage(tmp_path):
    """When the BED tiles whole chromosomes, bed_partition reproduces
    chrom_partition routing exactly."""
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    # Mirror a 2-fold chrom split: chr1 -> fold 0, chr2 -> fold 1.
    bed = _write_bed(tmp_path / "f.bed", [("chr1", 0, 1000, 0), ("chr2", 0, 1000, 1)])
    bed_folds = create_genome_folds(
        chrom_sizes, "bed_partition", {"k": 2, "path": str(bed)}
    )
    intervals = [Interval("chr1", 100, 200, "+"), Interval("chr2", 100, 200, "+")]

    tr_b, va_b, te_b = partition_intervals_by_fold(intervals, bed_folds, 0, 1)
    assert [i.chrom for i in te_b] == ["chr1"]
    assert [i.chrom for i in va_b] == ["chr2"]
    assert tr_b == []


# ---------------------------------------------------------------------------
# fold_bed_path + packaged Borzoi data
# ---------------------------------------------------------------------------


def test_fold_bed_path_unknown_species_raises():
    with pytest.raises(ValueError, match="No packaged fold BED"):
        fold_bed_path("zebrafish")


@pytest.mark.parametrize("species", ["human", "mouse", "HUMAN"])
def test_fold_bed_path_resolves(species):
    p = fold_bed_path(species)
    assert p.exists() and p.suffix == ".gz"


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_packaged_borzoi_folds_load(species):
    """The shipped Borzoi fold files parse into 8 disjoint folds."""
    path = fold_bed_path(species)
    # Derive permissive chrom_sizes from the file itself so every row is kept.
    sizes: dict[str, int] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            c, _s, e, _lab = line.split()
            sizes[c] = max(sizes.get(c, 0), int(e) + 1)
    folds = create_genome_folds(sizes, "bed_partition", {"k": 8, "path": str(path)})
    assert len(folds) == 8
    assert all(folds)  # every fold non-empty
