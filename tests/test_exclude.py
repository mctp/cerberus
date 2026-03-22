from interlap import InterLap

from cerberus.exclude import get_exclude_intervals, is_excluded


def test_get_exclude_intervals_basic(tmp_path):
    f = tmp_path / "exclude.bed"
    content = """chr1\t10\t20
chr1\t70\t80
"""
    f.write_text(content)

    # Pass as dict
    intervals = get_exclude_intervals({"ex": f})

    assert "chr1" in intervals
    assert "chr2" not in intervals

    # Check overlaps using closed interval logic [start, end-1]
    # BED 10-20 becomes [10, 19]

    # Query 10-20 [10, 19] -> Should overlap [10, 19]
    assert (10, 19) in intervals["chr1"]

    # Query 15-16 [15, 15] -> Should overlap [10, 19]
    assert (15, 15) in intervals["chr1"]

    # Query 70-80 [70, 79] -> Should overlap [70, 79]
    assert (70, 79) in intervals["chr1"]

    # No overlap
    # 0-10 -> [0, 9]. [0, 9] vs [10, 19] -> No overlap (9 < 10)
    assert (0, 9) not in intervals["chr1"]

    # 20-30 -> [20, 29]. [20, 29] vs [10, 19] -> No overlap (20 > 19)
    assert (20, 29) not in intervals["chr1"]

    # 60-70 -> [60, 69]. [60, 69] vs [70, 79] -> No overlap (69 < 70)
    assert (60, 69) not in intervals["chr1"]


def test_get_exclude_intervals_multiple_files(tmp_path):
    f1 = tmp_path / "ex1.bed"
    f1.write_text("chr1\t0\t10")
    f2 = tmp_path / "ex2.bed"
    f2.write_text("chr1\t20\t30")

    intervals = get_exclude_intervals({"ex1": f1, "ex2": f2})

    # 0-10 -> [0, 9]
    assert (0, 9) in intervals["chr1"]
    # 20-30 -> [20, 29]
    assert (20, 29) in intervals["chr1"]
    # 10-20 -> [10, 19]
    assert (10, 19) not in intervals["chr1"]


def test_get_exclude_intervals_empty(tmp_path):
    f = tmp_path / "empty.bed"
    f.write_text("")

    intervals = get_exclude_intervals({"ex": f})
    assert not intervals


def test_is_excluded():
    tree = InterLap()
    # Add [20, 30) -> [20, 29]
    tree.add((20, 29))
    # Add [50, 60) -> [50, 59]
    tree.add((50, 59))

    intervals = {"chr1": tree}

    # Test non-overlapping region
    # 0-10 -> is_excluded checks (0, 9)
    assert not is_excluded(intervals, "chr1", 0, 10)

    # Test overlapping region
    # 20-30 -> is_excluded checks (20, 29)
    assert is_excluded(intervals, "chr1", 20, 30)
    # 25-26 -> checks (25, 25)
    assert is_excluded(intervals, "chr1", 25, 26)

    # Test partial overlap
    # 15-25 -> checks (15, 24). Overlaps [20, 29].
    assert is_excluded(intervals, "chr1", 15, 25)

    # Test multi-region overlap
    # 20-60 -> checks (20, 59). Overlaps both.
    assert is_excluded(intervals, "chr1", 20, 60)

    # Test bounds
    # 90-100 -> checks (90, 99)
    assert not is_excluded(intervals, "chr1", 90, 100)

    # Test adjacent
    # 10-20 -> checks (10, 19). Ends at 19. Start at 20. No overlap.
    assert not is_excluded(intervals, "chr1", 10, 20)
    # 30-40 -> checks (30, 39). Start 30. End 29. No overlap.
    assert not is_excluded(intervals, "chr1", 30, 40)

    # Test unknown chrom
    assert not is_excluded(intervals, "chr2", 0, 10)

    # Test empty intervals
    assert not is_excluded({}, "chr1", 0, 10)
