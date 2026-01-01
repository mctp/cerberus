import pytest
from interlap import InterLap
from cerberus.interval import Interval
from cerberus.samplers import IntervalSampler, SlidingWindowSampler
from cerberus.genome import create_genome_folds

def test_load_bed3(tmp_path):
    f = tmp_path / "test.bed"
    f.write_text("chr1\t100\t200\nchr2\t300\t400")
    
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    assert len(intervals) == 2
    assert intervals[0] == Interval("chr1", 100, 200, "+")
    assert intervals[1] == Interval("chr2", 300, 400, "+")
    
    # Test __getitem__
    assert sampler[0] == intervals[0]
    assert sampler[1] == intervals[1]

def test_load_bed6_strand(tmp_path):
    f = tmp_path / "test.bed"
    f.write_text("chr1\t100\t200\tname\t0\t-\nchr2\t300\t400\tname\t0\t+")
    
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    assert len(intervals) == 2
    assert intervals[0].strand == "-"
    assert intervals[1].strand == "+"

def test_load_bed_padded_size(tmp_path):
    f = tmp_path / "test.bed"
    f.write_text("chr1\t100\t200") # Center = 150
    
    # Target length 50 -> 125 to 175
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=50, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    assert len(intervals) == 1
    assert intervals[0].start == 125
    assert intervals[0].end == 175
    assert len(intervals[0]) == 50

def test_load_narrowPeak_summit(tmp_path):
    f = tmp_path / "test.narrowPeak"
    # col 10 is summit offset. 
    # chrom start end name score strand signal pval qval peak
    # chr1 100 200 . 0 . 0 0 0 10 (summit at 100+10=110)
    line = "chr1\t100\t200\t.\t0\t+\t0\t0\t0\t10" 
    f.write_text(line)
    
    # Target length 20 -> center at 110 -> 100 to 120
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=20, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    assert len(intervals) == 1
    # Center = 110
    # Start = 110 - 10 = 100
    # End = 100 + 20 = 120
    assert intervals[0].start == 100
    assert intervals[0].end == 120

def test_load_narrowPeak_midpoint_fallback(tmp_path):
    # Test that if we pass padded_size but invalid summit (e.g. -1), it uses midpoint
    f = tmp_path / "test.narrowPeak"
    # chr1 100 200 ... summit=-1
    line = "chr1\t100\t200\t.\t0\t+\t0\t0\t0\t-1" 
    f.write_text(line)
    
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=20, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    assert len(intervals) == 1
    # Center = (100+200)//2 = 150
    # Start = 150 - 10 = 140
    # End = 140 + 20 = 160
    assert intervals[0].start == 140
    assert intervals[0].end == 160

def test_load_bed_invalid_columns(tmp_path):
    f = tmp_path / "test.bed"
    f.write_text("chr1\t100") # Only 2 cols
    
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    with pytest.raises(ValueError, match="Invalid BED line"):
        IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=1, exclude_intervals={}, folds=folds)

def test_load_narrowPeak_invalid_columns(tmp_path):
    f = tmp_path / "test.narrowPeak"
    f.write_text("chr1\t100\t200\t.\t0\t+\t0\t0\t0") # 9 cols
    
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    with pytest.raises(ValueError, match="Invalid narrowPeak line"):
        IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=1, exclude_intervals={}, folds=folds)

def test_load_narrowPeak_invalid_summit(tmp_path):
    f = tmp_path / "test.narrowPeak"
    f.write_text("chr1\t100\t200\t.\t0\t+\t0\t0\t0\tinvalid") # 10 cols, bad summit
    
    chrom_sizes = {"chr1": 1000}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    with pytest.raises(ValueError): # int conversion fail
        IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=1, exclude_intervals={}, folds=folds)

def test_load_bed_with_chrom_sizes(tmp_path):
    f = tmp_path / "test.bed"
    # chr1 size 1000. chr2 size 500.
    # chr1:100-200 -> Valid
    # chr1:950-1050 -> Invalid (end > 1000)
    # chr2:100-200 -> Valid
    # chr3:100-200 -> Invalid (chrom not in sizes)
    f.write_text("chr1\t100\t200\nchr1\t950\t1050\nchr2\t100\t200\nchr3\t100\t200")
    
    chrom_sizes = {"chr1": 1000, "chr2": 500}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, padded_size=100, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 2
    assert intervals[0].chrom == "chr1"
    assert intervals[0].start == 100
    assert intervals[1].chrom == "chr2"
    assert intervals[1].start == 100

def test_load_narrowPeak_with_chrom_sizes_and_padded_size(tmp_path):
    f = tmp_path / "test.narrowPeak"
    # chr1 size 100. padded_size 20.
    # peak at 90. center=90+0=90. start=80, end=100. Valid.
    # peak at 95. center=95. start=85, end=105. Invalid (end > 100).
    line1 = "chr1\t90\t91\t.\t0\t+\t0\t0\t0\t0"
    line2 = "chr1\t95\t96\t.\t0\t+\t0\t0\t0\t0"
    f.write_text(f"{line1}\n{line2}")
    
    chrom_sizes = {"chr1": 100}
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, padded_size=20, chrom_sizes=chrom_sizes, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 1
    assert intervals[0].start == 80
    assert intervals[0].end == 100

def test_load_bed_with_excludes(tmp_path):
    f = tmp_path / "test.bed"
    # chr1: 0-100 (overlaps exclude 40-60)
    # chr1: 200-300 (no overlap)
    # chr2: 0-100 (overlaps exclude 0-20)
    # chr2: 200-300 (no overlap)
    f.write_text("chr1\t0\t100\nchr1\t200\t300\nchr2\t0\t100\nchr2\t200\t300")
    
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    
    exclude_intervals = {}
    
    # Exclude 1: chr1:40-60
    tree1 = InterLap()
    tree1.add((40, 60))
    exclude_intervals["chr1"] = tree1
    
    # Exclude 2: chr2:0-20
    tree2 = InterLap()
    tree2.add((0, 20))
    exclude_intervals["chr2"] = tree2
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = IntervalSampler(f, chrom_sizes=chrom_sizes, exclude_intervals=exclude_intervals, padded_size=100, folds=folds)
    intervals = list(sampler)
    
    # Should only have chr1:200-300 and chr2:200-300
    assert len(intervals) == 2
    assert intervals[0].chrom == "chr1"
    assert intervals[0].start == 200
    assert intervals[1].chrom == "chr2"
    assert intervals[1].start == 200


def test_sliding_window_basic():
    chrom_sizes = {"chr1": 100}
    # Window 20, Stride 20 (Non-overlapping)
    # 0-20, 20-40, 40-60, 60-80, 80-100
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = SlidingWindowSampler(chrom_sizes, padded_size=20, stride=20, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 5
    assert intervals[0] == Interval("chr1", 0, 20, "+")
    assert intervals[4] == Interval("chr1", 80, 100, "+")

def test_sliding_window_overlap():
    chrom_sizes = {"chr1": 30}
    # Window 20, Stride 10
    # 0-20, 10-30
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = SlidingWindowSampler(chrom_sizes, padded_size=20, stride=10, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 2
    assert intervals[0] == Interval("chr1", 0, 20, "+")
    assert intervals[1] == Interval("chr1", 10, 30, "+")

def test_sliding_window_exclude():
    chrom_sizes = {"chr1": 100}
    # Window 20, Stride 20
    # 0-20, 20-40, 40-60, 60-80, 80-100
    
    # Exclude: 30-50
    # 0-20: ok
    # 20-40: Overlaps 30-40 -> Exclude
    # 40-60: Overlaps 40-50 -> Exclude
    # 60-80: ok
    # 80-100: ok
    
    exclude_intervals = {}
    tree = InterLap()
    tree.add((30, 50))
    exclude_intervals["chr1"] = tree
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = SlidingWindowSampler(chrom_sizes, padded_size=20, stride=20, exclude_intervals=exclude_intervals, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 3
    assert intervals[0].start == 0
    assert intervals[1].start == 60
    assert intervals[2].start == 80

def test_sliding_window_multiple_chroms():
    chrom_sizes = {"chr2": 40, "chr1": 40}
    # Window 20, Stride 20
    # chr1: 0-20, 20-40
    # chr2: 0-20, 20-40
    # Should sort chroms
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 5})
    sampler = SlidingWindowSampler(chrom_sizes, padded_size=20, stride=20, exclude_intervals={}, folds=folds)
    intervals = list(sampler)
    
    assert len(intervals) == 4
    assert intervals[0].chrom == "chr1"
    assert intervals[1].chrom == "chr1"
    assert intervals[2].chrom == "chr2"
    assert intervals[3].chrom == "chr2"

def test_create_folds():
    chrom_sizes = {"chr1": 100, "chr2": 90, "chr3": 50, "chr4": 40}
    
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    assert len(folds) == 2
    
    # Expected: Fold 0 [chr1, chr4], Fold 1 [chr2, chr3]
    # Sum: 140 vs 140
    
    fold0 = set(folds[0])
    fold1 = set(folds[1])
    
    assert fold0 == {"chr1", "chr4"}
    assert fold1 == {"chr2", "chr3"}
