import pytest
from cerberus.core import Interval

def test_interval_init():
    i = Interval("chr1", 100, 200)
    assert i.chrom == "chr1"
    assert i.start == 100
    assert i.end == 200
    assert i.strand == "+"

def test_interval_init_strand():
    i = Interval("chr1", 100, 200, "-")
    assert i.strand == "-"

def test_interval_len():
    i = Interval("chr1", 100, 200)
    assert len(i) == 100
    
    i = Interval("chr1", 100, 100)
    assert len(i) == 0

def test_interval_repr():
    i = Interval("chr1", 100, 200, "-")
    # Dataclass repr: Interval(chrom='chr1', start=100, end=200, strand='-')
    assert repr(i) == "Interval(chrom='chr1', start=100, end=200, strand='-')"

def test_interval_equality():
    i1 = Interval("chr1", 100, 200)
    i2 = Interval("chr1", 100, 200)
    i3 = Interval("chr1", 100, 201)
    
    assert i1 == i2
    assert i1 != i3
