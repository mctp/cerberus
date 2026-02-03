import pytest
import sys
from pathlib import Path
import random
from cerberus.sequence import compute_intervals_gc
from cerberus.interval import Interval
from cerberus.samplers import ComplexityMatchedSampler  # Ensure this imports without error

def create_dummy_fasta(path: Path, chroms: dict[str, str]):
    """Creates a dummy FASTA file."""
    with open(path, "w") as f:
        for name, seq in chroms.items():
            f.write(f">{name}\n")
            # Write in chunks of 80 chars
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
    return path

def test_compute_intervals_gc(tmp_path):
    # 1. Setup Dummy FASTA
    fasta_path = tmp_path / "test.fa"
    chrom1_seq = "ACGT" * 25  # 100 bp, 50% GC
    chrom2_seq = "AAAA" * 25  # 100 bp, 0% GC
    chrom3_seq = "GGGG" * 25  # 100 bp, 100% GC
    
    # Add some mixed content
    # GC content of "ACGG" is 0.75
    chrom4_seq = "ACGG" * 25 
    
    chroms = {
        "chr1": chrom1_seq,
        "chr2": chrom2_seq,
        "chr3": chrom3_seq,
        "chr4": chrom4_seq
    }
    create_dummy_fasta(fasta_path, chroms)
    
    # 2. Define Intervals
    intervals = [
        Interval("chr1", 0, 100, "+"),  # 0.5
        Interval("chr2", 0, 100, "+"),  # 0.0
        Interval("chr3", 0, 100, "+"),  # 1.0
        Interval("chr4", 0, 100, "+"),  # 0.75
        Interval("chr1", 0, 4, "+"),    # ACGT -> 0.5
        Interval("chr1", 0, 2, "+"),    # AC -> 0.5
    ]
    
    # 3. Compute
    gc_values = compute_intervals_gc(intervals, fasta_path)
    
    # 4. Verify
    expected = [0.5, 0.0, 1.0, 0.75, 0.5, 0.5]
    
    assert len(gc_values) == len(expected)
    for v, e in zip(gc_values, expected):
        assert abs(v - e) < 1e-6

def test_compute_intervals_gc_invalid_chrom(tmp_path):
    fasta_path = tmp_path / "test_invalid.fa"
    create_dummy_fasta(fasta_path, {"chr1": "ACGT"})
    
    intervals = [
        Interval("chr_missing", 0, 4, "+")
    ]
    
    # Should catch exception and return 0.0
    gc_values = compute_intervals_gc(intervals, fasta_path)
    assert gc_values == [0.0]

def test_samplers_no_pyfaidx_dependency():
    """
    Check if samplers module can be imported without pyfaidx.
    Note: pyfaidx is already imported by sequence.py, so it will be in sys.modules.
    But we can inspect the source code of samplers.py to ensure 'import pyfaidx' is absent.
    """
    with open("src/cerberus/samplers.py", "r") as f:
        content = f.read()
    
    assert "import pyfaidx" not in content, "samplers.py should not import pyfaidx directly"
    assert "from pyfaidx" not in content
