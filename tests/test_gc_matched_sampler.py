import pytest
from pathlib import Path
import random
from cerberus.samplers import create_sampler, RandomSampler, ComplexityMatchedSampler
from cerberus.sequence import calculate_gc_content

@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    # Create a small genome with varying GC content
    # chr1: high GC
    # chr2: low GC
    
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("GC" * 500 + "\n") # 1000 bp, 100% GC
        f.write(">chr2\n")
        f.write("AT" * 500 + "\n") # 1000 bp, 0% GC
        f.write(">chr3\n")
        f.write("ACGT" * 250 + "\n") # 1000 bp, 50% GC
        
    # Index it
    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))
    
    return fasta_path

def test_calculate_gc_content():
    assert calculate_gc_content("GCGC") == 1.0
    assert calculate_gc_content("ATAT") == 0.0
    assert calculate_gc_content("ACGT") == 0.5
    assert calculate_gc_content("NNNN") == 0.0
    assert calculate_gc_content("GCN") == 1.0 # 2 / 2

def test_random_sampler():
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    padded_size = 100
    num_intervals = 50
    exclude_intervals = {}
    folds = []
    
    sampler = RandomSampler(chrom_sizes, padded_size, num_intervals, folds, exclude_intervals, seed=42)
    
    assert len(sampler) == num_intervals
    for interval in sampler:
        assert interval.chrom in chrom_sizes
        assert interval.end - interval.start == padded_size
        assert interval.start >= 0
        assert interval.end <= chrom_sizes[interval.chrom]

def test_complexity_matched_sampler_gc_only(mock_fasta):
    # Setup:
    # Target: 10 intervals from chr1 (100% GC)
    # Candidate: 100 random intervals from chr1, chr2, chr3
    
    # We want to select candidates that are also high GC (chr1)
    
    chrom_sizes = {"chr1": 1000, "chr2": 1000, "chr3": 1000}
    padded_size = 10
    
    # Create target intervals (manually pointing to chr1)
    target_bed = mock_fasta.parent / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(10):
            # 100% GC
            f.write(f"chr1\t{i*10}\t{i*10+10}\n")
            
    config = {
        "sampler_type": "complexity_matched",
        "padded_size": 10,
        "sampler_args": {
            "target_sampler": {
                "type": "interval",
                "args": {"intervals_path": str(target_bed)}
            },
            "candidate_sampler": {
                "type": "random",
                "args": {"num_intervals": 300} # Enough to likely hit all chroms
            },
            "bins": 10,
            "candidate_ratio": 1.0,
            "metrics": ["gc"]
        }
    }

    sampler = create_sampler(
        config, chrom_sizes, [], {}, fasta_path=mock_fasta
    )
    
    assert isinstance(sampler, ComplexityMatchedSampler)
    
    # Initial length should match target * ratio
    assert len(sampler) == 10
    
    # Check GC content of selected samples
    # Target is 100% GC.
    # Candidates from chr1 are 100% GC.
    # Candidates from chr2 are 0% GC.
    # Candidates from chr3 are 50% GC.
    
    # ComplexityMatchedSampler with gc metric should pick mostly chr1 candidates.
    
    count_chr1 = 0
    count_chr2 = 0
    count_chr3 = 0
    
    for interval in sampler:
        if interval.chrom == "chr1": count_chr1 += 1
        elif interval.chrom == "chr2": count_chr2 += 1
        elif interval.chrom == "chr3": count_chr3 += 1
        
    print(f"Counts: chr1={count_chr1}, chr2={count_chr2}, chr3={count_chr3}")
    
    # Ideally all chr1
    assert count_chr1 == 10
    assert count_chr2 == 0
    assert count_chr3 == 0
