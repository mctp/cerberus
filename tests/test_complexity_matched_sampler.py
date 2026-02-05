import pytest
from pathlib import Path
import random
import numpy as np
from cerberus.samplers import create_sampler, ComplexityMatchedSampler, ListSampler, Interval
from interlap import InterLap

@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    
    # 1. High GC, High Complexity, High CpG
    # Random sequence with 80% G/C
    rng = np.random.default_rng(42)
    bases = np.array(['A', 'C', 'G', 'T'])
    
    def generate_seq(length, p):
        return "".join(rng.choice(bases, size=length, p=p))

    # chr1: High GC (80%), High complexity
    seq1 = generate_seq(2000, [0.1, 0.4, 0.4, 0.1])
    
    # chr2: Low GC (20%), High complexity
    seq2 = generate_seq(2000, [0.4, 0.1, 0.1, 0.4])
    
    # chr3: Repetitive (Low complexity) - "ACACAC..."
    seq3 = "AC" * 1000
    
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq1 + "\n")
        f.write(">chr2\n")
        f.write(seq2 + "\n")
        f.write(">chr3\n")
        f.write(seq3 + "\n")
        
    # Index it
    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))
    
    return fasta_path

def test_complexity_matched_sampler(mock_fasta):
    chrom_sizes = {"chr1": 2000, "chr2": 2000, "chr3": 2000}
    
    # Target: 20 intervals from chr1 (High GC, High Complexity)
    target_bed = mock_fasta.parent / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(20):
            # Intervals of size 50
            start = i * 50
            f.write(f"chr1\t{start}\t{start+50}\n")
            
    config = {
        "sampler_type": "complexity_matched",
        "padded_size": 50,
        "sampler_args": {
            "target_sampler": {
                "type": "interval",
                "args": {"intervals_path": str(target_bed)}
            },
            "candidate_sampler": {
                "type": "random",
                "args": {"num_intervals": 1000} 
            },
            "bins": 5,
            "candidate_ratio": 1.0,
            "metrics": ["gc", "dust", "cpg"]
        }
    }
    
    sampler = create_sampler(
        config, chrom_sizes, [], {}, fasta_path=mock_fasta, seed=42
    )
    
    assert isinstance(sampler, ComplexityMatchedSampler)
    assert len(sampler) == 20
    
    # Check if selected intervals are mostly from chr1
    # chr1 matches targets best (same chrom)
    # chr2 has different GC
    # chr3 has different Dust score (low complexity)
    
    chr1_count = sum(1 for iv in sampler if iv.chrom == "chr1")
    chr2_count = sum(1 for iv in sampler if iv.chrom == "chr2")
    chr3_count = sum(1 for iv in sampler if iv.chrom == "chr3")
    
    print(f"Counts: chr1={chr1_count}, chr2={chr2_count}, chr3={chr3_count}")
    
    # We expect overwhelming preference for chr1
    assert chr1_count > 15
    assert chr2_count < 5
    assert chr3_count < 5

def test_complexity_matched_exclusions(mock_fasta):
    """
    Ensure ComplexityMatchedSampler explicitly filters out excluded regions,
    even if the candidate sampler provides them.
    """
    chrom_sizes = {"chr1": 2000}
    
    # Target: 1 interval
    target_intervals = [Interval("chr1", 0, 100, "+")]
    target_sampler = ListSampler(intervals=target_intervals, chrom_sizes=chrom_sizes)
    
    # Candidate: 2 intervals, one valid, one excluded
    # Both are on chr1 which matches target (and is homogeneous in mock_fasta)
    candidate_intervals = [
        Interval("chr1", 100, 200, "+"),  # Valid
        Interval("chr1", 200, 300, "+")   # Excluded
    ]
    candidate_sampler = ListSampler(intervals=candidate_intervals, chrom_sizes=chrom_sizes)
    
    # Define exclusions
    exclude_intervals = {"chr1": InterLap()}
    exclude_intervals["chr1"].add((200, 300))
    
    # Create sampler with exclusions
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        exclude_intervals=exclude_intervals,
        candidate_ratio=10.0, # Ask for many to force picking if available
        bins=1,
        metrics=["gc"],
        seed=42
    )
    
    # It should only pick the valid one
    for interval in sampler:
        assert interval.start != 200
        assert interval.start == 100
