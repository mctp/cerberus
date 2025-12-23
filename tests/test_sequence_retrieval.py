import os
import pytest
import torch
import numpy as np
import pysam
from pathlib import Path
from cerberus.sequence import SequenceExtractor, encode_dna
from cerberus.core import Interval

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_sequence_extraction_correctness_vs_pysam(human_genome):
    """
    Verify correctness of sequence extraction from real genome file
    by comparing with pysam.
    """
    fasta_path = human_genome["fasta"]
    # Setup
    extractor = SequenceExtractor(fasta_path=fasta_path)
    
    # Use pysam for reference
    pysam_fasta = pysam.FastaFile(str(fasta_path))
    
    # Get chroms and lengths
    # pysam.FastaFile attributes: references (list of names), lengths (list of lengths)
    chroms = pysam_fasta.references
    lengths = pysam_fasta.lengths
    chrom_sizes = dict(zip(chroms, lengths))
    
    # Generate intervals
    num_intervals = 20
    length = 100
    intervals = []
    
    np.random.seed(42)
    
    for _ in range(num_intervals):
        chrom = np.random.choice(chroms)
        size = chrom_sizes[chrom]
        
        if size < length:
            continue
            
        start = np.random.randint(0, size - length)
        end = start + length
        intervals.append(Interval(chrom, start, end, "+"))
        
    assert len(intervals) > 0, "No valid intervals generated"
    
    for interval in intervals:
        # A. Cerberus extraction
        seq_tensor = extractor.extract(interval) # (4, L)
        
        # B. Reference extraction via pysam
        # pysam fetch is 0-based half-open [start, end)
        seq_str = pysam_fasta.fetch(interval.chrom, interval.start, interval.end)
        
        # Encode
        expected_tensor = encode_dna(seq_str)
        
        # C. Compare
        assert torch.equal(seq_tensor, expected_tensor), \
            f"Mismatch at {interval}. \nRef Seq: {seq_str}"
            
    pysam_fasta.close()
