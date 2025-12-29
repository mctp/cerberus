import pytest
import torch
from unittest.mock import MagicMock, patch
from cerberus.sequence import SequenceExtractor, encode_dna
from cerberus.interval import Interval

def test_encode_dna_invalid_encoding():
    with pytest.raises(ValueError, match="Unsupported encoding"):
        encode_dna("ACGT", encoding="INVALID")

@patch("cerberus.sequence.pyfaidx")
def test_sequence_extractor_chrom_not_found(mock_pyfaidx):
    mock_fasta = MagicMock()
    mock_pyfaidx.Fasta.return_value = mock_fasta
    
    # Simulate chrom not in fasta
    mock_fasta.__contains__.return_value = False
    
    extractor = SequenceExtractor("test.fa")
    interval = Interval("chrMissing", 0, 10)
    
    with pytest.raises(ValueError, match="Chromosome chrMissing not found"):
        extractor.extract(interval)

@patch("cerberus.sequence.pyfaidx")
def test_sequence_extractor_lazy_load(mock_pyfaidx):
    extractor = SequenceExtractor("test.fa")
    assert extractor.fasta is None
    
    # extract triggers load
    mock_fasta = MagicMock()
    mock_pyfaidx.Fasta.return_value = mock_fasta
    mock_fasta.__contains__.return_value = True
    
    mock_seq = MagicMock()
    mock_seq.seq = "ACGT"
    # fasta[chrom][start:end] -> sequence
    mock_fasta.__getitem__.return_value.__getitem__.return_value = mock_seq
    
    extractor.extract(Interval("chr1", 0, 4))
    
    assert extractor.fasta is not None
    mock_pyfaidx.Fasta.assert_called_once()
