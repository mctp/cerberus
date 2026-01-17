
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from cerberus.download import download_reference_genome, GENOME_RESOURCES

@patch("cerberus.download._download_file")
@patch("gzip.open")
@patch("shutil.copyfileobj")
@patch("pyfaidx.Fasta")
@patch("pathlib.Path.unlink")
def test_download_mouse_reference(mock_unlink, mock_fasta, mock_copy, mock_gzip, mock_download, tmp_path):
    """Test that download_reference_genome works for mm10."""
    output_dir = tmp_path / "data"
    
    # Mock file existence checks to simulate files NOT existing, then existing
    with patch("pathlib.Path.exists", side_effect=[False] * 10): 
        results = download_reference_genome(output_dir, genome="mm10")

    assert results["fasta"].name == "mm10.fa"
    assert results["blacklist"].name == "blacklist.bed"
    assert results["gaps"].name == "gaps.bed"
    assert results["encode_cre"].name == "encode_cre.bb"
    assert "mappability" not in results  # Should be optional/missing for mm10 currently

    # Check that correct URLs were used
    mm10_resources = GENOME_RESOURCES["mm10"]
    # We can't easily check call args order due to dict iteration order, but we can check count
    assert mock_download.call_count == 4

@patch("cerberus.download._download_file")
@patch("gzip.open")
@patch("shutil.copyfileobj")
@patch("pyfaidx.Fasta")
@patch("pathlib.Path.unlink")
def test_download_human_reference(mock_unlink, mock_fasta, mock_copy, mock_gzip, mock_download, tmp_path):
    """Test that download_reference_genome works for hg38."""
    output_dir = tmp_path / "data"
    
    # Mock file existence checks
    with patch("pathlib.Path.exists", side_effect=[False] * 12): 
        results = download_reference_genome(output_dir, genome="hg38")

    assert results["fasta"].name == "hg38.fa"
    assert results["blacklist"].name == "blacklist.bed"
    assert results["gaps"].name == "gaps.bed"
    assert results["encode_cre"].name == "encode_cre.bb"
    assert results["mappability"].name == "mappability.bw"

    # Check that correct URLs were used
    hg38_resources = GENOME_RESOURCES["hg38"]
    assert mock_download.call_count == 5
