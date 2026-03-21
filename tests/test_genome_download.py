import os
import pytest
from pathlib import Path
from cerberus.download import download_human_reference
from cerberus.genome import create_human_genome_config

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow download tests")
def test_download_human_reference():
    base_dir = Path(os.environ.get("CERBERUS_DATA_DIR", "tests/data"))
    output_base = base_dir / "genome"
    genome_name = "hg38"
    genome_dir = output_base / genome_name
    
    # Call the function
    # This should create {tmp_path}/genome/hg38 and download files there
    results = download_human_reference(output_base, name=genome_name)
    
    # Verify outputs
    assert genome_dir.exists()
    assert results["fasta"].exists()
    assert results["fai"].exists()
    assert results["blacklist"].exists()
    assert results["gaps"].exists()
    
    # Verify file sizes are not zero (basic check)
    assert results["fasta"].stat().st_size > 0
    assert results["fai"].stat().st_size > 0
    assert results["blacklist"].stat().st_size > 0
    assert results["gaps"].stat().st_size > 0
    assert results["encode_cre"].exists()
    assert results["encode_cre"].stat().st_size > 0

    # Verify filtering (length > 3)
    with open(results["gaps"], "r") as f:
        for line in f:
            parts = line.split()
            start, end = int(parts[1]), int(parts[2])
            length = end - start
            assert length > 3, f"Found gap of length {length} at {parts[0]}:{start}-{end}"
    
    print(f"\nDownload completed. Files saved to {genome_dir}")

    # Test config creation
    genome_config = create_human_genome_config(genome_dir)
    assert genome_config.name == "hg38"
    assert genome_config.fasta_path == results["fasta"]
    assert "blacklist" in genome_config.exclude_intervals
    assert "unmappable" in genome_config.exclude_intervals
    assert genome_config.exclude_intervals["blacklist"] == results["blacklist"]
    assert genome_config.exclude_intervals["unmappable"] == results["gaps"]
