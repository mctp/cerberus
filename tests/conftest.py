import os
import shutil
from pathlib import Path

import pytest

from cerberus.download import download_dataset, download_reference_genome


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("CERBERUS_PRUNE_DOWNLOADS"):
        base_dir = get_base_dir()
        print(f"\nPruning downloads from {base_dir}...")
        
        for subdir in ["genome", "dataset"]:
            d = base_dir / subdir
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d)

def get_base_dir():
    return Path(os.environ.get("CERBERUS_DATA_DIR", "tests/data"))

@pytest.fixture(scope="session")
def human_genome():
    """
    Downloads the hg38 genome if not present.
    Returns the dictionary of file paths.
    """
    if os.environ.get("RUN_SLOW_TESTS") is None:
        pytest.skip("Skipping slow tests (RUN_SLOW_TESTS not set)")

    base_dir = get_base_dir()
    genome_dir = base_dir / "genome"
    return download_reference_genome(genome_dir, genome="hg38")

@pytest.fixture(scope="session")
def mouse_genome():
    """
    Downloads the mm10 genome if not present.
    Returns the dictionary of file paths.
    """
    if os.environ.get("RUN_SLOW_TESTS") is None:
        pytest.skip("Skipping slow tests (RUN_SLOW_TESTS not set)")

    base_dir = get_base_dir()
    genome_dir = base_dir / "genome"
    return download_reference_genome(genome_dir, genome="mm10")

@pytest.fixture(scope="session")
def mdapca2b_ar_dataset():
    """
    Downloads the mdapca2b-ar dataset if not present.
    Returns a dictionary with paths to the files.
    """
    if os.environ.get("RUN_SLOW_TESTS") is None:
        pytest.skip("Skipping slow tests (RUN_SLOW_TESTS not set)")

    base_dir = get_base_dir()
    data_dir = base_dir / "dataset"
    return download_dataset(data_dir, name="mdapca2b_ar")

@pytest.fixture
def mock_files(tmp_path):
    """Creates dummy files required for config validation."""
    fasta = tmp_path / "genome.fa"
    fasta.touch()
    
    exclude = tmp_path / "exclude.bed"
    exclude.touch()
    
    input_bw = tmp_path / "input.bw"
    input_bw.touch()
    
    target_bw = tmp_path / "target.bw"
    target_bw.touch()
    
    return {
        "fasta": fasta,
        "exclude": exclude,
        "input": input_bw,
        "target": target_bw
    }
