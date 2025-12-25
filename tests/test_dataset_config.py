import pytest
from pathlib import Path
from typing import cast
from cerberus.dataset import CerberusDataset
from cerberus.config import (
    validate_data_config,
    validate_genome_config,
    GenomeConfig,
    DataConfig,
    SamplerConfig,
)
from cerberus.genome import create_genome_config

def test_validate_genome_config_valid(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.touch()
    (tmp_path / "genome.fa.fai").touch()
    
    # Test valid full config
    config = create_genome_config(name="genome", fasta_path=genome, species="human")
    validated = validate_genome_config(config)
    assert validated['fasta_path'] == genome
    assert validated['name'] == "genome"

    config = create_genome_config(name="hg38", fasta_path=genome, species="human")
    validated = validate_genome_config(config)
    assert validated['fasta_path'] == genome
    assert validated['name'] == "hg38"

def test_validate_genome_config_invalid_input(tmp_path):
    # Missing keys
    config = cast(GenomeConfig, {"path": "test"})
    with pytest.raises(ValueError, match="missing required keys"):
        validate_genome_config(config)

def test_validate_genome_config_allowed_chroms_invalid(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.touch()
    (tmp_path / "genome.fa.fai").touch()
    
    config = create_genome_config(name="test", fasta_path=genome, species="human")
    config['allowed_chroms'] = "chr1" # type: ignore
    
    with pytest.raises(TypeError, match="allowed_chroms must be a list"):
        validate_genome_config(config)

def test_validate_genome_config_missing_file(tmp_path):
    # create_genome_config raises FileNotFoundError, so we test that
    with pytest.raises(FileNotFoundError):
        create_genome_config(name="test", fasta_path=tmp_path / "missing.fa", species="human")

def test_validate_data_config_valid(tmp_path):
    # Create dummy files
    cons = tmp_path / "cons.bw"
    cons.touch()
    counts = tmp_path / "counts.bw"
    counts.touch()
    
    config = cast(DataConfig, {
        "inputs": {"cons": str(cons)},
        "targets": {"counts": str(counts)},
        "input_len": 2048,
        "output_len": 1024,
        "bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False
    })
    validated = validate_data_config(config)
    
    # Check that values are converted to Path objects
    assert isinstance(validated['inputs']['cons'], Path)
    assert validated['inputs']['cons'] == cons
    assert validated['output_len'] == 1024
    assert validated['input_len'] == 2048
    # Check default in_memory
    assert validated['in_memory'] is False

def test_validate_data_config_missing_key():
    config = cast(DataConfig, {
        "inputs": {},
        # Missing targets and others
        "output_len": 1024
    })
    with pytest.raises(ValueError, match="Data config missing required keys"):
        validate_data_config(config)

def test_dataset_init(tmp_path):
    genome_path = tmp_path / "genome.fa"
    genome_path.touch()
    
    # Create .fai file with extra chrom
    fai_path = tmp_path / "genome.fa.fai"
    fai_path.write_text("chr1\t1000\t0\t80\t81\nchr2\t2000\t0\t80\t81\n")
    
    # Create dummy peaks file
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\n")

    # Restrict to chr1
    genome_config = create_genome_config(
        name="test_genome", 
        fasta_path=genome_path,
        species="human",
        allowed_chroms=["chr1"]
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False
    })
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 100,
        "exclude_intervals": {},
        "sampler_args": {"intervals_path": str(peaks)}
    })
    ds = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None)
    
    # Check GenomeConfig object
    assert ds.genome_config['name'] == "test_genome"
    assert ds.genome_config['fasta_path'] == genome_path
    assert ds.genome_config['chrom_sizes'] is not None
    assert "chr1" in ds.genome_config['chrom_sizes']
    assert "chr2" not in ds.genome_config['chrom_sizes']
    
    # Check computed config
    assert "chr1" in ds.genome_config['chrom_sizes']

def test_validate_data_config_invalid_types(tmp_path):
    config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 2048,
        "output_len": "1024", # Invalid type
        "bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False
    })
    with pytest.raises(ValueError, match="output_len must be a positive integer"):
        validate_data_config(config)

def test_validate_data_config_missing_file(tmp_path):
    config = cast(DataConfig, {
        "inputs": {"cons": str(tmp_path / "missing.bw")},
        "targets": {},
        "input_len": 2048,
        "output_len": 1024,
        "bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False
    })
    with pytest.raises(FileNotFoundError, match="inputs file 'cons' not found"):
        validate_data_config(config)

def test_validate_genome_config_invalid_chrom_sizes(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.touch()
    
    config = cast(GenomeConfig, {
        "name": "test", 
        "fasta_path": genome, 
        "allowed_chroms": ["chr1"],
        "exclude_intervals": {},
        "fold_type": "chrom_partition",
        "fold_args": {"k": 5},
        "chrom_sizes": {"chr1": "100"}, # Invalid value type
    })
    
    with pytest.raises(TypeError, match="chrom_sizes values must be integers"):
        validate_genome_config(config)
