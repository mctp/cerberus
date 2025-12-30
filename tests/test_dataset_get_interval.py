import pytest
import torch
from pathlib import Path
from typing import cast
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig, SamplerConfig
from cerberus.samplers import DummySampler

def test_get_interval_with_arbitrary_interval(tmp_path):
    """
    Test that get_interval works with an arbitrary interval not in the sampler.
    """
    # 1. Setup Files
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text(f"chr1\t1000\t6\t1000\t1001\n")
    
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\n")
    
    # 2. Configs
    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={}
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 10, 
        "output_len": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 10,
        "sampler_args": {"intervals_path": str(peaks)}
    })
    
    # 3. Instantiate Dataset
    ds = CerberusDataset(
        genome_config, 
        data_config, 
        sampler_config,
    )
    
    # 4. Call get_interval with an arbitrary interval
    interval = Interval("chr1", 50, 60)
    result = ds.get_interval(interval)
    
    assert "inputs" in result
    assert result["inputs"].shape == (4, 10)
    assert result["intervals"] == "chr1:50-60(+)"

def test_get_interval_equivalence_to_getitem(tmp_path):
    """
    Test that get_interval(sampler[i]) returns the same as __getitem__(i).
    """
    # Setup ...
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "C" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text(f"chr1\t1000\t6\t1000\t1001\n")
    
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t110\n")
    
    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={}
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 10, 
        "output_len": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 10,
        "sampler_args": {"intervals_path": str(peaks)}
    })
    
    ds = CerberusDataset(genome_config, data_config, sampler_config)
    
    idx = 0
    item_from_getitem = ds[idx]
    
    interval_obj = ds.sampler[idx]
    # Pass Interval object directly
    item_from_get_interval = ds.get_interval(interval_obj)
    
    assert torch.equal(item_from_getitem["inputs"], item_from_get_interval["inputs"])

def test_dummy_sampler_and_parsing(tmp_path):
    """
    Test dummy sampler usage and input parsing in get_interval.
    """
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "T" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text(f"chr1\t1000\t6\t1000\t1001\n")
    
    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={}
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 10, 
        "output_len": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    # Use Dummy Sampler
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "dummy",
        "padded_size": 10,
        "sampler_args": {}
    })
    
    ds = CerberusDataset(genome_config, data_config, sampler_config)
    
    assert isinstance(ds.sampler, DummySampler)
    assert len(ds) == 0
    
    # Test __getitem__ raises error
    with pytest.raises(NotImplementedError):
        ds[0]
        
    # Test string parsing
    res = ds.get_interval("chr1:100-110")
    assert res["inputs"].shape == (4, 10)
    assert res["intervals"] == "chr1:100-110(+)" # default strand +
    
    # Test tuple parsing
    res = ds.get_interval(("chr1", 200, 210))
    assert res["inputs"].shape == (4, 10)
    assert res["intervals"] == "chr1:200-210(+)"
    
    # Test Interval object
    res = ds.get_interval(Interval("chr1", 300, 310))
    assert res["inputs"].shape == (4, 10)
