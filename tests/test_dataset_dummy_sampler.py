import pytest
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig
from typing import cast
from cerberus.samplers import DummySampler

def test_dataset_implicit_dummy_sampler(tmp_path):
    # Setup
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "N" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text(f"chr1\t1000\t6\t1000\t1001\n")
    
    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 200, 
        "output_len": 200,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "use_sequence": True,
    })
    
    # Init without sampler config
    ds = CerberusDataset(genome_config, data_config, sampler_config=None, sampler=None)
    
    # Assertions
    assert isinstance(ds.sampler, DummySampler)
    assert len(ds) == 0
    
    # Verify get_interval works
    # chr1:100-300 (len 200)
    out = ds.get_interval("chr1:100-300")
    assert out["inputs"].shape == (4, 200)
