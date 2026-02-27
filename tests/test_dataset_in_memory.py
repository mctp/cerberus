from cerberus.config import DataConfig, SamplerConfig
from cerberus.genome import create_genome_config
from cerberus.sequence import InMemorySequenceExtractor, SequenceExtractor
from typing import cast
from cerberus.dataset import CerberusDataset

def test_dataset_in_memory_extractor(tmp_path):
    # Setup
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "N" * 100 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text(f"chr1\t100\t6\t100\t101\n")
    
    # Dummy bed
    (tmp_path / "dummy.bed").write_text("chr1\t10\t20\n")
    
    base_data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 10,
        "output_len": 10,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 10,
        "sampler_args": {"intervals_path": str(tmp_path / "dummy.bed")}
    })

    # Test 1: in_memory=True
    genome_config = create_genome_config(
        name="test",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"]
    )
    
    data_config = cast(DataConfig, base_data_config.copy())
    
    ds_mem = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None, in_memory=True)
    assert isinstance(ds_mem.sequence_extractor, InMemorySequenceExtractor)
    
    seq = ds_mem[0]["inputs"]
    assert seq.shape == (4, 10)

    # Test 2: in_memory=False
    ds_disk = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None, in_memory=False)
    assert isinstance(ds_disk.sequence_extractor, SequenceExtractor)
    
    seq = ds_disk[0]["inputs"]
    assert seq.shape == (4, 10)
