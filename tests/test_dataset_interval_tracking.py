import pytest
import torch
from cerberus.transform import Jitter, ReverseComplement
from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from typing import cast
from cerberus.dataset import CerberusDataset

# Mocking helpers
class MockSequenceExtractor:
    def extract(self, interval):
        # Return dummy sequence tensor (4, len)
        length = interval.end - interval.start
        return torch.zeros(4, length)

@pytest.fixture
def mock_dataset(tmp_path):
    # Minimal config to instantiate Dataset
    # We will override components manually to test specific transforms
    
    genome_path = tmp_path / "genome.fa"
    genome_path.touch()
    
    genome_config = cast(GenomeConfig, {
        "name": "test",
        "fasta_path": genome_path,
        "species": "human",
        "allowed_chroms": ["chr1"],
        "chrom_sizes": {"chr1": 1000},
        "exclude_intervals": {},
        "fold_type": "chrom_partition",
        "fold_args": {"k": 2},
    })
    
    data_config = cast(DataConfig, {
        "inputs": {}, 
        "targets": {},
        "input_len": 100, 
        "output_len": 100,
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
        "padded_size": 100,
        "sampler_args": {"intervals_path": str(tmp_path / "dummy.bed")}
    })
    (tmp_path / "dummy.bed").write_text("chr1\t100\t200\n")
    
    ds = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=MockSequenceExtractor())
    return ds

def test_interval_string_jitter(mock_dataset):
    # Config: Jitter 50->10. Max Jitter 0 (Center crop).
    # Original Interval: chr1:100-200 (Length 100)
    # Jitter Input Len: 50.
    # Slack: 50. Center: 25.
    # Start offset: 25.
    # New Interval Start: 100 + 25 = 125.
    # New Interval End: 125 + 50 = 175.
    
    # Manually set transforms
    mock_dataset.transforms.transforms = [
        Jitter(input_len=50, max_jitter=0)
    ]
    
    # Original interval in sampler is chr1:100-200(+) (padded_size=100)
    # We assume sampler returns this.
    # Verify sampler content first
    assert len(mock_dataset) == 1
    # Sampler interval might depend on centering. 
    # Bed was 100-200. Padded 100. Center 150. Start 100, End 200.
    
    item = mock_dataset[0]
    interval_str = item["intervals"]
    
    # Expected: chr1:125-175(+)
    assert interval_str == "chr1:125-175(+)"

def test_interval_string_rc(mock_dataset):
    # Config: RC probability 1.0
    mock_dataset.transforms.transforms = [
        ReverseComplement(probability=1.0)
    ]
    
    # Original: chr1:100-200(+)
    
    item = mock_dataset[0]
    interval_str = item["intervals"]
    
    # Expected: chr1:100-200(-)
    assert interval_str == "chr1:100-200(-)"

def test_interval_string_jitter_and_rc(mock_dataset):
    # Chain: Jitter -> RC
    mock_dataset.transforms.transforms = [
        Jitter(input_len=50, max_jitter=0), # -> 125-175(+)
        ReverseComplement(probability=1.0)  # -> 125-175(-)
    ]
    
    item = mock_dataset[0]
    interval_str = item["intervals"]
    
    assert interval_str == "chr1:125-175(-)"
