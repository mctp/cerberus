
import os
import pytest
import torch
from cerberus.config import DataConfig, SamplerConfig
from cerberus.signal import SignalExtractor
from cerberus.sequence import SequenceExtractor
from cerberus.interval import Interval
from cerberus.transform import Bin, Jitter
from cerberus.dataset import CerberusDataset
from cerberus.samplers import SubsetSampler
from cerberus.genome import create_human_genome_config

@pytest.fixture
def mappability_file(human_genome):
    return human_genome["mappability"]

@pytest.fixture
def fasta_file(human_genome):
    return human_genome["fasta"]

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_sequence_and_signal_extraction(human_genome, mappability_file, fasta_file):
    """
    Test combined extraction of sequence and mappability signal.
    """
    # 1. Manual Extraction
    seq_extractor = SequenceExtractor(fasta_file, encoding="ACGT")
    sig_extractor = SignalExtractor({"mappability": mappability_file})
    
    # Interval at 50M of chr1, 1000bp
    interval = Interval("chr1", 50000000, 50001000, "+")
    
    seq = seq_extractor.extract(interval) # (4, 1000)
    sig = sig_extractor.extract(interval) # (1, 1000)
    manual_combined = torch.cat([seq, sig], dim=0) # (5, 1000)
    
    # Check dimensionality: (4 + 1, 1000) = (5, 1000)
    assert manual_combined.shape == (5, 1000)

    # 2. Dataset Extraction
    # Create valid configs
    genome_dir = fasta_file.parent
    genome_config = create_human_genome_config(genome_dir)
    
    # dummy configs to pass validation
    data_config = cast(DataConfig, {
        "inputs": {"mappability": mappability_file},
        "targets": {}, # No targets for this test
        "input_len": 1000,
        "output_len": 1000,
        "bin_size": 1,
        "output_bin_size": 1,
        "max_jitter": 0,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "in_memory": False
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "sliding_window", # Dummy
        "padded_size": 1000,
        "sampler_args": {"stride": 100}
    })
    
    # Create SubsetSampler with our interval
    sampler = SubsetSampler(
        intervals=[interval],
        folds=[], 
        exclude_intervals={} 
    )
    
    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        sampler=sampler,
        transforms=[], # Explicitly no transforms
        deterministic_transforms=[]
    )
    
    item = dataset[0]
    ds_inputs = item["inputs"]
    
    assert ds_inputs.shape == manual_combined.shape
    assert torch.allclose(ds_inputs, manual_combined)
    print(f"Verified Dataset output shape: {ds_inputs.shape}")

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_mappability_extraction(mappability_file):
    # Setup SignalExtractor with mappability as a track
    extractor = SignalExtractor({"mappability": mappability_file})
    
    # 1000bp interval on chr1
    # chr1:1000000-1001000
    interval = Interval("chr1", 1000000, 1001000, "+")
    
    signal = extractor.extract(interval)
    
    # Check shape: (1, 1000)
    assert signal.shape == (1, 1000)
    assert isinstance(signal, torch.Tensor)
    
    # Check values range [0, 1]
    # Note: If region is unmappable, it might be 0. If perfectly mappable, 1.
    assert torch.all(signal >= 0.0)
    assert torch.all(signal <= 1.0)
    
    # Check that it's not all zeros (hopefully picked a region with some mappability)
    # 1Mb on chr1 should have some mappability.
    if torch.all(signal == 0):
        print("Warning: Signal is all zeros. Could be correct but worth noting.")
    else:
        print(f"Mean mappability: {signal.mean().item()}")

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_mappability_transform_binning(mappability_file):
    extractor = SignalExtractor({"mappability": mappability_file})
    interval = Interval("chr1", 1000000, 1001000, "+")
    
    # Extract raw signal
    inputs = extractor.extract(interval) # (1, 1000)
    targets = torch.randn(1, 1000) # Dummy targets
    
    # Apply Binning to INPUTS
    bin_transform = Bin(bin_size=10, method="avg", apply_to="inputs")
    
    t_inputs, t_targets, t_interval = bin_transform(inputs, targets, interval)
    
    # Check new shape: 1000 / 10 = 100
    assert t_inputs.shape == (1, 100)
    assert t_targets.shape == (1, 1000) # Targets untouched
    
    # Check values are still in range [0, 1] (avg of 0-1 is 0-1)
    assert torch.all(t_inputs >= 0.0)
    assert torch.all(t_inputs <= 1.0)

@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests")
def test_mappability_transform_jitter(mappability_file):
    extractor = SignalExtractor({"mappability": mappability_file})
    # Interval larger than needed
    interval = Interval("chr1", 1000000, 1001000, "+") # 1000bp
    
    inputs = extractor.extract(interval) # (1, 1000)
    targets = torch.zeros(1, 1000)
    
    # Jitter to crop to 500bp
    jitter = Jitter(input_len=500, max_jitter=None)
    
    t_inputs, t_targets, t_interval = jitter(inputs, targets, interval)
    
    assert t_inputs.shape == (1, 500)
    # Check interval updated
    assert t_interval.end - t_interval.start == 500
