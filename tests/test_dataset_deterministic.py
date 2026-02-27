import pytest
from typing import cast
from cerberus.dataset import CerberusDataset
from cerberus.transform import Jitter, ReverseComplement, DataTransform
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig, SamplerConfig

@pytest.fixture
def basic_files(tmp_path):
    # Genome files
    genome = tmp_path / "genome.fa"
    genome.touch()
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t1000\t0\t80\t81\n")
    
    # Peaks file
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\n")
    
    return genome, peaks

def test_deterministic_transforms_auto_generation(basic_files):
    """Test that deterministic transforms are auto-generated from config."""
    genome_path, peaks_path = basic_files
    
    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human"
    )
    
    # Config with augmentation enabled
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 100,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 50, # Jitter enabled
        "log_transform": False,
        "reverse_complement": True, # RC enabled
        "target_scale": 1.0,
        "use_sequence": True,
    })

    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 200, # 100 input + 2*50 jitter
        "sampler_args": {"intervals_path": str(peaks_path)}
    })

    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config
    )

    # Check main transforms (should have augmentation)
    train_transforms = dataset.transforms.transforms
    jitter_train = next(t for t in train_transforms if isinstance(t, Jitter))
    assert jitter_train.max_jitter == 50
    assert any(isinstance(t, ReverseComplement) for t in train_transforms)

    # Check deterministic transforms (should have NO augmentation)
    det_transforms = dataset.deterministic_transforms.transforms
    jitter_det = next(t for t in det_transforms if isinstance(t, Jitter))
    assert jitter_det.max_jitter == 0
    assert not any(isinstance(t, ReverseComplement) for t in det_transforms)

def test_split_folds_deterministic_behavior(basic_files):
    """Test that split_folds returns deterministic datasets for val/test."""
    genome_path, peaks_path = basic_files
    
    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human"
    )
    
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 100,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 50,
        "log_transform": False,
        "reverse_complement": True,
        "target_scale": 1.0,
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 200,
        "sampler_args": {"intervals_path": str(peaks_path)}
    })

    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config
    )

    train_ds, val_ds, test_ds = dataset.split_folds()

    assert train_ds.is_train is True
    assert val_ds.is_train is False
    assert test_ds.is_train is False

    # Verify active transforms for val_ds
    # Since is_train=False, calling .transforms on it won't reveal which one is USED by get_interval
    # But we can check properties
    assert val_ds.transforms is not None
    assert val_ds.deterministic_transforms is not None
    
    # We can also check that _get_interval actually uses the deterministic one
    # by patching or mocking, but checking the object properties and is_train flag is sufficient 
    # given we tested the logic in unit tests.
    
    # Check that deterministic transforms in val_ds are correct
    det_transforms = val_ds.deterministic_transforms.transforms
    jitter_det = next(t for t in det_transforms if isinstance(t, Jitter))
    assert jitter_det.max_jitter == 0

def test_manual_transforms_validation(basic_files):
    """Test validation of transform arguments."""
    genome_path, peaks_path = basic_files
    
    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human"
    )
    
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
        "sampler_args": {"intervals_path": str(peaks_path)}
    })
    
    custom_transforms: list[DataTransform] = [Jitter(input_len=100, max_jitter=50)]

    # 1. Provide only transforms -> Should raise ValueError
    with pytest.raises(ValueError, match="Both 'transforms' and 'deterministic_transforms' must be provided"):
        CerberusDataset(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            transforms=custom_transforms
        )

    # 2. Provide both -> Should pass
    CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        transforms=custom_transforms,
        deterministic_transforms=custom_transforms # Just for testing
    )
