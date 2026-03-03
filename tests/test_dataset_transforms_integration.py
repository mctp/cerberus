import pytest
from cerberus.dataset import CerberusDataset
from cerberus.transform import Jitter, TargetCrop, Log1p, Bin, ReverseComplement, Compose
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig, SamplerConfig
from typing import cast

@pytest.fixture
def mock_genome(tmp_path):
    genome_path = tmp_path / "genome.fa"
    genome_path.touch()
    fai_path = tmp_path / "genome.fa.fai"
    fai_path.write_text("chr1\t1000\t0\t80\t81\n")
    return genome_path

def test_dataset_auto_transforms(mock_genome):
    # Config: Jitter(10), Shrink(50), Bin(2), Log, RC
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "output_bin_size": 2,
        "max_jitter": 10,
        "log_transform": True,
        "reverse_complement": True,
        "target_scale": 1.0,
        "count_pseudocount": 1.0,
        "encoding": "ACGT",
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 120, # 100 + 2*10
        "sampler_args": {"intervals_path": "dummy.bed"} # Mocked in sampler, but here we just need init
    })
    
    # We need a valid sampler init or mock it.
    # Dataset init calls _initialize_sampler which checks file existence.
    # So we need a dummy file.
    dummy_bed = mock_genome.parent / "dummy.bed"
    dummy_bed.write_text("chr1\t100\t200\n")
    sampler_config['sampler_args']['intervals_path'] = str(dummy_bed)

    genome_config = create_genome_config(name="test", fasta_path=mock_genome, species="human")

    ds = CerberusDataset(genome_config, data_config, sampler_config)
    
    # Check transforms
    assert ds.transforms is not None
    assert isinstance(ds.transforms, Compose)
    
    transforms = ds.transforms.transforms
    
    # Expected order: Jitter, RC, TargetCrop, Bin, Log
    # 1. Jitter
    assert isinstance(transforms[0], Jitter)
    assert transforms[0].input_len == 100
    assert transforms[0].max_jitter == 10

    # 2. RC
    assert isinstance(transforms[1], ReverseComplement)
    
    # 3. Shrinkage
    assert isinstance(transforms[2], TargetCrop)
    assert transforms[2].output_len == 50
    
    # 4. Binning
    assert isinstance(transforms[3], Bin)
    assert transforms[3].bin_size == 2
    
    # 5. Log
    assert isinstance(transforms[4], Log1p)

def test_dataset_no_jitter_defaults(mock_genome):
    # Config: No Jitter, Input=Output=100
    data_config = cast(DataConfig, {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 100,
        "output_bin_size": 1,
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "count_pseudocount": 1.0,
        "encoding": "ACGT",
        "use_sequence": True,
    })
    
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 100,
        "sampler_args": {"intervals_path": str(mock_genome.parent / "dummy.bed")}
    })
    # Ensure dummy bed exists from fixture or prev test? No, fixture is fresh.
    (mock_genome.parent / "dummy.bed").write_text("chr1\t100\t200\n")

    genome_config = create_genome_config(name="test", fasta_path=mock_genome, species="human")

    ds = CerberusDataset(genome_config, data_config, sampler_config)
    
    transforms = ds.transforms.transforms
    
    # 1. Jitter (Input len enforcement, max_jitter=0 acting as CenterCrop)
    assert isinstance(transforms[0], Jitter)
    assert transforms[0].input_len == 100
    assert transforms[0].max_jitter == 0
    
    # No other transforms
    assert len(transforms) == 1
