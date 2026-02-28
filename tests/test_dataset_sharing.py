
import pytest
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig
from typing import cast
from cerberus.dataset import CerberusDataset

@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "mock.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 1000 + "\n")
    return fasta_path

def test_dataset_extractor_sharing(mock_fasta):
    genome_config = cast(GenomeConfig, {
        "name": "mock_genome",
        "fasta_path": str(mock_fasta),
        "chrom_sizes": {"chr1": 1000},
        "allowed_chroms": ["chr1"],
        "fold_type": "chrom_partition",
        "fold_args": {"k": 3},
        "exclude_intervals": {}
    })
    data_config = cast(DataConfig, {
        "encoding": "ACGT",
        "inputs": {},
        "targets": {},
        "input_len": 10,
        "output_len": 10,
        "max_jitter": 0,
        "reverse_complement": False,
        "target_scale": 1.0,
        "count_pseudocount": 1.0,
        "log_transform": False,
        "output_bin_size": 1,
        "use_sequence": True,
    })
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "sliding_window",
        "padded_size": 10,
        "sampler_args": {"stride": 10}
    })

    # Initialize full dataset
    full_dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        in_memory=True,
    )

    # Split folds
    train, val, test = full_dataset.split_folds(test_fold=0, val_fold=1)

    # Verify memory sharing
    # The extractor object should be exactly the same instance
    assert full_dataset.sequence_extractor is not None
    assert train.sequence_extractor is full_dataset.sequence_extractor
    assert val.sequence_extractor is full_dataset.sequence_extractor
    assert test.sequence_extractor is full_dataset.sequence_extractor
    
    # Also verify that the underlying cache is populated and shared (implicit since object is shared)
    # Accessing one should affect the others if we were to mutate (which we shouldn't)
    
    # Verify in-memory type
    from cerberus.sequence import InMemorySequenceExtractor
    assert isinstance(full_dataset.sequence_extractor, InMemorySequenceExtractor)
    
    # Check that tensors are in shared memory (since we added .share_memory_())
    # Note: On CPU, is_shared() might return False unless multiprocessing is involved or moved to shm explicitly?
    # Actually tensor.share_memory_() moves it to shared memory.
    # Let's check a cached tensor.
    cached_tensor = full_dataset.sequence_extractor._cache["chr1"]
    # For CPU tensors, share_memory_() makes them shareable. is_shared() checks if it's in shared memory.
    assert cached_tensor.is_shared()
