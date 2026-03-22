import pytest

from cerberus.config import DataConfig
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config


def test_dataset_no_sampler_implicit(tmp_path):
    # Setup
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "N" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t1000\t6\t1000\t1001\n")

    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
    )

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=200,
        output_len=200,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    # Init without sampler config
    ds = CerberusDataset(genome_config, data_config, sampler_config=None, sampler=None)

    # Assertions
    assert ds.sampler is None
    assert len(ds) == 0

    # Verify get_interval works
    # chr1:100-300 (len 200)
    out = ds.get_interval("chr1:100-300")
    assert out["inputs"].shape == (4, 200)

    # Verify iteration fails
    with pytest.raises(TypeError, match="Dataset has no sampler configured"):
        _ = ds[0]

    # Verify split_folds returns empty datasets
    train, val, test = ds.split_folds()
    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == 0
    assert train.sampler is None
    assert val.sampler is None
    assert test.sampler is None

    # Verify resample does not crash
    ds.resample()


def test_dataset_no_sampler_explicit_none_config(tmp_path):
    # Similar to above but explicitly passing None for sampler_config
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "N" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t1000\t6\t1000\t1001\n")

    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
    )

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=200,
        output_len=200,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    ds = CerberusDataset(genome_config, data_config, sampler_config=None)
    assert ds.sampler is None
    assert len(ds) == 0
