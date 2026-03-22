from pathlib import Path

import pytest
from pydantic import ValidationError

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config


def test_create_genome_config_valid(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.touch()
    (tmp_path / "genome.fa.fai").touch()

    # Test valid full config
    config = create_genome_config(name="genome", fasta_path=genome, species="human")
    assert config.fasta_path == genome
    assert config.name == "genome"

    config = create_genome_config(name="hg38", fasta_path=genome, species="human")
    assert config.fasta_path == genome
    assert config.name == "hg38"


def test_genome_config_missing_required_fields():
    """GenomeConfig with missing required fields should raise ValidationError."""
    with pytest.raises(ValidationError):
        GenomeConfig(name="test")  # type: ignore[call-arg]


def test_genome_config_allowed_chroms_invalid(tmp_path):
    """allowed_chroms must be a list, not a string."""
    genome = tmp_path / "genome.fa"
    genome.touch()
    (tmp_path / "genome.fa.fai").touch()

    with pytest.raises(ValidationError):
        GenomeConfig(
            name="test",
            fasta_path=genome,
            exclude_intervals={},
            allowed_chroms="chr1",  # type: ignore[arg-type]
            chrom_sizes={"chr1": 1000},
            fold_type="chrom_partition",
            fold_args={"k": 5},
        )


def test_create_genome_config_missing_file(tmp_path):
    """create_genome_config raises FileNotFoundError for missing FASTA."""
    with pytest.raises(FileNotFoundError):
        create_genome_config(
            name="test", fasta_path=tmp_path / "missing.fa", species="human"
        )


def test_data_config_valid(tmp_path):
    cons = tmp_path / "cons.bw"
    cons.touch()
    counts = tmp_path / "counts.bw"
    counts.touch()

    config = DataConfig(
        inputs={"cons": cons},
        targets={"counts": counts},
        input_len=2048,
        output_len=1024,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    # Check that values are converted to Path objects
    assert isinstance(config.inputs["cons"], Path)
    assert config.inputs["cons"] == cons
    assert config.output_len == 1024
    assert config.input_len == 2048


def test_data_config_missing_key():
    """DataConfig with missing required fields should raise ValidationError."""
    with pytest.raises(ValidationError):
        DataConfig(
            inputs={},
            output_len=1024,
            use_sequence=True,
        )  # type: ignore[call-arg]


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
        allowed_chroms=["chr1"],
    )

    data_config = DataConfig(
        inputs={},
        targets={},
        input_len=100,
        output_len=50,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig(
        sampler_type="interval",
        padded_size=100,
        sampler_args={"intervals_path": peaks},
    )
    ds = CerberusDataset(
        genome_config,
        data_config,
        sampler_config,
        sequence_extractor=None,
        sampler=None,
        exclude_intervals=None,
    )

    # Check GenomeConfig object
    assert ds.genome_config.name == "test_genome"
    assert ds.genome_config.fasta_path == genome_path
    assert ds.genome_config.chrom_sizes is not None
    assert "chr1" in ds.genome_config.chrom_sizes
    assert "chr2" not in ds.genome_config.chrom_sizes


def test_data_config_accepts_missing_file(tmp_path):
    """DataConfig accepts paths without checking existence."""
    cfg = DataConfig(
        inputs={"cons": tmp_path / "missing.bw"},
        targets={},
        input_len=2048,
        output_len=1024,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    assert cfg.inputs["cons"] == tmp_path / "missing.bw"


def test_genome_config_invalid_chrom_sizes(tmp_path):
    """chrom_sizes values must be integers; non-numeric strings should fail."""
    genome = tmp_path / "genome.fa"
    genome.touch()

    with pytest.raises(ValidationError):
        GenomeConfig(
            name="test",
            fasta_path=genome,
            allowed_chroms=["chr1"],
            exclude_intervals={},
            fold_type="chrom_partition",
            fold_args={"k": 5},
            chrom_sizes={"chr1": [1, 2, 3]},  # type: ignore[dict-item]
        )


def test_genome_config_fold_args_accepts_any_dict(tmp_path):
    """fold_args is now dict[str, Any] — no type validation at model level.

    Type checking of fold_args values (e.g. k must be int, test_fold must be
    non-negative) was removed when FoldArgs was deleted. The plain dict is
    accepted as-is; runtime consumers use bracket access.
    """
    genome = tmp_path / "genome.fa"
    genome.touch()

    # Extra or unconventional values are accepted at config level
    cfg = GenomeConfig(
        name="test",
        fasta_path=genome,
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "val_fold": [1, 2]},
        chrom_sizes={"chr1": 100},
    )
    assert cfg.fold_args["val_fold"] == [1, 2]

    cfg2 = GenomeConfig(
        name="test",
        fasta_path=genome,
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": -1},
        chrom_sizes={"chr1": 100},
    )
    assert cfg2.fold_args["test_fold"] == -1
