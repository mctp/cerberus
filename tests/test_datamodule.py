from unittest.mock import MagicMock, patch

import pytest

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.datamodule import CerberusDataModule


@pytest.fixture
def mock_dataset_cls():
    with (
        patch("cerberus.datamodule.CerberusDataset") as mock,
        patch.object(CerberusDataModule, "_validate_paths"),
    ):
        yield mock


def test_datamodule_setup(
    mock_dataset_cls,
):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance

    # Mock split_folds return values
    train_ds = MagicMock()
    val_ds = MagicMock()
    test_ds = MagicMock()
    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)

    # Configs
    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)

    # Test setup with runtime params
    dm.setup(batch_size=16, num_workers=2)

    # Verify Dataset init — seed is auto-generated when not provided
    mock_dataset_cls.assert_called_once_with(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        in_memory=False,
        seed=dm.seed,
        prepare_cache=None,
    )

    # Verify split_folds
    mock_instance.split_folds.assert_called_once_with(test_fold=0, val_fold=1)

    assert dm.train_dataset == train_ds
    assert dm.val_dataset == val_ds
    assert dm.test_dataset == test_ds
    assert dm.batch_size == 16
    assert dm.num_workers == 2


def test_datamodule_dataloaders(
    mock_dataset_cls,
):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    val_ds = MagicMock()
    test_ds = MagicMock()
    train_ds.__len__.return_value = 100
    val_ds.__len__.return_value = 20
    test_ds.__len__.return_value = 20
    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    dm.setup(batch_size=16, num_workers=2)

    # Test dataloaders
    train_dl = dm.train_dataloader()
    assert train_dl.batch_size == 16
    assert train_dl.num_workers == 2
    assert train_dl.dataset == train_ds
    assert train_dl.drop_last == False  # Default

    val_dl = dm.val_dataloader()
    assert val_dl.batch_size == 16
    assert val_dl.dataset == val_ds

    test_dl = dm.test_dataloader()
    assert test_dl.batch_size == 16
    assert test_dl.dataset == test_ds


def test_datamodule_resample_via_dataloader(
    mock_dataset_cls,
):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    train_ds.__len__.return_value = 10  # Ensure non-zero length
    mock_instance.split_folds.return_value = (train_ds, MagicMock(), MagicMock())

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    dm.setup(batch_size=16, num_workers=2)

    # Mock trainer attached to datamodule
    dm.trainer = MagicMock()
    dm.trainer.global_rank = 0
    dm.trainer.current_epoch = 1
    dm.trainer.world_size = 1

    # Calling train_dataloader should trigger resample
    dm.train_dataloader()

    # Check if resample was called with correct seed (seed + epoch*world + rank)
    train_ds.resample.assert_called_once_with(seed=dm.seed + 1)


def test_datamodule_drop_last(
    mock_dataset_cls,
):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    train_ds.__len__.return_value = 100
    mock_instance.split_folds.return_value = (train_ds, MagicMock(), MagicMock())

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    # Init with drop_last=True
    dm = CerberusDataModule(genome_config, data_config, sampler_config, drop_last=True)
    dm.setup(batch_size=16, num_workers=2)

    # Test dataloader
    train_dl = dm.train_dataloader()
    assert train_dl.drop_last == True


# ---------------------------------------------------------------------------
# _validate_paths tests
# ---------------------------------------------------------------------------


def test_validate_paths_missing_fasta(tmp_path):
    """_validate_paths raises FileNotFoundError when fasta_path does not exist."""
    from pathlib import Path

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=Path(tmp_path / "nonexistent.fa"),
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    with pytest.raises(FileNotFoundError, match="Genome FASTA not found"):
        dm.setup(batch_size=4)


def test_validate_paths_missing_input_channel(tmp_path):
    """_validate_paths raises FileNotFoundError for a missing input channel file."""
    from pathlib import Path

    # Create the fasta so that check passes
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGT\n")

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=fasta,
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={"signal": Path(tmp_path / "missing_input.bw")},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    with pytest.raises(FileNotFoundError, match="input channel 'signal'"):
        dm.setup(batch_size=4)


def test_validate_paths_missing_target_channel(tmp_path):
    """_validate_paths raises FileNotFoundError for a missing target channel file."""
    from pathlib import Path

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGT\n")

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=fasta,
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={"H3K27ac": Path(tmp_path / "missing_target.bw")},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    with pytest.raises(FileNotFoundError, match="target channel 'H3K27ac'"):
        dm.setup(batch_size=4)


def test_validate_paths_all_exist(tmp_path):
    """_validate_paths passes when all configured paths exist."""

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGT\n")
    input_bw = tmp_path / "input.bw"
    input_bw.write_bytes(b"")
    target_bw = tmp_path / "target.bw"
    target_bw.write_bytes(b"")

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=fasta,
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={"signal": input_bw},
        targets={"H3K27ac": target_bw},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    with patch("cerberus.datamodule.CerberusDataset") as mock_ds:
        mock_instance = MagicMock()
        mock_ds.return_value = mock_instance
        mock_instance.split_folds.return_value = (MagicMock(), MagicMock(), MagicMock())

        dm = CerberusDataModule(genome_config, data_config, sampler_config)
        # Should not raise
        dm.setup(batch_size=4)
