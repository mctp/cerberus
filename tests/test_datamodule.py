import pytest
from unittest.mock import MagicMock, patch
from cerberus.datamodule import CerberusDataModule
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, FoldArgs, RandomSamplerArgs

@pytest.fixture
def mock_dataset_cls():
    with patch("cerberus.datamodule.CerberusDataset") as mock:
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
        fold_args=FoldArgs.model_construct(k=5, test_fold=0, val_fold=1),
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
        sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
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
        fold_args=FoldArgs.model_construct(k=5, test_fold=0, val_fold=1),
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
        sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    dm.setup(batch_size=16, num_workers=2)

    # Test dataloaders
    train_dl = dm.train_dataloader()
    assert train_dl.batch_size == 16
    assert train_dl.num_workers == 2
    assert train_dl.dataset == train_ds
    assert train_dl.drop_last == False # Default

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
        fold_args=FoldArgs.model_construct(k=5, test_fold=0, val_fold=1),
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
        sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
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
        fold_args=FoldArgs.model_construct(k=5, test_fold=0, val_fold=1),
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
        sampler_args=RandomSamplerArgs.model_construct(num_intervals=10),
    )

    # Init with drop_last=True
    dm = CerberusDataModule(genome_config, data_config, sampler_config, drop_last=True)
    dm.setup(batch_size=16, num_workers=2)

    # Test dataloader
    train_dl = dm.train_dataloader()
    assert train_dl.drop_last == True
