import pytest
from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.datamodule import CerberusDataModule
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig

@pytest.fixture
def mock_dataset_cls():
    with patch("cerberus.datamodule.CerberusDataset") as mock:
        yield mock

@patch("cerberus.datamodule.validate_genome_config")
@patch("cerberus.datamodule.validate_data_config")
@patch("cerberus.datamodule.validate_sampler_config")
@patch("cerberus.datamodule.validate_data_and_sampler_compatibility")
def test_datamodule_setup(
    mock_validate_compatibility,
    mock_validate_sampler,
    mock_validate_data,
    mock_validate_genome,
    mock_dataset_cls,
):
    # Setup mocks for validation
    mock_validate_genome.side_effect = lambda x: x
    mock_validate_data.side_effect = lambda x: x
    mock_validate_sampler.side_effect = lambda x: x

    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    
    # Mock split_folds return values
    train_ds = MagicMock()
    val_ds = MagicMock()
    test_ds = MagicMock()
    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)
    
    # Configs
    genome_config = cast(GenomeConfig, {"mock": "genome", "fold_args": {"test_fold": 0, "val_fold": 1}})
    data_config = cast(DataConfig, {"mock": "data"})
    sampler_config = cast(SamplerConfig, {"mock": "sampler", "sampler_type": "random"})
    
    dm = CerberusDataModule(genome_config, data_config, sampler_config)
    
    # Test setup with runtime params
    dm.setup(batch_size=16, num_workers=2)
    
    # Verify Dataset init
    mock_dataset_cls.assert_called_once_with(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        in_memory=False,
        seed=None,
        prepare_cache=None,
    )
    
    # Verify validation was called
    mock_validate_genome.assert_called_once_with(genome_config)
    mock_validate_data.assert_called_once_with(data_config)
    mock_validate_sampler.assert_called_once_with(sampler_config)
    mock_validate_compatibility.assert_called_once()
    
    # Verify split_folds
    mock_instance.split_folds.assert_called_once_with(test_fold=0, val_fold=1)
    
    assert dm.train_dataset == train_ds
    assert dm.val_dataset == val_ds
    assert dm.test_dataset == test_ds
    assert dm.batch_size == 16
    assert dm.num_workers == 2

@patch("cerberus.datamodule.validate_genome_config")
@patch("cerberus.datamodule.validate_data_config")
@patch("cerberus.datamodule.validate_sampler_config")
@patch("cerberus.datamodule.validate_data_and_sampler_compatibility")
def test_datamodule_dataloaders(
    mock_validate_compatibility,
    mock_validate_sampler,
    mock_validate_data,
    mock_validate_genome,
    mock_dataset_cls,
):
    # Setup mocks for validation
    mock_validate_genome.side_effect = lambda x: x
    mock_validate_data.side_effect = lambda x: x
    mock_validate_sampler.side_effect = lambda x: x

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
    
    dm = CerberusDataModule(
        cast(GenomeConfig, {"fold_args": {"test_fold": 0, "val_fold": 1}}),
        cast(DataConfig, {}),
        cast(SamplerConfig, {"sampler_type": "random"}),
    )
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

@patch("cerberus.datamodule.validate_genome_config")
@patch("cerberus.datamodule.validate_data_config")
@patch("cerberus.datamodule.validate_sampler_config")
@patch("cerberus.datamodule.validate_data_and_sampler_compatibility")
def test_datamodule_resample_via_dataloader(
    mock_validate_compatibility,
    mock_validate_sampler,
    mock_validate_data,
    mock_validate_genome,
    mock_dataset_cls,
):
    # Setup mocks for validation
    mock_validate_genome.side_effect = lambda x: x
    mock_validate_data.side_effect = lambda x: x
    mock_validate_sampler.side_effect = lambda x: x

    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    train_ds.__len__.return_value = 10  # Ensure non-zero length
    mock_instance.split_folds.return_value = (train_ds, MagicMock(), MagicMock())
    
    dm = CerberusDataModule(
        cast(GenomeConfig, {"fold_args": {"test_fold": 0, "val_fold": 1}}),
        cast(DataConfig, {}),
        cast(SamplerConfig, {"sampler_type": "random"}),
    )
    dm.setup(batch_size=16, num_workers=2)
    
    # Mock trainer attached to datamodule
    dm.trainer = MagicMock()
    dm.trainer.global_rank = 0
    dm.trainer.current_epoch = 1
    dm.trainer.world_size = 1
    
    # Calling train_dataloader should trigger resample
    dm.train_dataloader()
    
    # Check if resample was called with correct seed (epoch + rank = 1 + 0 = 1)
    train_ds.resample.assert_called_once_with(seed=1)

@patch("cerberus.datamodule.validate_genome_config")
@patch("cerberus.datamodule.validate_data_config")
@patch("cerberus.datamodule.validate_sampler_config")
@patch("cerberus.datamodule.validate_data_and_sampler_compatibility")
def test_datamodule_drop_last(
    mock_validate_compatibility,
    mock_validate_sampler,
    mock_validate_data,
    mock_validate_genome,
    mock_dataset_cls,
):
    # Setup mocks for validation
    mock_validate_genome.side_effect = lambda x: x
    mock_validate_data.side_effect = lambda x: x
    mock_validate_sampler.side_effect = lambda x: x

    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    train_ds.__len__.return_value = 100
    mock_instance.split_folds.return_value = (train_ds, MagicMock(), MagicMock())
    
    # Init with drop_last=True
    dm = CerberusDataModule(
        cast(GenomeConfig, {"fold_args": {"test_fold": 0, "val_fold": 1}}),
        cast(DataConfig, {}),
        cast(SamplerConfig, {"sampler_type": "random"}),
        drop_last=True
    )
    dm.setup(batch_size=16, num_workers=2)
    
    # Test dataloader
    train_dl = dm.train_dataloader()
    assert train_dl.drop_last == True
