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
def test_val_batch_size_configuration(
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
    # Mocks for len so dataloader init works
    train_ds.__len__.return_value = 10
    val_ds.__len__.return_value = 10
    test_ds.__len__.return_value = 10
    
    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)
    
    dm = CerberusDataModule(
        cast(GenomeConfig, {"fold_args": {}}),
        cast(DataConfig, {}),
        cast(SamplerConfig, {}),
    )
    
    # 1. Test default behavior (no val_batch_size provided)
    dm.setup(batch_size=16)
    assert dm.batch_size == 16
    assert dm.val_batch_size == 16
    
    assert dm.train_dataloader().batch_size == 16
    assert dm.val_dataloader().batch_size == 16
    assert dm.test_dataloader().batch_size == 16

    # Reset
    dm._is_initialized = False 
    
    # 2. Test explicit val_batch_size
    dm.setup(batch_size=16, val_batch_size=32)
    assert dm.batch_size == 16
    assert dm.val_batch_size == 32
    
    assert dm.train_dataloader().batch_size == 16
    assert dm.val_dataloader().batch_size == 32
    assert dm.test_dataloader().batch_size == 32

    # Reset
    dm._is_initialized = False 
    
    # 3. Test setup without changing default (batch_size None)
    # If setup is called again (e.g. stage='test'), we preserve existing
    
    dm.setup(batch_size=8)
    # Should update val_batch_size too because logic says:
    # if val_batch_size is not None: set it
    # elif batch_size is not None: set val_batch_size = batch_size
    assert dm.batch_size == 8
    assert dm.val_batch_size == 8
    
    # Reset
    dm._is_initialized = False 
    
    dm.setup(batch_size=10, val_batch_size=20)
    assert dm.batch_size == 10
    assert dm.val_batch_size == 20
