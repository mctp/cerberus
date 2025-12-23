import pytest
from unittest.mock import MagicMock, patch
from cerberus.datamodule import CerberusDataModule

@pytest.fixture
def mock_dataset_cls():
    with patch("cerberus.datamodule.CerberusDataset") as mock:
        yield mock

def test_datamodule_setup(mock_dataset_cls):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    
    # Mock split_folds return values
    train_ds = MagicMock()
    val_ds = MagicMock()
    test_ds = MagicMock()
    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)
    
    # Configs
    genome_config = {"mock": "genome"}
    data_config = {"mock": "data"}
    sampler_config = {"mock": "sampler"}
    
    dm = CerberusDataModule(genome_config, data_config, sampler_config, batch_size=16)
    
    # Test setup
    dm.setup()
    
    # Verify Dataset init
    mock_dataset_cls.assert_called_once_with(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config
    )
    
    # Verify split_folds
    mock_instance.split_folds.assert_called_once_with(test_fold=0, val_fold=1)
    
    assert dm.train_dataset == train_ds
    assert dm.val_dataset == val_ds
    assert dm.test_dataset == test_ds

def test_datamodule_dataloaders(mock_dataset_cls):
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
    
    dm = CerberusDataModule({}, {}, {}, batch_size=16)
    dm.setup()
    
    # Test dataloaders
    train_dl = dm.train_dataloader()
    assert train_dl.batch_size == 16
    assert train_dl.dataset == train_ds
    
    val_dl = dm.val_dataloader()
    assert val_dl.batch_size == 16
    assert val_dl.dataset == val_ds
    
    test_dl = dm.test_dataloader()
    assert test_dl.batch_size == 16
    assert test_dl.dataset == test_ds

def test_datamodule_resample_via_dataloader(mock_dataset_cls):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    train_ds.__len__.return_value = 10  # Ensure non-zero length
    mock_instance.split_folds.return_value = (train_ds, MagicMock(), MagicMock())
    
    dm = CerberusDataModule({}, {}, {})
    dm.setup()
    
    # Mock trainer attached to datamodule
    dm.trainer = MagicMock()
    dm.trainer.global_rank = 0
    dm.trainer.current_epoch = 1
    
    # Calling train_dataloader should trigger resample
    dm.train_dataloader()
    
    # Check if resample was called with correct seed (epoch + rank = 1 + 0 = 1)
    train_ds.resample.assert_called_once_with(seed=1)
