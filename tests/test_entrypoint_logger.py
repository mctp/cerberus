from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.entrypoints import train
from cerberus.config import TrainConfig
import pytorch_lightning as pl

def test_train_wrapper_logger_setup():
    # Mock model and datamodule
    model = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    
    train_config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 1,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
    })
    
    # Patch pl.Trainer and CSVLogger
    # Since CSVLogger is imported inside the function, we patch the source where it comes from.
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
         patch("pytorch_lightning.loggers.CSVLogger") as mock_csv_logger_cls:
        
        train(
            model, 
            datamodule, 
            train_config, 
            num_workers=0, 
            in_memory=False, 
            logger=True, # Trigger the logic
            default_root_dir="/tmp/logs"
        )
        
        # Verify CSVLogger was instantiated with correct dir
        mock_csv_logger_cls.assert_called_once_with(save_dir="/tmp/logs")
        
        # Verify Trainer was called with the logger instance
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs["logger"] == mock_csv_logger_cls.return_value

def test_train_wrapper_logger_setup_default_dir():
    # Test fallback to "." when no default_root_dir
    model = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    
    train_config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 1,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
    })
    
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
         patch("pytorch_lightning.loggers.CSVLogger") as mock_csv_logger_cls:
        
        train(
            model, 
            datamodule, 
            train_config, 
            num_workers=0, 
            in_memory=False, 
            logger=True
        )
        
        # Verify CSVLogger was instantiated with "."
        mock_csv_logger_cls.assert_called_once_with(save_dir=".")
