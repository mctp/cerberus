import pytest
from unittest.mock import MagicMock, patch
from cerberus.entrypoints import train
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def test_train_wrapper_calls_trainer_fit():
    # Mock model and datamodule
    model = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    
    train_config = {
        "max_epochs": 10,
        "patience": 5
    }
    
    # Patch pl.Trainer
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls:
        mock_trainer_instance = mock_trainer_cls.return_value
        
        train(model, datamodule, train_config, accelerator="cpu")
        
        # Verify Trainer init
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs["max_epochs"] == 10
        assert call_kwargs["accelerator"] == "cpu"
        
        # Verify callbacks
        callbacks = call_kwargs["callbacks"]
        # Should have at least 3 default callbacks
        assert len(callbacks) >= 3
        # Check for specific callbacks types
        callback_types = [type(cb) for cb in callbacks]
        assert LearningRateMonitor in callback_types
        assert ModelCheckpoint in callback_types
        assert EarlyStopping in callback_types
        
        # Verify fit called
        mock_trainer_instance.fit.assert_called_once_with(model, datamodule=datamodule)

def test_train_wrapper_custom_callbacks():
    model = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    train_config = {"max_epochs": 1, "patience": 1}
    
    custom_cb = MagicMock(spec=pl.Callback)
    
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls:
        train(model, datamodule, train_config, callbacks=[custom_cb])
        
        call_kwargs = mock_trainer_cls.call_args[1]
        callbacks = call_kwargs["callbacks"]
        assert custom_cb in callbacks
