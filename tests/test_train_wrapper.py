from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.train import _train as train
from cerberus.config import TrainConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def test_train_wrapper_calls_trainer_fit():
    # Mock model and datamodule
    model = MagicMock(spec=pl.LightningModule)
    datamodule = MagicMock()
    
    train_config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 5,
        "optimizer": "adamw",
        "scheduler_type": "default",
        "scheduler_args": {},
        "filter_bias_and_bn": True,
    })
    
    # Patch pl.Trainer
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls:
        mock_trainer_instance = mock_trainer_cls.return_value
        
        train(model, datamodule, train_config, num_workers=2, in_memory=False, accelerator="cpu")
        
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
        
        # Verify datamodule setup called with runtime params
        datamodule.setup.assert_called_once_with(
            batch_size=32,
            val_batch_size=None,
            num_workers=2,
            in_memory=False
        )

def test_train_wrapper_custom_callbacks():
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
    
    custom_cb = MagicMock(spec=pl.Callback)
    
    with patch("pytorch_lightning.Trainer") as mock_trainer_cls:
        train(model, datamodule, train_config, num_workers=2, in_memory=False, callbacks=[custom_cb])
        
        call_kwargs = mock_trainer_cls.call_args[1]
        callbacks = call_kwargs["callbacks"]
        assert custom_cb in callbacks
