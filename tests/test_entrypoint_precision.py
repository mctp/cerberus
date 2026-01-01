from unittest.mock import MagicMock, patch
from typing import cast
from cerberus.entrypoints import train
from cerberus.config import TrainConfig
import pytorch_lightning as pl

def test_train_wrapper_matmul_precision():
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
    
    # Test default (highest)
    with patch("pytorch_lightning.Trainer"), \
         patch("torch.set_float32_matmul_precision") as mock_set_precision:
        
        train(
            model, 
            datamodule, 
            train_config, 
            num_workers=0, 
            in_memory=False, 
            logger=False
        )
        
        mock_set_precision.assert_called_once_with("highest")
        
    # Test medium
    with patch("pytorch_lightning.Trainer"), \
         patch("torch.set_float32_matmul_precision") as mock_set_precision:
        
        train(
            model, 
            datamodule, 
            train_config, 
            num_workers=0, 
            in_memory=False, 
            logger=False,
            matmul_precision="medium"
        )
        
        mock_set_precision.assert_called_once_with("medium")
