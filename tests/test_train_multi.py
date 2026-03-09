from typing import cast
from unittest.mock import MagicMock, patch
from cerberus.train import train_multi
from cerberus.config import TrainConfig, GenomeConfig, ModelConfig

def test_train_multi_loop():
    # Mocks
    genome_config = cast(GenomeConfig, {
        "fold_args": {"k": 3}
    })
    data_config = MagicMock(spec=dict)
    def data_config_getitem(k):
        if k == "input_len": return 10
        if k == "output_len": return 10
        if k == "output_bin_size": return 1
        if k == "count_pseudocount": return 1.0
        if k == "target_scale": return 1.0
        return None
    data_config.__getitem__.side_effect = data_config_getitem
    
    sampler_config = MagicMock(spec=dict)
    model_config = cast(ModelConfig, {
        "model_cls": MagicMock(),
        "loss_cls": MagicMock(),
        "loss_args": {},
        "metrics_cls": MagicMock(),
        "metrics_args": {},
        "model_args": {
            "input_channels": ["A"],
            "output_channels": ["B"],
        }
    })
    
    train_config = cast(TrainConfig, {
        "batch_size": 32,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "patience": 5,
        "optimizer": "adamw",
        "filter_bias_and_bn": True,
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    })

    # Patches
    with patch("cerberus.train.CerberusDataModule") as mock_dm_cls, \
         patch("cerberus.train.instantiate") as mock_instantiate, \
         patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("cerberus.train.update_ensemble_metadata") as mock_metadata:
        
        mock_trainer_instance = mock_trainer_cls.return_value
        
        trainers = train_multi(
            genome_config,
            data_config,
            sampler_config,
            model_config,
            train_config,
            seed=1234,
            root_dir="test_runs"
        )
        
        # Verify k=3 iterations
        assert len(trainers) == 3
        assert mock_dm_cls.call_count == 3
        assert mock_instantiate.call_count == 3
        assert mock_trainer_cls.call_count == 3
        
        # Check fold arguments for DataModule
        # Call 1: test=0, val=1
        mock_dm_cls.assert_any_call(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=0,
            val_fold=1,
            seed=1234,
        )
        # Call 2: test=1, val=2
        mock_dm_cls.assert_any_call(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=1,
            val_fold=2,
            seed=1234,
        )
        # Call 3: test=2, val=0
        mock_dm_cls.assert_any_call(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=2,
            val_fold=0,
            seed=1234,
        )
        
        # Check logging directories passed to Trainer via CSVLogger check in train
        # Wait, train creates CSVLogger if not passed.
        # train_multi calls train with default_root_dir
        
        # Let's inspect calls to train (which calls pl.Trainer)
        # We can check what default_root_dir was passed to pl.Trainer constructor
        # because train passes **trainer_kwargs to pl.Trainer
        
        # The call_args_list of Trainer should show different directories
        # Trainer init args
        args_list = mock_trainer_cls.call_args_list
        
        loggers = [call.kwargs.get("logger") for call in args_list]
        assert len(loggers) == 3
        
        # Verify logger paths
        save_dirs = []
        for l in loggers:
            assert l is not None
            save_dirs.append(l.save_dir)

        assert any("fold_0" in str(d) for d in save_dirs)
        assert any("fold_1" in str(d) for d in save_dirs)
        assert any("fold_2" in str(d) for d in save_dirs)
