from unittest.mock import MagicMock, patch

from cerberus.config import GenomeConfig, ModelConfig, TrainConfig
from cerberus.train import train_multi


def _make_genome_config(k: int = 3) -> MagicMock:
    gc = MagicMock(spec=GenomeConfig)
    gc.fold_args = {"k": k, "test_fold": None, "val_fold": None}
    gc.model_copy.return_value = gc
    return gc


def test_train_multi_loop():
    # Mocks
    genome_config = _make_genome_config(k=3)
    data_config = MagicMock()
    sampler_config = MagicMock()
    model_config = MagicMock(spec=ModelConfig)
    model_config.model_cls = MagicMock()
    model_config.loss_cls = MagicMock()
    model_config.loss_args = {}
    model_config.metrics_cls = MagicMock()
    model_config.metrics_args = {}
    model_config.model_args = {
        "input_channels": ["A"],
        "output_channels": ["B"],
    }
    model_config.pretrained = []

    train_config = TrainConfig(
        batch_size=32,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        patience=5,
        optimizer="adamw",
        scheduler_type="default",
        scheduler_args={},
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )

    # Patches
    with (
        patch("cerberus.train.CerberusDataModule") as mock_dm_cls,
        patch("cerberus.train.instantiate") as mock_instantiate,
        patch("pytorch_lightning.Trainer") as mock_trainer_cls,
        patch("pathlib.Path.mkdir"),
        patch("cerberus.train.update_ensemble_metadata"),
    ):
        trainers = train_multi(
            genome_config,
            data_config,
            sampler_config,
            model_config,
            train_config,
            root_dir="test_runs",
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
            seed=42,
        )
        # Call 2: test=1, val=2
        mock_dm_cls.assert_any_call(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=1,
            val_fold=2,
            seed=42,
        )
        # Call 3: test=2, val=0
        mock_dm_cls.assert_any_call(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            test_fold=2,
            val_fold=0,
            seed=42,
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
