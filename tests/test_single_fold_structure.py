
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import yaml
from cerberus.train import train_single, train_multi
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig

@pytest.fixture
def minimal_configs():
    genome_config: GenomeConfig = {
        "name": "test_genome",
        "fasta_path": Path("genome.fa"),
        "allowed_chroms": ["chr1"],
        "fold_type": "chrom_partition",
        "chrom_sizes": {"chr1": 1000},
        "fold_args": {"k": 2, "test_fold": 0, "val_fold": 1},
        "exclude_intervals": {}
    }

    data_config: DataConfig = {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 10,
        "output_bin_size": 1,
        "use_sequence": False,
        "max_jitter": 0,
        "encoding": "ACGT",
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
    }

    sampler_config: SamplerConfig = {
        "sampler_type": "random",
        "padded_size": 100,
        "sampler_args": {"num_samples": 10}
    }

    model_config: ModelConfig = {
        "name": "TestModel",
        "model_cls": "cerberus.models.bpnet.BPNet",
        "loss_cls": "cerberus.models.bpnet.BPNetLoss",
        "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
        "loss_args": {},
        "metrics_args": {},
        "model_args": {}
    }

    train_config: TrainConfig = {
        "batch_size": 32,
        "max_epochs": 1,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "filter_bias_and_bn": False,
        "patience": 1,
        "scheduler_type": "default",
        "scheduler_args": {},
        "reload_dataloaders_every_n_epochs": 0,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }

    return genome_config, data_config, sampler_config, model_config, train_config

def test_train_single_creates_structure(tmp_path, minimal_configs):
    genome_config, data_config, sampler_config, model_config, train_config = minimal_configs
    
    # Mock dependencies
    with patch("cerberus.train.CerberusDataModule") as mock_dm_cls, \
         patch("cerberus.train.instantiate") as mock_instantiate, \
         patch("cerberus.train._train") as mock_train:
        
        root_dir = tmp_path / "output"
        
        # 1. Test explicit test_fold=0
        train_single(
            genome_config, data_config, sampler_config, model_config, train_config,
            test_fold=0,
            root_dir=root_dir
        )
        
        # Verify metadata file
        meta_path = root_dir / "ensemble_metadata.yaml"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0]
        
        # Verify train called with correct root_dir (fold_0 subdirectory)
        mock_train.assert_called()
        call_kwargs = mock_train.call_args[1]
        expected_dir = root_dir / "fold_0"
        assert Path(call_kwargs["root_dir"]) == expected_dir

def test_train_single_default_behavior(tmp_path, minimal_configs):
    genome_config, data_config, sampler_config, model_config, train_config = minimal_configs
    
    with patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.instantiate"), \
         patch("cerberus.train._train") as mock_train:
        
        root_dir = tmp_path / "output_default"
        
        # No test_fold passed, should default to 0
        train_single(
            genome_config, data_config, sampler_config, model_config, train_config,
            root_dir=root_dir
        )
        
        # Verify metadata
        meta_path = root_dir / "ensemble_metadata.yaml"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0]
        
        # Verify directory
        expected_dir = root_dir / "fold_0"
        assert Path(mock_train.call_args[1]["root_dir"]) == expected_dir

def test_train_single_updates_metadata(tmp_path, minimal_configs):
    genome_config, data_config, sampler_config, model_config, train_config = minimal_configs
    
    root_dir = tmp_path / "output_incremental"
    root_dir.mkdir()
    
    # Pre-existing metadata
    with open(root_dir / "ensemble_metadata.yaml", "w") as f:
        yaml.dump({"folds": [0]}, f)
        
    with patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.instantiate"), \
         patch("cerberus.train._train"):
        
        # Train fold 1
        train_single(
            genome_config, data_config, sampler_config, model_config, train_config,
            test_fold=1,
            root_dir=root_dir
        )
        
        with open(root_dir / "ensemble_metadata.yaml") as f:
            meta = yaml.safe_load(f)
        
        # Should now have [0, 1] (order might vary but set should match)
        assert set(meta["folds"]) == {0, 1}

def test_train_multi_delegation(tmp_path, minimal_configs):
    genome_config, data_config, sampler_config, model_config, train_config = minimal_configs
    
    # k=2
    genome_config["fold_args"]["k"] = 2

    with patch("cerberus.train.train_single") as mock_train_single:
        root_dir = tmp_path / "multi_output"
        
        train_multi(
            genome_config, data_config, sampler_config, model_config, train_config,
            root_dir=root_dir
        )
        
        assert mock_train_single.call_count == 2
        
        # Check args for first call (fold 0)
        call_args_0 = mock_train_single.call_args_list[0]
        kwargs_0 = call_args_0[1]
        
        # Should pass parent root_dir
        assert Path(kwargs_0["root_dir"]) == root_dir
        assert kwargs_0["test_fold"] == 0
        
        # Check args for second call (fold 1)
        call_args_1 = mock_train_single.call_args_list[1]
        kwargs_1 = call_args_1[1]
        
        assert Path(kwargs_1["root_dir"]) == root_dir
        assert kwargs_1["test_fold"] == 1
