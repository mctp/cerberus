import pytest
import yaml
from pathlib import Path
from cerberus.config import parse_hparams_config

def test_parse_hparams_config_generated_success(tmp_path):
    # create a mock hparams.yaml content with strings instead of paths
    # (Paths are sanitized to strings in hparams.yaml usually)
    
    # Create dummy files for validation
    (tmp_path / "genome.fa").touch()
    (tmp_path / "peaks.bed").touch()
    (tmp_path / "input.bw").touch()
    (tmp_path / "target.bw").touch()
    
    config = {
        "train_config": {
            "batch_size": 32,
            "max_epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "patience": 5,
            "optimizer": "adamw",
            "filter_bias_and_bn": True,
            "scheduler_type": "default",
            "scheduler_args": {},
            "adam_eps": 1e-8,
            "gradient_clip_val": None,
        },
        "genome_config": {
            "name": "test_genome",
            "fasta_path": str(tmp_path / "genome.fa"),
            "exclude_intervals": {"blacklist": str(tmp_path / "peaks.bed")},
            "allowed_chroms": ["chr1"],
            "chrom_sizes": {"chr1": 1000},
            "fold_type": "random",
            "fold_args": {"k": 5}
        },
        "data_config": {
            "inputs": {"seq": str(tmp_path / "input.bw")},
            "targets": {"out": str(tmp_path / "target.bw")},
            "input_len": 100,
            "output_len": 100,
            "max_jitter": 0,
            "output_bin_size": 1,
            "encoding": "ACGT",
            "log_transform": False,
            "reverse_complement": False,
        "target_scale": 1.0,
            "use_sequence": True
        },
        "sampler_config": {
            "sampler_type": "interval",
            "padded_size": 100,
            "sampler_args": {"intervals_path": str(tmp_path / "peaks.bed")}
        },
        "model_config": {
            "name": "test_model",
            "model_cls": "cerberus.models.bpnet.BPNet",
            "loss_cls": "cerberus.loss.BPNetLoss",
            "metrics_cls": "cerberus.metrics.BPNetMetrics",
            "loss_args": {},
            "metrics_args": {},
            "model_args": {
                "input_channels": ["seq"],
                "output_channels": ["out"],
                "output_type": "signal"
            }
        }
    }
    
    hparams_path = tmp_path / "hparams.yaml"
    with open(hparams_path, "w") as f:
        yaml.dump(config, f)
        
    # Test parsing
    parsed = parse_hparams_config(hparams_path)
    
    assert parsed["train_config"]["batch_size"] == 32
    # Verify Path conversion
    assert isinstance(parsed["genome_config"]["fasta_path"], Path)
    assert parsed["genome_config"]["fasta_path"].name == "genome.fa"
