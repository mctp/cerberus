import pytest
import yaml
from pathlib import Path
from cerberus.config import parse_hparams_config

def test_parse_hparams_config_success():
    # Use existing hparams file
    hparams_path = Path("tests/data/models/chip_ar_mdapca2b-geminet/multi-fold/fold_0/lightning_logs/version_0/hparams.yaml")
    
    if not hparams_path.exists():
        pytest.skip(f"Test hparams file not found at {hparams_path}")
        
    config = parse_hparams_config(hparams_path)
    
    assert isinstance(config, dict)
    # Check keys
    assert "train_config" in config
    assert "genome_config" in config
    assert "data_config" in config
    assert "sampler_config" in config
    assert "model_config" in config
    
    # Check some values to ensure they are parsed correctly
    assert config["data_config"]["input_len"] == 2048
    assert isinstance(config["genome_config"]["fasta_path"], Path)
    assert config["genome_config"]["name"] == "hg38"
    assert config["model_config"]["name"] == "GemiNet"
    # Check class string
    assert config["model_config"]["model_cls"] == "cerberus.models.geminet.GeminiNet"
    
def test_parse_hparams_config_not_found():
    with pytest.raises(FileNotFoundError):
        parse_hparams_config("non_existent_hparams.yaml")

def test_parse_hparams_config_missing_sections(tmp_path):
    p = tmp_path / "missing_sections.yaml"
    data = {"train_config": {}}
    with open(p, 'w') as f:
        yaml.dump(data, f)
        
    with pytest.raises(ValueError, match="missing required sections"):
        parse_hparams_config(p)
