from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cerberus.config import CerberusConfig
from cerberus.model_ensemble import parse_hparams_config


def test_parse_hparams_config_success():
    # Use existing hparams file
    hparams_path = Path("tests/data/fixtures/hparams_bpnet.yaml")

    config = parse_hparams_config(hparams_path)

    assert isinstance(config, CerberusConfig)

    # Check some values to ensure they are parsed correctly
    assert config.data_config.input_len == 2114
    assert isinstance(config.genome_config.fasta_path, Path)
    assert config.genome_config.name == "hg38"
    assert config.model_config_.name == "BPNet"
    # Check class string
    assert config.model_config_.model_cls == "cerberus.models.bpnet.BPNet"


def test_parse_hparams_config_not_found():
    with pytest.raises(FileNotFoundError):
        parse_hparams_config("non_existent_hparams.yaml")


def test_parse_hparams_config_missing_sections(tmp_path):
    p = tmp_path / "missing_sections.yaml"
    data = {"train_config": {}}
    with open(p, "w") as f:
        yaml.dump(data, f)

    with pytest.raises(ValidationError):
        parse_hparams_config(p)
