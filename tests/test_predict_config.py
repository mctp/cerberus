import pytest
from cerberus.config import validate_predict_config, PredictConfig

def test_validate_predict_config_valid(tmp_path):
    config: PredictConfig = {
        "stride": 50,
        "use_folds": ["test", "val"],
        "aggregation": "model"
    }
    validated = validate_predict_config(config)
    assert validated == config

def test_validate_predict_config_minimal():
    config = {
        "stride": 50,
        "aggregation": "interval+model"
    }
    validated = validate_predict_config(config) # type: ignore
    assert validated["use_folds"] == ["test", "val"]

def test_validate_predict_config_invalid_stride():
    config = {
        "stride": -10,
        "aggregation": "model"
    }
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        validate_predict_config(config) # type: ignore

def test_validate_predict_config_invalid_folds():
    config = {
        "stride": 50,
        "aggregation": "model",
        "use_folds": ["train", "invalid"]
    }
    with pytest.raises(ValueError, match="use_folds must be subset of"):
        validate_predict_config(config) # type: ignore

def test_validate_predict_config_invalid_aggregation():
    config = {
        "stride": 50,
        "aggregation": "mean" # invalid now
    }
    with pytest.raises(ValueError, match=r"aggregation must be 'model' or 'interval\+model'"):
        validate_predict_config(config) # type: ignore
