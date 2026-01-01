import pytest
from cerberus.config import validate_predict_config, PredictConfig

def test_validate_predict_config_valid(tmp_path):
    p = tmp_path / "regions.bed"
    p.touch()
    
    config: PredictConfig = {
        "stride": 50,
        "intervals": ["chr1:100-200"],
        "intervals_paths": [p],
        "use_folds": ["test", "val"],
        "aggregation": "mean"
    }
    validated = validate_predict_config(config)
    assert validated == config

def test_validate_predict_config_minimal():
    config = {
        "stride": 50,
        "aggregation": "median"
    }
    validated = validate_predict_config(config) # type: ignore
    assert validated["intervals"] == []
    assert validated["intervals_paths"] == []
    assert validated["use_folds"] == ["test", "val"]

def test_validate_predict_config_invalid_stride():
    config = {
        "stride": -10,
        "aggregation": "mean"
    }
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        validate_predict_config(config) # type: ignore

def test_validate_predict_config_invalid_folds():
    config = {
        "stride": 50,
        "aggregation": "mean",
        "use_folds": ["train", "invalid"]
    }
    with pytest.raises(ValueError, match="use_folds must be subset of"):
        validate_predict_config(config) # type: ignore

def test_validate_predict_config_invalid_aggregation():
    config = {
        "stride": 50,
        "aggregation": "sum" # invalid
    }
    with pytest.raises(ValueError, match="aggregation must be 'mean' or 'median'"):
        validate_predict_config(config) # type: ignore

def test_validate_predict_config_path_validation(tmp_path):
    # Missing file should raise FileNotFoundError
    config = {
        "stride": 50,
        "aggregation": "mean",
        "intervals_paths": [tmp_path / "missing.bed"]
    }
    with pytest.raises(FileNotFoundError):
        validate_predict_config(config) # type: ignore
    
    # Existing file should pass
    p = tmp_path / "exists.bed"
    p.touch()
    config["intervals_paths"] = [p]
    validate_predict_config(config) # type: ignore
