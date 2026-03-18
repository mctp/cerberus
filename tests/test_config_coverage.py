"""Coverage tests for cerberus.config — untested code paths."""
import pytest
from pathlib import Path
from typing import cast, Any
from cerberus.config import (
    _resolve_path,
    _sanitize_config,
    import_class,
    validate_model_config,
    ModelConfig,
)


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:

    def test_existing_path_returned(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.touch()
        result = _resolve_path(f)
        assert result == f

    def test_absolute_path_suffix_matching(self, tmp_path):
        """When path is absolute and not found, try suffix matching in search_paths."""
        # Create file at search_path/sub/file.txt
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "file.txt"
        f.touch()

        # Query with a non-existent absolute path ending in sub/file.txt
        fake_abs = Path("/nonexistent/root/sub/file.txt")
        result = _resolve_path(fake_abs, search_paths=[tmp_path])
        assert result.exists()
        assert result.resolve() == f.resolve()

    def test_relative_path_in_search_path(self, tmp_path):
        f = tmp_path / "data.bin"
        f.touch()
        result = _resolve_path(Path("data.bin"), search_paths=[tmp_path])
        assert result.exists()

    def test_not_found_returns_original(self):
        p = Path("/surely/nonexistent/path.xyz")
        result = _resolve_path(p)
        assert result == p


# ---------------------------------------------------------------------------
# _sanitize_config
# ---------------------------------------------------------------------------

class TestSanitizeConfig:

    def test_nested_dicts(self):
        config = {"a": {"b": Path("/some/path")}}
        result = _sanitize_config(config)
        assert result == {"a": {"b": "/some/path"}}

    def test_lists(self):
        config = [Path("/a"), Path("/b")]
        result = _sanitize_config(config)
        assert result == ["/a", "/b"]

    def test_path_objects(self):
        result = _sanitize_config(Path("/my/path"))
        assert result == "/my/path"

    def test_scalars_unchanged(self):
        assert _sanitize_config(42) == 42
        assert _sanitize_config("hello") == "hello"
        assert _sanitize_config(True) is True

    def test_mixed_nested(self):
        config = {
            "paths": [Path("/a"), {"nested": Path("/b")}],
            "value": 123,
        }
        result = _sanitize_config(config)
        assert result == {
            "paths": ["/a", {"nested": "/b"}],
            "value": 123,
        }


# ---------------------------------------------------------------------------
# import_class
# ---------------------------------------------------------------------------

class TestImportClass:

    def test_valid_import(self):
        cls = import_class("cerberus.loss.MSEMultinomialLoss")
        from cerberus.loss import MSEMultinomialLoss
        assert cls is MSEMultinomialLoss

    def test_invalid_module_raises(self):
        with pytest.raises(ImportError, match="Could not import"):
            import_class("nonexistent.module.ClassName")

    def test_invalid_class_in_valid_module_raises(self):
        with pytest.raises(ImportError, match="Could not import"):
            import_class("cerberus.loss.NonExistentClass")

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="must be a string"):
            import_class(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_model_config
# ---------------------------------------------------------------------------

class TestValidateModelConfig:

    def _base_config(self) -> ModelConfig:
        return cast(ModelConfig, {
            "name": "test_model",
            "model_cls": "cerberus.models.bpnet.BPNet",
            "loss_cls": "cerberus.loss.MSEMultinomialLoss",
            "loss_args": {},
            "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
            "metrics_args": {},
            "model_args": {},
            "pretrained": [],
        })

    def test_valid_config(self):
        config = self._base_config()
        result = validate_model_config(config)
        assert result["name"] == "test_model"

    def test_invalid_output_type(self):
        config = self._base_config()
        config["model_args"]["output_type"] = "invalid_type"
        with pytest.raises(ValueError, match="output_type"):
            validate_model_config(config)

    def test_valid_output_type_signal(self):
        config = self._base_config()
        config["model_args"]["output_type"] = "signal"
        result = validate_model_config(config)
        assert result["model_args"]["output_type"] == "signal"

    def test_valid_output_type_decoupled(self):
        config = self._base_config()
        config["model_args"]["output_type"] = "decoupled"
        result = validate_model_config(config)
        assert result["model_args"]["output_type"] == "decoupled"

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_model_config(cast(ModelConfig, {"name": "x"}))
