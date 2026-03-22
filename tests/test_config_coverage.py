"""Coverage tests for cerberus.config — untested code paths."""
import pytest
from pathlib import Path
from pydantic import ValidationError
from cerberus.config import (
    _resolve_path,
    import_class,
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
# model_dump(mode="json") — replacement for _sanitize_config
# ---------------------------------------------------------------------------

class TestModelDumpJson:

    def test_path_objects_serialized_to_strings(self, tmp_path):
        """model_dump(mode='json') converts Path objects to strings."""
        cons = tmp_path / "cons.bw"
        cons.touch()
        from cerberus.config import DataConfig
        cfg = DataConfig(
            inputs={"cons": cons},
            targets={},
            input_len=100,
            output_len=50,
            max_jitter=0,
            output_bin_size=1,
            encoding="ACGT",
            log_transform=False,
            reverse_complement=False,
            use_sequence=True,
            target_scale=1.0,
        )
        dumped = cfg.model_dump(mode="json")
        assert isinstance(dumped["inputs"]["cons"], str)

    def test_scalars_unchanged_in_dump(self):
        """Scalars pass through model_dump(mode='json') without transformation."""
        cfg = ModelConfig(
            name="test_model",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            metrics_args={},
            model_args={},
            pretrained=[],
        )
        dumped = cfg.model_dump(mode="json")
        assert dumped["name"] == "test_model"
        assert dumped["count_pseudocount"] == 0.0
        assert isinstance(dumped["count_pseudocount"], float)

    def test_nested_dicts_serialized(self):
        """Nested model_args dicts are serialized correctly."""
        cfg = ModelConfig(
            name="test",
            model_cls="cerberus.models.bpnet.BPNet",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={"alpha": 1.0},
            metrics_cls="cerberus.metrics.DefaultMetricCollection",
            metrics_args={},
            model_args={"hidden": 64},
            pretrained=[],
        )
        dumped = cfg.model_dump(mode="json")
        assert dumped["loss_args"]["alpha"] == 1.0
        assert dumped["model_args"]["hidden"] == 64


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
        with pytest.raises(ImportError):
            import_class(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelConfig Pydantic validation
# ---------------------------------------------------------------------------

class TestModelConfigValidation:

    def _base_kwargs(self) -> dict:
        return {
            "name": "test_model",
            "model_cls": "cerberus.models.bpnet.BPNet",
            "loss_cls": "cerberus.loss.MSEMultinomialLoss",
            "loss_args": {},
            "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
            "metrics_args": {},
            "model_args": {},
            "pretrained": [],
        }

    def test_valid_config(self):
        cfg = ModelConfig(**self._base_kwargs())
        assert cfg.name == "test_model"

    def test_invalid_output_type(self):
        kw = self._base_kwargs()
        kw["model_args"]["output_type"] = "invalid_type"
        with pytest.raises(ValidationError, match="output_type"):
            ModelConfig(**kw)

    def test_valid_output_type_signal(self):
        kw = self._base_kwargs()
        kw["model_args"]["output_type"] = "signal"
        cfg = ModelConfig(**kw)
        assert cfg.model_args["output_type"] == "signal"

    def test_valid_output_type_decoupled(self):
        kw = self._base_kwargs()
        kw["model_args"]["output_type"] = "decoupled"
        cfg = ModelConfig(**kw)
        assert cfg.model_args["output_type"] == "decoupled"

    def test_missing_keys_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(name="x")  # type: ignore[call-arg]
