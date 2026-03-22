"""Tests for cerberus.pretrained — pretrained weight loading utilities."""

import pytest
import torch
import torch.nn as nn

from cerberus.config import PretrainedConfig
from cerberus.pretrained import (
    _extract_prefix,
    _unwrap_compiled,
    load_pretrained_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class ParentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias_model = SubModule()
        self.signal_model = SubModule()


# ---------------------------------------------------------------------------
# _unwrap_compiled
# ---------------------------------------------------------------------------


class TestUnwrapCompiled:
    def test_returns_same_if_not_compiled(self):
        model = nn.Linear(4, 4)
        assert _unwrap_compiled(model) is model

    def test_unwraps_compiled_model(self):
        model = nn.Linear(4, 4)
        # Simulate torch.compile by attaching _orig_mod
        wrapper = nn.Module()
        wrapper._orig_mod = model  # type: ignore[assignment]
        assert _unwrap_compiled(wrapper) is model


# ---------------------------------------------------------------------------
# _extract_prefix
# ---------------------------------------------------------------------------


class TestExtractPrefix:
    def test_extracts_matching_keys(self):
        state_dict = {
            "bias_model.linear.weight": torch.zeros(4, 4),
            "bias_model.linear.bias": torch.zeros(4),
            "signal_model.linear.weight": torch.ones(4, 4),
            "signal_model.linear.bias": torch.ones(4),
        }
        extracted = _extract_prefix(state_dict, "bias_model")
        assert set(extracted.keys()) == {"linear.weight", "linear.bias"}
        assert torch.equal(extracted["linear.weight"], torch.zeros(4, 4))

    def test_strips_prefix_correctly(self):
        state_dict = {"encoder.layer.0.weight": torch.randn(3, 3)}
        extracted = _extract_prefix(state_dict, "encoder")
        assert "layer.0.weight" in extracted

    def test_raises_on_no_match(self):
        state_dict = {"bias_model.weight": torch.zeros(4)}
        with pytest.raises(ValueError, match="No keys found with prefix"):
            _extract_prefix(state_dict, "nonexistent")

    def test_error_shows_available_keys(self):
        state_dict = {"alpha.w": torch.zeros(1), "beta.w": torch.zeros(1)}
        with pytest.raises(ValueError, match="alpha.w"):
            _extract_prefix(state_dict, "gamma")

    def test_does_not_match_partial_prefix(self):
        """'bias' should not match 'bias_model.weight'."""
        state_dict = {"bias_model.weight": torch.zeros(4)}
        with pytest.raises(ValueError, match="No keys found"):
            _extract_prefix(state_dict, "bias")


# ---------------------------------------------------------------------------
# load_pretrained_weights — full model load
# ---------------------------------------------------------------------------


class TestLoadPretrainedFull:
    def test_load_full_model(self, tmp_path):
        """Load all weights into the whole model (source=None, target=None)."""
        model = ParentModel()
        # Save current state
        weights_path = tmp_path / "full.pt"
        torch.save(model.state_dict(), weights_path)

        # Create a fresh model (different weights)
        model2 = ParentModel()
        assert not torch.equal(
            model.bias_model.linear.weight, model2.bias_model.linear.weight
        )

        cfg = PretrainedConfig(
            weights_path=str(weights_path), source=None, target=None, freeze=False,
        )
        load_pretrained_weights(model2, [cfg])

        # Weights should now match
        assert torch.equal(
            model.bias_model.linear.weight, model2.bias_model.linear.weight
        )
        assert torch.equal(
            model.signal_model.linear.weight, model2.signal_model.linear.weight
        )

    def test_loaded_params_are_trainable_when_not_frozen(self, tmp_path):
        model = ParentModel()
        weights_path = tmp_path / "full.pt"
        torch.save(model.state_dict(), weights_path)

        cfg = PretrainedConfig(
            weights_path=str(weights_path), source=None, target=None, freeze=False,
        )
        load_pretrained_weights(model, [cfg])

        for p in model.parameters():
            assert p.requires_grad is True


# ---------------------------------------------------------------------------
# load_pretrained_weights — sub-module load (target)
# ---------------------------------------------------------------------------


class TestLoadPretrainedTarget:
    def test_load_into_submodule(self, tmp_path):
        """Load standalone weights into a named sub-module."""
        sub = SubModule()
        weights_path = tmp_path / "sub.pt"
        torch.save(sub.state_dict(), weights_path)

        # Fresh parent — bias_model should differ
        parent = ParentModel()

        cfg = PretrainedConfig(
            weights_path=str(weights_path),
            source=None,
            target="bias_model",
            freeze=False,
        )
        load_pretrained_weights(parent, [cfg])

        assert torch.equal(parent.bias_model.linear.weight, sub.linear.weight)
        assert torch.equal(parent.bias_model.linear.bias, sub.linear.bias)

    def test_target_does_not_affect_other_submodule(self, tmp_path):
        sub = SubModule()
        weights_path = tmp_path / "sub.pt"
        torch.save(sub.state_dict(), weights_path)

        parent = ParentModel()
        original_signal_weight = parent.signal_model.linear.weight.clone()

        cfg = PretrainedConfig(
            weights_path=str(weights_path),
            source=None,
            target="bias_model",
            freeze=False,
        )
        load_pretrained_weights(parent, [cfg])

        # signal_model should be unchanged
        assert torch.equal(parent.signal_model.linear.weight, original_signal_weight)


# ---------------------------------------------------------------------------
# load_pretrained_weights — source prefix extraction
# ---------------------------------------------------------------------------


class TestLoadPretrainedSource:
    def test_extract_submodule_from_full_checkpoint(self, tmp_path):
        """Extract bias_model keys from a full checkpoint and load into bias_model."""
        full_model = ParentModel()
        weights_path = tmp_path / "full.pt"
        torch.save(full_model.state_dict(), weights_path)

        # Fresh parent
        parent = ParentModel()

        cfg = PretrainedConfig(
            weights_path=str(weights_path),
            source="bias_model",
            target="bias_model",
            freeze=False,
        )
        load_pretrained_weights(parent, [cfg])

        assert torch.equal(
            parent.bias_model.linear.weight, full_model.bias_model.linear.weight
        )

    def test_source_prefix_not_found_raises(self, tmp_path):
        model = ParentModel()
        weights_path = tmp_path / "full.pt"
        torch.save(model.state_dict(), weights_path)

        cfg = PretrainedConfig(
            weights_path=str(weights_path),
            source="nonexistent_module",
            target=None,
            freeze=False,
        )
        with pytest.raises(ValueError, match="No keys found with prefix"):
            load_pretrained_weights(model, [cfg])


# ---------------------------------------------------------------------------
# load_pretrained_weights — freeze
# ---------------------------------------------------------------------------


class TestLoadPretrainedFreeze:
    def test_freeze_all_params(self, tmp_path):
        model = ParentModel()
        weights_path = tmp_path / "full.pt"
        torch.save(model.state_dict(), weights_path)

        cfg = PretrainedConfig(
            weights_path=str(weights_path), source=None, target=None, freeze=True,
        )
        load_pretrained_weights(model, [cfg])

        for p in model.parameters():
            assert p.requires_grad is False

    def test_freeze_only_target_submodule(self, tmp_path):
        """Freezing target='bias_model' should not freeze signal_model."""
        sub = SubModule()
        weights_path = tmp_path / "sub.pt"
        torch.save(sub.state_dict(), weights_path)

        parent = ParentModel()

        cfg = PretrainedConfig(
            weights_path=str(weights_path),
            source=None,
            target="bias_model",
            freeze=True,
        )
        load_pretrained_weights(parent, [cfg])

        # bias_model frozen
        for p in parent.bias_model.parameters():
            assert p.requires_grad is False

        # signal_model still trainable
        for p in parent.signal_model.parameters():
            assert p.requires_grad is True


# ---------------------------------------------------------------------------
# load_pretrained_weights — multiple configs
# ---------------------------------------------------------------------------


class TestLoadPretrainedMultiple:
    def test_apply_multiple_configs(self, tmp_path):
        """Apply two pretrained configs in sequence."""
        bias_sub = SubModule()
        signal_sub = SubModule()

        bias_path = tmp_path / "bias.pt"
        signal_path = tmp_path / "signal.pt"
        torch.save(bias_sub.state_dict(), bias_path)
        torch.save(signal_sub.state_dict(), signal_path)

        parent = ParentModel()

        configs = [
            PretrainedConfig(
                weights_path=str(bias_path),
                source=None,
                target="bias_model",
                freeze=True,
            ),
            PretrainedConfig(
                weights_path=str(signal_path),
                source=None,
                target="signal_model",
                freeze=False,
            ),
        ]
        load_pretrained_weights(parent, configs)

        assert torch.equal(parent.bias_model.linear.weight, bias_sub.linear.weight)
        assert torch.equal(parent.signal_model.linear.weight, signal_sub.linear.weight)

        # bias frozen, signal trainable
        for p in parent.bias_model.parameters():
            assert p.requires_grad is False
        for p in parent.signal_model.parameters():
            assert p.requires_grad is True

    def test_empty_pretrained_list_is_noop(self):
        model = ParentModel()
        original = model.bias_model.linear.weight.clone()
        load_pretrained_weights(model, [])
        assert torch.equal(model.bias_model.linear.weight, original)


# ---------------------------------------------------------------------------
# load_pretrained_weights — strict loading
# ---------------------------------------------------------------------------


class TestLoadPretrainedStrict:
    def test_strict_rejects_shape_mismatch(self, tmp_path):
        """strict=True should reject mismatched weight shapes."""
        # Save weights from a Linear(4, 4)
        small = nn.Linear(4, 4)
        weights_path = tmp_path / "small.pt"
        torch.save({"linear.weight": small.weight, "linear.bias": small.bias}, weights_path)

        # Try to load into a Linear(8, 8)
        class BigModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

        model = BigModule()
        cfg = PretrainedConfig(
            weights_path=str(weights_path), source=None, target=None, freeze=False,
        )
        with pytest.raises(RuntimeError):
            load_pretrained_weights(model, [cfg])

    def test_strict_rejects_missing_keys(self, tmp_path):
        """strict=True should reject when checkpoint has fewer keys than model."""
        weights_path = tmp_path / "partial.pt"
        torch.save({"linear.weight": torch.zeros(4, 4)}, weights_path)

        model = SubModule()  # expects linear.weight AND linear.bias
        cfg = PretrainedConfig(
            weights_path=str(weights_path), source=None, target=None, freeze=False,
        )
        with pytest.raises(RuntimeError):
            load_pretrained_weights(model, [cfg])
