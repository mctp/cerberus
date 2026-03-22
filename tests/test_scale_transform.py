import pytest
import torch

from cerberus.config import DataConfig
from cerberus.interval import Interval
from cerberus.transform import Compose, Scale, create_default_transforms


@pytest.fixture
def dummy_interval():
    return Interval("chr1", 0, 1000)


# ---------------------------------------------------------------------------
# Scale transform
# ---------------------------------------------------------------------------

class TestScale:
    def test_scale_targets(self, dummy_interval):
        inputs = torch.ones(1, 10)
        targets = torch.ones(1, 10) * 2.0
        scale = Scale(factor=5.0, apply_to="targets")
        out_in, out_tgt, _ = scale(inputs, targets, dummy_interval)
        assert torch.allclose(out_in, inputs)
        assert torch.allclose(out_tgt, torch.ones(1, 10) * 10.0)

    def test_scale_inputs(self, dummy_interval):
        inputs = torch.ones(1, 10) * 3.0
        targets = torch.ones(1, 10) * 2.0
        scale = Scale(factor=2.0, apply_to="inputs")
        out_in, out_tgt, _ = scale(inputs, targets, dummy_interval)
        assert torch.allclose(out_in, torch.ones(1, 10) * 6.0)
        assert torch.allclose(out_tgt, targets)

    def test_scale_both(self, dummy_interval):
        inputs = torch.ones(1, 10)
        targets = torch.ones(1, 10)
        scale = Scale(factor=3.0, apply_to="both")
        out_in, out_tgt, _ = scale(inputs, targets, dummy_interval)
        assert torch.allclose(out_in, torch.ones(1, 10) * 3.0)
        assert torch.allclose(out_tgt, torch.ones(1, 10) * 3.0)

    def test_scale_factor_one_is_noop(self, dummy_interval):
        inputs = torch.randn(1, 10)
        targets = torch.randn(1, 10)
        scale = Scale(factor=1.0, apply_to="targets")
        out_in, out_tgt, _ = scale(inputs, targets, dummy_interval)
        assert torch.allclose(out_in, inputs)
        assert torch.allclose(out_tgt, targets)

    def test_scale_preserves_interval(self, dummy_interval):
        scale = Scale(factor=10.0, apply_to="targets")
        _, _, out_int = scale(torch.zeros(1, 10), torch.zeros(1, 10), dummy_interval)
        assert out_int.chrom == "chr1"
        assert out_int.start == 0
        assert out_int.end == 1000


# ---------------------------------------------------------------------------
# create_default_transforms with target_scale
# ---------------------------------------------------------------------------

def _make_data_config(**overrides) -> DataConfig:
    base = {
        "inputs": {},
        "targets": {},
        "input_len": 100,
        "output_len": 50,
        "output_bin_size": 1,
        "encoding": "ACGT",
        "max_jitter": 0,
        "log_transform": False,
        "reverse_complement": False,
        "target_scale": 1.0,
        "use_sequence": True,
    }
    base.update(overrides)
    return DataConfig.model_construct(**base)


class TestCreateDefaultTransformsTargetScale:
    def test_no_scale_when_1(self):
        """target_scale=1.0 should NOT add a Scale transform."""
        config = _make_data_config(target_scale=1.0)
        transform_list = create_default_transforms(config)
        scale_transforms = [t for t in transform_list if isinstance(t, Scale)]
        assert len(scale_transforms) == 0

    def test_scale_added_when_not_1(self):
        """target_scale != 1.0 should add a Scale transform."""
        config = _make_data_config(target_scale=1000.0)
        transform_list = create_default_transforms(config)
        scale_transforms = [t for t in transform_list if isinstance(t, Scale)]
        assert len(scale_transforms) == 1
        assert scale_transforms[0].factor == 1000.0

    def test_missing_target_scale_raises(self):
        """Accessing missing target_scale should raise AttributeError, not silently default."""
        config = _make_data_config()
        # Remove target_scale from the model_construct'd object
        object.__delattr__(config, "target_scale")
        with pytest.raises(AttributeError):
            create_default_transforms(config)

    def test_scale_applied_to_targets(self, dummy_interval):
        """End-to-end: verify Scale transform actually scales the targets."""
        config = _make_data_config(
            input_len=10, output_len=10, target_scale=100.0, max_jitter=0
        )
        transform_list = create_default_transforms(config)
        composed = Compose(transform_list)
        inputs = torch.ones(1, 10)
        targets = torch.ones(1, 10) * 0.5
        _, out_tgt, _ = composed(inputs, targets, dummy_interval)
        assert torch.allclose(out_tgt, torch.ones(1, 10) * 50.0)
