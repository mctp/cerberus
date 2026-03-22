"""Coverage tests for cerberus.transform — untested code paths."""

import pytest
import torch

from cerberus.config import DataConfig
from cerberus.interval import Interval
from cerberus.transform import (
    Arcsinh,
    Bin,
    Jitter,
    Log1p,
    ReverseComplement,
    Scale,
    Sqrt,
    TargetCrop,
    create_default_transforms,
)


@pytest.fixture
def interval():
    return Interval("chr1", 1000, 2000)


@pytest.fixture
def inputs_and_targets():
    torch.manual_seed(60)
    inputs = torch.rand(4, 100) * 10
    targets = torch.rand(2, 100) * 10
    return inputs, targets


# ---------------------------------------------------------------------------
# create_default_transforms with DataConfig options
# ---------------------------------------------------------------------------


class TestCreateDefaultTransforms:
    def _make_data_config(self, **overrides) -> DataConfig:
        base: dict = {
            "inputs": {},
            "targets": {},
            "input_len": 100,
            "output_len": 100,
            "max_jitter": 0,
            "output_bin_size": 1,
            "encoding": "ACGT",
            "log_transform": False,
            "reverse_complement": False,
            "use_sequence": True,
            "target_scale": 1.0,
        }
        base.update(overrides)
        return DataConfig.model_construct(**base)

    def test_with_target_scale(self):
        dc = self._make_data_config(target_scale=2.5)
        transforms = create_default_transforms(dc)
        # Should include Scale transform
        has_scale = any(isinstance(t, Scale) for t in transforms)
        assert has_scale

    def test_with_output_bin_size(self):
        dc = self._make_data_config(output_bin_size=4)
        transforms = create_default_transforms(dc)
        bins = [t for t in transforms if isinstance(t, Bin)]
        assert len(bins) == 1
        # Must be sum pooling for correct inversion in predict_bigwig
        assert bins[0].method == "sum"

    def test_with_log_transform(self):
        dc = self._make_data_config(log_transform=True)
        transforms = create_default_transforms(dc)
        has_log1p = any(isinstance(t, Log1p) for t in transforms)
        assert has_log1p

    def test_deterministic_disables_rc_and_jitter(self):
        dc = self._make_data_config(reverse_complement=True, max_jitter=10)
        transforms = create_default_transforms(dc, deterministic=True)
        has_rc = any(isinstance(t, ReverseComplement) for t in transforms)
        assert not has_rc
        # Jitter present but with max_jitter effectively 0
        jitters = [t for t in transforms if isinstance(t, Jitter)]
        assert len(jitters) == 1

    def test_with_output_len_less_than_input(self):
        dc = self._make_data_config(input_len=200, output_len=100)
        transforms = create_default_transforms(dc)
        has_crop = any(isinstance(t, TargetCrop) for t in transforms)
        assert has_crop


# ---------------------------------------------------------------------------
# Log1p apply_to variants
# ---------------------------------------------------------------------------


class TestLog1pApplyTo:
    def test_apply_to_both(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        t = Log1p(apply_to="both")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, torch.log1p(inputs))
        torch.testing.assert_close(out_t, torch.log1p(targets))

    def test_apply_to_inputs(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        t = Log1p(apply_to="inputs")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, torch.log1p(inputs))
        torch.testing.assert_close(out_t, targets)


# ---------------------------------------------------------------------------
# Sqrt apply_to variants
# ---------------------------------------------------------------------------


class TestSqrtApplyTo:
    def test_apply_to_both(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        t = Sqrt(apply_to="both")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, torch.sqrt(inputs))
        torch.testing.assert_close(out_t, torch.sqrt(targets))


# ---------------------------------------------------------------------------
# Arcsinh apply_to variants
# ---------------------------------------------------------------------------


class TestArcsinhApplyTo:
    def test_apply_to_both(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        t = Arcsinh(apply_to="both")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, torch.arcsinh(inputs))
        torch.testing.assert_close(out_t, torch.arcsinh(targets))

    def test_apply_to_inputs(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        t = Arcsinh(apply_to="inputs")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, torch.arcsinh(inputs))
        torch.testing.assert_close(out_t, targets)


# ---------------------------------------------------------------------------
# Scale apply_to variants
# ---------------------------------------------------------------------------


class TestScaleApplyTo:
    def test_apply_to_both(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        factor = 3.0
        t = Scale(factor=factor, apply_to="both")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, inputs * factor)
        torch.testing.assert_close(out_t, targets * factor)

    def test_apply_to_inputs(self, inputs_and_targets, interval):
        inputs, targets = inputs_and_targets
        factor = 0.5
        t = Scale(factor=factor, apply_to="inputs")
        out_i, out_t, _ = t(inputs.clone(), targets.clone(), interval)
        torch.testing.assert_close(out_i, inputs * factor)
        torch.testing.assert_close(out_t, targets)


# ---------------------------------------------------------------------------
# Bin with method="avg"
# ---------------------------------------------------------------------------


class TestBinAvg:
    def test_avg_method(self, interval):
        targets = torch.arange(16, dtype=torch.float).unsqueeze(0)  # (1, 16)
        inputs = torch.randn(4, 16)
        t = Bin(bin_size=4, method="avg", apply_to="targets")
        _, out_t, _ = t(inputs, targets, interval)
        assert out_t.shape == (1, 4)
        # avg of [0,1,2,3]=1.5, [4,5,6,7]=5.5, etc.
        expected = torch.tensor([[1.5, 5.5, 9.5, 13.5]])
        torch.testing.assert_close(out_t, expected)


# ---------------------------------------------------------------------------
# TargetCrop no-op case
# ---------------------------------------------------------------------------


class TestTargetCropNoOp:
    def test_target_shorter_than_output_len(self, interval):
        """When target length <= output_len, no cropping occurs."""
        inputs = torch.randn(4, 100)
        targets = torch.randn(2, 50)
        t = TargetCrop(output_len=60)
        out_i, out_t, _ = t(inputs, targets, interval)
        # targets length 50 < output_len 60 -> no crop
        torch.testing.assert_close(out_t, targets)

    def test_target_equal_to_output_len(self, interval):
        inputs = torch.randn(4, 100)
        targets = torch.randn(2, 60)
        t = TargetCrop(output_len=60)
        _, out_t, _ = t(inputs, targets, interval)
        torch.testing.assert_close(out_t, targets)


# ---------------------------------------------------------------------------
# Jitter with max_jitter=None
# ---------------------------------------------------------------------------


class TestJitterFullRange:
    def test_max_jitter_none_uses_full_range(self):
        """max_jitter=None allows full slack range."""
        torch.manual_seed(70)
        inputs = torch.randn(4, 200)
        targets = torch.randn(2, 200)
        jitter = Jitter(input_len=100, max_jitter=None)

        starts_seen = set()
        for _ in range(100):
            iv = Interval("chr1", 0, 200)
            out_i, _, out_iv = jitter(inputs, targets, iv)
            assert out_i.shape[-1] == 100
            starts_seen.add(out_iv.start)

        # With full range (slack=100), we expect a variety of start positions
        assert len(starts_seen) > 5


# ---------------------------------------------------------------------------
# ReverseComplement with probability=0.0
# ---------------------------------------------------------------------------


class TestReverseComplementNever:
    def test_probability_zero_never_applies(self, interval):
        """With probability=0.0, the transform should never be applied."""
        torch.manual_seed(71)
        inputs = torch.randn(4, 50)
        targets = torch.randn(2, 50)
        rc = ReverseComplement(probability=0.0)

        for _ in range(20):
            iv = Interval("chr1", 0, 50, "+")
            out_i, out_t, out_iv = rc(inputs, targets, iv)
            torch.testing.assert_close(out_i, inputs)
            torch.testing.assert_close(out_t, targets)
            assert out_iv.strand == "+"
