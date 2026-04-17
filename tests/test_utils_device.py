from unittest.mock import patch

import torch

from cerberus.utils import resolve_device


def test_resolve_device_uses_explicit_device_string():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_resolve_device_none_and_auto_share_auto_detection():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device() == torch.device("cpu")
        assert resolve_device(None) == torch.device("cpu")
        assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_prefers_cuda_over_mps():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device() == torch.device("cuda")


def test_resolve_device_falls_back_to_mps_when_cuda_unavailable():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device() == torch.device("mps")
