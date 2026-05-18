import logging
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
    # ``patch.dict(..., clear=True)`` isolates the test from a process-level
    # ``LOCAL_RANK`` that a DDP-aware test runner might leave set.
    with (
        patch.dict("os.environ", {}, clear=True),
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


def test_resolve_device_uses_local_rank_for_auto_cuda():
    with (
        patch.dict("os.environ", {"LOCAL_RANK": "1"}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device() == torch.device("cuda:1")
        assert resolve_device("auto") == torch.device("cuda:1")


def test_resolve_device_explicit_cuda_ignores_local_rank():
    # Explicit device strings bypass auto-detection entirely.
    with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
        assert resolve_device("cuda") == torch.device("cuda")
        assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_resolve_device_warns_on_out_of_range_local_rank(caplog):
    with (
        patch.dict("os.environ", {"LOCAL_RANK": "3"}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.backends.mps.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="cerberus.utils"),
    ):
        assert resolve_device() == torch.device("cuda")
    assert any("out of range" in m for m in caplog.messages)


def test_resolve_device_warns_on_non_integer_local_rank(caplog):
    with (
        patch.dict("os.environ", {"LOCAL_RANK": "not-a-number"}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.backends.mps.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="cerberus.utils"),
    ):
        assert resolve_device() == torch.device("cuda")
    assert any("not an integer" in m for m in caplog.messages)
