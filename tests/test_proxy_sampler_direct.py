import pytest

from cerberus.samplers import ProxySampler


class MockProxy(ProxySampler):
    """Trivial subclass of ProxySampler for testing purposes."""

    pass


def test_proxy_sampler_uninitialized():
    """Test that ProxySampler raises IndexError when not initialized."""
    sampler = MockProxy()

    # Test __iter__
    with pytest.raises(IndexError, match="Source sampler not initialized"):
        list(sampler)

    # Test __getitem__
    with pytest.raises(IndexError, match="Source sampler not initialized"):
        _ = sampler[0]
