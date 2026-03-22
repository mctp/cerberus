"""Test that Jitter does not permanently mutate sampler intervals.

When dataset.__getitem__ passes the sampler's stored Interval reference
directly to transforms, Jitter mutates it in place (shrinking from
padded_size to input_len). On any subsequent access to the same index,
slack=0 and Jitter becomes a no-op — jitter augmentation is silently
lost after the first epoch.

This test verifies that repeated access to the same index preserves
the original interval width, allowing Jitter to apply fresh randomness
each time.
"""

import copy

import torch

from cerberus.interval import Interval
from cerberus.samplers import ListSampler
from cerberus.transform import Compose, Jitter


class FakeExtractor:
    """Returns a constant tensor for any interval."""

    def extract(self, interval: Interval) -> torch.Tensor:
        width = interval.end - interval.start
        return torch.ones(1, width)


class MinimalDataset:
    """Stripped-down dataset that reproduces the __getitem__ → Jitter path.

    Mirrors the real CerberusDataset.__getitem__ which copies the interval
    before passing it through transforms.
    """

    def __init__(self, sampler, input_len, max_jitter):
        self.sampler = sampler
        self.input_len = input_len
        self.extractor = FakeExtractor()
        self.transforms = Compose([Jitter(input_len=input_len, max_jitter=max_jitter)])

    def __getitem__(self, idx):
        interval = copy.copy(self.sampler[idx])
        inputs = self.extractor.extract(interval)
        targets = self.extractor.extract(interval)
        inputs, targets, interval = self.transforms(inputs, targets, interval)
        return inputs, targets, interval


def test_jitter_does_not_mutate_sampler_interval():
    """Accessing the same index twice should allow Jitter both times.

    With padded_size=120 and input_len=100, the first Jitter call crops
    to 100 and mutates the interval. If the interval is not copied, the
    second call sees slack=0 and Jitter is a no-op — the interval is
    stuck at input_len forever.
    """
    padded_size = 120
    input_len = 100
    max_jitter = 10

    intervals = [Interval("chr1", 1000, 1000 + padded_size, "+")]
    sampler = ListSampler(intervals=intervals)

    ds = MinimalDataset(sampler, input_len, max_jitter)

    # First access — Jitter should crop from 120 to 100
    _, _, iv1 = ds[0]
    assert len(iv1) == input_len

    # Check sampler's stored interval: it should still be padded_size
    stored = sampler[0]
    assert len(stored) == padded_size, (
        f"Sampler interval was mutated from {padded_size} to {len(stored)} "
        f"after first __getitem__. Jitter will be a no-op on subsequent accesses."
    )

    # Second access — Jitter should still have slack to work with
    _, _, iv2 = ds[0]
    assert len(iv2) == input_len


def test_jitter_produces_different_offsets_across_accesses():
    """With enough slack and randomness, two accesses to the same index
    should (usually) produce different jitter offsets.

    We use a large slack to make collisions extremely unlikely.
    """
    padded_size = 200
    input_len = 100
    max_jitter = 50  # slack=100, uniform over [0, 100]

    intervals = [Interval("chr1", 0, padded_size, "+")]
    sampler = ListSampler(intervals=intervals)

    ds = MinimalDataset(sampler, input_len, max_jitter)

    # Collect start positions from multiple accesses
    starts = set()
    for _ in range(20):
        _, _, iv = ds[0]
        starts.add(iv.start)

    # With the bug (Jitter mutated in place), all starts after the first
    # would be identical. With the fix, we should see variety.
    assert len(starts) > 1, (
        f"All 20 accesses produced the same start position {starts}. "
        "Jitter is likely a no-op after the first access."
    )
