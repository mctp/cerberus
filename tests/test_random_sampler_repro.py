import unittest

from cerberus.samplers import MultiSampler, RandomSampler, ScaledSampler


class TestRandomSamplerInconsistencies(unittest.TestCase):
    def test_random_sampler_resample_none_behavior(self):
        """
        Demonstrates that calling resample(None) on a seeded RandomSampler
        advances the RNG state, producing new intervals (Next Epoch behavior).
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 5
        seed = 42

        # 1. Initialize with seed
        s = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=seed)
        intervals_epoch1 = list(s)

        # 2. Resample with None
        s.resample(None)
        intervals_epoch2 = list(s)

        # Verify inequality (New behavior)
        # We expect the sampler to generate NEW intervals for the next epoch
        starts1 = [i.start for i in intervals_epoch1]
        starts2 = [i.start for i in intervals_epoch2]
        self.assertNotEqual(
            starts1,
            starts2,
            "Intervals identical after resample(None), expected new intervals",
        )

    def test_multisampler_resample_propagation(self):
        """
        Demonstrates that MultiSampler.resample(None) propagates None,
        allowing child RandomSampler to advance RNG state.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 5
        seed = 42

        rs = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=seed)
        ms = MultiSampler([rs], chrom_sizes, folds=[], exclude_intervals={}, seed=seed)

        intervals_1 = list(ms)

        # Resample MultiSampler with None
        ms.resample(None)
        intervals_2 = list(ms)

        # We expect intervals to be DIFFERENT now (next epoch behavior)
        starts1 = [i.start for i in intervals_1]
        starts2 = [i.start for i in intervals_2]
        self.assertNotEqual(
            starts1, starts2, "MultiSampler intervals identical after resample(None)"
        )

    def test_scaled_sampler_resample_none_behavior(self):
        """
        Demonstrates that ScaledSampler also suffers from the same reset issue.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 10
        seed = 42

        # Base sampler
        rs = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=seed)

        # Scaled sampler (subsample to 5)
        ss = ScaledSampler(rs, num_samples=5, seed=seed)

        indices_1 = ss._indices

        # Resample with None
        ss.resample(None)

        indices_2 = ss._indices

        # If not fixed, indices will be identical (reset)
        # We expect different indices if we want "Next Epoch" behavior
        self.assertNotEqual(
            indices_1, indices_2, "ScaledSampler indices identical after resample(None)"
        )


if __name__ == "__main__":
    unittest.main()
