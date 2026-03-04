
import unittest
from cerberus.samplers import RandomSampler, MultiSampler
from cerberus.interval import Interval

class TestMultiSamplerCorrelation(unittest.TestCase):
    def test_multisampler_decorrelates_sub_samplers(self):
        """
        Verify that MultiSampler de-correlates identical sub-samplers
        via generate_sub_seeds, even when sub-samplers start with the same seed.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 5

        # Two identical samplers with same seed
        rs1 = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=42)
        rs2 = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=42)

        # Verify they are identical initially
        self.assertEqual([i.start for i in rs1], [i.start for i in rs2])

        # Wrap in MultiSampler with explicit seed
        ms = MultiSampler([rs1, rs2], chrom_sizes, folds=[], exclude_intervals={}, seed=42)

        # MultiSampler uses generate_sub_seeds to give each sub-sampler a
        # different derived seed, de-correlating them.
        intervals1 = list(rs1)
        intervals2 = list(rs2)

        self.assertNotEqual(intervals1, intervals2, "Sub-samplers should be de-correlated by MultiSampler")

if __name__ == '__main__':
    unittest.main()
