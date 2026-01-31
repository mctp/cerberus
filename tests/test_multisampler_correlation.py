
import unittest
from cerberus.samplers import RandomSampler, MultiSampler
from cerberus.interval import Interval

class TestMultiSamplerCorrelation(unittest.TestCase):
    def test_multisampler_none_seed_correlation(self):
        """
        Demonstrates that MultiSampler(seed=None) fails to de-correlate 
        identical sub-samplers, because it propagates None to them.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 5
        
        # Two identical samplers with same seed (or None, which settles to a seed)
        # To be strict, let's start them with same seed 42.
        rs1 = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=42)
        rs2 = RandomSampler(chrom_sizes, padded_size, num_intervals, seed=42)
        
        # Verify they are identical initially
        self.assertEqual([i.start for i in rs1], [i.start for i in rs2])
        
        # Wrap in MultiSampler with seed=None
        ms = MultiSampler([rs1, rs2], chrom_sizes, folds=[], exclude_intervals={}, seed=None)
        
        # Trigger resample(None) which happens on init if generate_on_init=True (default)
        # But let's call it explicitly to be sure we are testing the "Next Epoch" or "Init" behavior.
        # Actually, ms init called resample(None).
        # rs1.resample(None) and rs2.resample(None) were called.
        
        # Since ms.seed was None, it passed None to sub-samplers.
        # rs1 advanced from 42 -> X
        # rs2 advanced from 42 -> Y
        # Since rs1 and rs2 were in same state 42, X == Y.
        
        # Check indices from MultiSampler
        # MultiSampler stores (sampler_idx, interval_idx)
        # We can just check the intervals directly from the sub-samplers
        
        intervals1 = list(rs1)
        intervals2 = list(rs2)
        
        print(f"RS1 Intervals: {[i.start for i in intervals1]}")
        print(f"RS2 Intervals: {[i.start for i in intervals2]}")
        
        # This assertion failing proves the bug. 
        # We WANT them to be different (de-correlated).
        self.assertNotEqual(intervals1, intervals2, "Sub-samplers should be de-correlated even if MultiSampler seed is None")

if __name__ == '__main__':
    unittest.main()
