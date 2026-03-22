
import unittest

from interlap import InterLap

from cerberus.samplers import RandomSampler, ScaledSampler


class TestScaledSamplerFix(unittest.TestCase):
    def test_scaled_sampler_split_folds_after_resample(self):
        """
        Verify that ScaledSampler updates its seed during resample(None),
        ensuring that split_folds returns different splits in the next epoch.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 10
        seed = 42
        folds = [
            {"chr1": InterLap([(0, 5000)])},
            {"chr1": InterLap([(5000, 10000)])}
        ]
        
        # Base sampler
        rs = RandomSampler(chrom_sizes, padded_size, num_intervals, folds=folds, seed=seed)
        
        # Scaled sampler
        ss = ScaledSampler(rs, num_samples=5, seed=seed)
        
        # Epoch 0
        train0, val0, test0 = ss.split_folds(test_fold=0, val_fold=1)
        seed0 = ss.seed
        
        # Next Epoch
        ss.resample(None)
        
        # Epoch 1
        train1, val1, test1 = ss.split_folds(test_fold=0, val_fold=1)
        seed1 = ss.seed
        
        # Check that seed updated
        self.assertNotEqual(seed0, seed1, "ScaledSampler seed should update after resample(None)")
        
        # Check that split samplers have different seeds
        self.assertNotEqual(train0.seed, train1.seed, "Split samplers should have different seeds across epochs")
        
        # If RandomSampler logic works as expected, the underlying intervals in train0 and train1 
        # should also be different (because RandomSampler also resampled).
        # But here we focus on ScaledSampler properties.

if __name__ == '__main__':
    unittest.main()
