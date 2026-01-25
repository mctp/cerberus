
import unittest
from cerberus.samplers import RandomSampler
from interlap import InterLap
import random

class TestRandomSamplerInconsistency(unittest.TestCase):
    def test_split_folds_stuck_seed(self):
        """
        Demonstrates that split_folds returns samplers with the same seed
        even after the parent sampler has been resampled (advanced).
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 20
        # Define folds such that we have distinct regions
        folds = [
            {"chr1": InterLap([(0, 3000)])},      # Train
            {"chr1": InterLap([(3000, 6000)])},   # Val
            {"chr1": InterLap([(6000, 10000)])},  # Test
        ]
        
        seed = 42
        s = RandomSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            num_intervals=num_intervals,
            folds=folds,
            seed=seed
        )
        
        # Split 1 (Initial state)
        train1, val1, test1 = s.split_folds()
        
        # Advance parent
        s.resample(None)
        
        # Split 2 (Advanced state)
        train2, val2, test2 = s.split_folds()
        
        # Check seeds
        print(f"Train1 Seed: {train1.seed}")
        print(f"Train2 Seed: {train2.seed}")
        
        # Current behavior: Seeds are identical because they are derived from s.seed (42)
        # Expected/Desired behavior: Seeds should differ because parent RNG advanced
        self.assertNotEqual(train1.seed, train2.seed, "Seeds are identical but should be different after parent resample")
        
        # Check intervals overlap
        # Since seeds are different, intervals should be different
        ints1 = list(train1)
        ints2 = list(train2)
        min_len = min(len(ints1), len(ints2))
        
        if min_len > 0:
            self.assertNotEqual(
                [i.start for i in ints1[:min_len]],
                [i.start for i in ints2[:min_len]],
                "Intervals are identical prefix but should be different"
            )

if __name__ == '__main__':
    unittest.main()
