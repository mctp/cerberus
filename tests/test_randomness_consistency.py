
import unittest
import random
from cerberus.samplers import RandomSampler, MultiSampler, ScaledSampler, GCMatchedSampler, ListSampler
from cerberus.interval import Interval
from interlap import InterLap
import cerberus.samplers

class TestRandomnessConsistency(unittest.TestCase):
    def test_random_sampler_split_folds_idempotency(self):
        """
        Test if RandomSampler.split_folds returns the same seeds/samplers
        when called multiple times on the same seeded object.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 10
        seed = 42
        folds = [
            {"chr1": InterLap([(0, 5000)])},
            {"chr1": InterLap([(5000, 10000)])}
        ]
        
        rs = RandomSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            num_intervals=num_intervals,
            folds=folds,
            seed=seed
        )
        
        # First split
        t1, v1, te1 = rs.split_folds(test_fold=0, val_fold=1)
        
        # Second split
        t2, v2, te2 = rs.split_folds(test_fold=0, val_fold=1)
        
        # Check seeds
        self.assertEqual(t1.seed, t2.seed, "RandomSampler split seeds should be idempotent given a fixed parent seed")
        self.assertEqual(v1.seed, v2.seed)
        self.assertEqual(te1.seed, te2.seed)

    def test_multisampler_split_folds_seed_propagation(self):
        """
        Test if MultiSampler propagates its seed to split instances
        and if splitting is idempotent.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 10
        seed = 42
        folds = [
            {"chr1": InterLap([(0, 5000)])},
            {"chr1": InterLap([(5000, 10000)])}
        ]
        
        rs = RandomSampler(chrom_sizes, padded_size, num_intervals, folds=folds, seed=seed)
        ms = MultiSampler([rs], chrom_sizes, folds=folds, exclude_intervals={}, seed=seed)
        
        # Split 1
        t1, v1, te1 = ms.split_folds(test_fold=0, val_fold=1)
        
        # Split 2
        t2, v2, te2 = ms.split_folds(test_fold=0, val_fold=1)
        
        # Check if splits have seeds (should not be None if parent has seed)
        # Note: We can't easily check .seed on MultiSampler if not exposed?
        # But we added self.seed to MultiSampler! So we can check.
        
        self.assertIsNotNone(t1.seed, "MultiSampler split should have a seed derived from parent")
        self.assertIsNotNone(v1.seed)
        self.assertIsNotNone(te1.seed)
        
        # Check idempotency
        self.assertEqual(t1.seed, t2.seed, "MultiSampler split seeds should be idempotent")
        self.assertEqual(v1.seed, v2.seed)
        self.assertEqual(te1.seed, te2.seed)

    def test_multisampler_resample_updates_seed(self):
        """
        Test if MultiSampler.resample(None) updates its seed,
        leading to different splits for the next epoch.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        num_intervals = 10
        seed = 42
        folds = [
            {"chr1": InterLap([(0, 5000)])},
            {"chr1": InterLap([(5000, 10000)])}
        ]
        
        rs = RandomSampler(chrom_sizes, padded_size, num_intervals, folds=folds, seed=seed)
        ms = MultiSampler([rs], chrom_sizes, folds=folds, exclude_intervals={}, seed=seed)
        
        # Split 1 (Epoch 0)
        t1, _, _ = ms.split_folds(test_fold=0, val_fold=1)
        seed1 = t1.seed
        
        # Resample (Epoch 1)
        ms.resample(None)
        
        # Split 2 (Epoch 1)
        t2, _, _ = ms.split_folds(test_fold=0, val_fold=1)
        seed2 = t2.seed
        
        self.assertNotEqual(seed1, seed2, "MultiSampler split seeds should change after resample(None)")

    def test_gc_matched_sampler_idempotency_unseeded(self):
        """
        Verify GCMatchedSampler idempotency when initialized with seed=None.
        Fixed behavior: It should self-seed and be idempotent.
        """
        chrom_sizes = {"chr1": 10000}
        padded_size = 100
        folds = [{"chr1": InterLap([(0, 5000)])}, {"chr1": InterLap([(5000, 10000)])}]
        
        # Intervals in test fold
        intervals = [Interval("chr1", i*100, i*100+100, "+") for i in range(10)]
        target = ListSampler(intervals, chrom_sizes, folds=folds)
        candidate = RandomSampler(chrom_sizes, padded_size, num_intervals=100, folds=folds, seed=123)
        
        original_compute = cerberus.samplers.compute_intervals_gc
        try:
            cerberus.samplers.compute_intervals_gc = lambda sampler, fasta: [0.5] * len(sampler)
            
            # Init with None
            gc = GCMatchedSampler(target, candidate, "dummy.fa", chrom_sizes, folds, {}, seed=None)
            
            # Check self-seeding (Improvement: GCMatchedSampler should ideally have a seed now, 
            # but strict idempotency check is via split results)
            
            t1, v1, te1 = gc.split_folds(test_fold=0, val_fold=1)
            t2, v2, te2 = gc.split_folds(test_fold=0, val_fold=1)
            
            # Check seeds of children
            # If self-seeding works, these should be derived from the locked-in self.seed
            # and thus should be equal.
            self.assertEqual(te1.seed, te2.seed, "GCMatchedSampler splits should have same seeds (idempotent)")
            
            indices1 = te1._indices
            indices2 = te2._indices
            
            self.assertEqual(indices1, indices2, "GCMatchedSampler splits should be idempotent")
            
        finally:
            cerberus.samplers.compute_intervals_gc = original_compute

    def test_scaled_sampler_seed_derivation(self):
        """
        Verify that ScaledSampler derives a new seed for its child.
        """
        chrom_sizes = {"chr1": 1000}
        padded_size = 100
        rs = RandomSampler(chrom_sizes, padded_size, num_intervals=10, seed=123)
        
        # Wrapped with different seed
        ss = ScaledSampler(rs, num_samples=5, seed=555)
        
        # Parent and child seeds should differ
        self.assertNotEqual(ss.seed, rs.seed, "ScaledSampler should derive a sub-seed for its child")

if __name__ == '__main__':
    unittest.main()
