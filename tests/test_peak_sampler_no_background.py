
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from cerberus.samplers import PeakSampler, IntervalSampler, RandomSampler, GCMatchedSampler, create_sampler
from cerberus.interval import Interval
from interlap import InterLap

class TestPeakSamplerNoBackground(unittest.TestCase):
    def test_zero_background_ratio(self):
        chrom_sizes = {"chr1": 10000}
        exclude_intervals = {"chr1": InterLap()}
        
        # Mock dependencies
        with patch("cerberus.samplers.IntervalSampler") as MockIntervalSampler, \
             patch("cerberus.samplers.RandomSampler") as MockRandomSampler, \
             patch("cerberus.samplers.GCMatchedSampler") as MockGCMatchedSampler:

            # Setup Mock IntervalSampler (Positives)
            positives = MagicMock(spec=IntervalSampler)
            positives.__len__.return_value = 10
            positives.__iter__.return_value = iter([Interval("chr1", i*100, i*100+50, "+") for i in range(10)])
            positives.__getitem__.side_effect = lambda i: Interval("chr1", i*100, i*100+50, "+")
            # Make sure split_folds returns valid samplers (mocks)
            positives.split_folds.return_value = (MagicMock(spec=IntervalSampler), MagicMock(spec=IntervalSampler), MagicMock(spec=IntervalSampler))
            
            MockIntervalSampler.return_value = positives

            # We don't expect RandomSampler or GCMatchedSampler to be called
            
            sampler = PeakSampler(
                intervals_path="dummy_peaks.bed",
                fasta_path=None, # Should be allowed now
                chrom_sizes=chrom_sizes,
                padded_size=50,
                exclude_intervals=exclude_intervals,
                background_ratio=0.0
            )

            # Check if RandomSampler was NOT initialized
            MockRandomSampler.assert_not_called()
            
            # Check if GCMatchedSampler was NOT initialized
            MockGCMatchedSampler.assert_not_called()
            
            # Check total length (should be 10)
            self.assertEqual(len(sampler), 10)
            
            # Verify iteration yields positives
            # Note: PeakSampler (MultiSampler) shuffles by default, so we check if elements are the same
            self.assertCountEqual(list(sampler), list(positives))
            
            print("Verified: PeakSampler with background_ratio=0.0 does not create background samplers.")

    def test_create_sampler_peak_no_background(self):
        chrom_sizes = {"chr1": 10000}
        exclude_intervals = {"chr1": InterLap()}
        
        config = {
            "sampler_type": "peak",
            "padded_size": 50,
            "sampler_args": {
                "intervals_path": "dummy_peaks.bed",
                "background_ratio": 0.0 # Explicitly set to 0.0
            }
        }
        
        with patch("cerberus.samplers.PeakSampler") as MockPeakSampler:
            create_sampler(
                config=config,
                chrom_sizes=chrom_sizes,
                exclude_intervals=exclude_intervals,
                folds=[],
                fasta_path=None # Should be allowed
            )
            
            MockPeakSampler.assert_called_once()
            call_args = MockPeakSampler.call_args
            self.assertEqual(call_args.kwargs["background_ratio"], 0.0)
            self.assertIsNone(call_args.kwargs["fasta_path"])
            print("Verified: create_sampler allows fasta_path=None when background_ratio=0.0")

if __name__ == "__main__":
    unittest.main()
