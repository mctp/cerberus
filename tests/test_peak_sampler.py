
import pytest
from unittest.mock import MagicMock, patch
from cerberus.samplers import PeakSampler, IntervalSampler, RandomSampler, GCMatchedSampler
from cerberus.interval import Interval
from interlap import InterLap

@pytest.fixture
def mock_dependencies():
    with patch("cerberus.samplers.IntervalSampler") as mock_interval_sampler, \
         patch("cerberus.samplers.RandomSampler") as mock_random_sampler, \
         patch("cerberus.samplers.GCMatchedSampler") as mock_gc_matched_sampler, \
         patch("cerberus.samplers.compute_intervals_gc") as mock_compute_gc:
         
        # Setup mock return values
        mock_interval_instance = MagicMock(spec=IntervalSampler)
        mock_interval_instance.__len__.return_value = 100
        # Create some fake intervals
        intervals = [Interval("chr1", i*100, i*100+50, "+") for i in range(100)]
        mock_interval_instance.__iter__.return_value = iter(intervals)
        mock_interval_instance.__getitem__.side_effect = lambda i: intervals[i]
        mock_interval_sampler.return_value = mock_interval_instance
        
        mock_random_instance = MagicMock(spec=RandomSampler)
        mock_random_sampler.return_value = mock_random_instance
        
        mock_gc_instance = MagicMock(spec=GCMatchedSampler)
        mock_gc_matched_sampler.return_value = mock_gc_instance
        
        yield {
            "interval_sampler": mock_interval_sampler,
            "random_sampler": mock_random_sampler,
            "gc_matched_sampler": mock_gc_matched_sampler,
            "compute_gc": mock_compute_gc
        }

def test_peak_sampler_init(mock_dependencies):
    chrom_sizes = {"chr1": 10000}
    exclude_intervals = {"chr1": InterLap()}
    
    # Init PeakSampler
    sampler = PeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        exclude_intervals=exclude_intervals,
        background_ratio=1.0
    )
    
    # Verify IntervalSampler created (Positives)
    mock_dependencies["interval_sampler"].assert_called_once()
    
    # Verify RandomSampler created (Candidates)
    mock_dependencies["random_sampler"].assert_called_once()
    call_args = mock_dependencies["random_sampler"].call_args
    # Check that num_intervals is max(10000, 100 * 1.0 * 10) = 10000
    assert call_args.kwargs["num_intervals"] == PeakSampler.MIN_CANDIDATES
    
    # Verify GCMatchedSampler created (Negatives)
    mock_dependencies["gc_matched_sampler"].assert_called_once()
    
    # Check exclusions passed to RandomSampler include the peaks
    random_excludes = call_args.kwargs["exclude_intervals"]
    assert "chr1" in random_excludes
    # We added 100 peaks to it. The original had 0.
    assert len(list(random_excludes["chr1"])) == 100

def test_peak_sampler_resample(mock_dependencies):
    chrom_sizes = {"chr1": 10000}
    exclude_intervals = {"chr1": InterLap()}
    
    sampler = PeakSampler(
        intervals_path="peaks.bed",
        fasta_path="genome.fa",
        chrom_sizes=chrom_sizes,
        padded_size=50,
        exclude_intervals=exclude_intervals
    )
    
    # Call resample
    sampler.resample(seed=42)
    
    # Verify negatives.resample was called
    assert sampler.negatives is not None
    # PeakSampler is a MultiSampler with [positives, negatives].
    # MultiSampler propagates seed + index.
    # positives is index 0 (seed 42), negatives is index 1 (seed 43).
    sampler.negatives.resample.assert_called_with(43)  # type: ignore
