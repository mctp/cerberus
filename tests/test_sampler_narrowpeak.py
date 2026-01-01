import pytest
from cerberus.samplers import IntervalSampler

def test_interval_sampler_reads_narrowpeak_gz(mdapca2b_ar_dataset):
    """
    Test that IntervalSampler can read .narrowPeak.gz files directly.
    """
    narrowpeak_path = mdapca2b_ar_dataset["narrowPeak"]
    
    # Mock chrom_sizes to accept any chromosome with large size
    chrom_sizes = {f"chr{i}": 10**9 for i in range(1, 23)}
    chrom_sizes.update({"chrX": 10**9, "chrY": 10**9, "chrM": 10**9})
    
    folds = [{} for _ in range(5)]
    exclude_intervals = {}
    
    try:
        sampler = IntervalSampler(
            file_path=narrowpeak_path,
            chrom_sizes=chrom_sizes,
            padded_size=1000,
            exclude_intervals=exclude_intervals,
            folds=folds
        )
        
        print(f"Successfully loaded {len(sampler)} intervals from narrowPeak.gz")
        assert len(sampler) > 0
        
    except Exception as e:
        pytest.fail(f"IntervalSampler failed to read narrowPeak.gz: {e}")
