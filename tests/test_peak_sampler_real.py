
import gzip

from cerberus.exclude import get_exclude_intervals
from cerberus.genome import create_genome_folds, create_human_genome_config
from cerberus.samplers import PeakSampler


def test_peak_sampler_real_data(human_genome, mdapca2b_ar_dataset, tmp_path):
    """
    Real test of PeakSampler using mdapca2b-ar.narrowPeak.gz and hg38 genome.
    """
    # 1. Setup paths and config
    genome_dir = human_genome["fasta"].parent
    original_intervals_path = mdapca2b_ar_dataset["narrowPeak"]
    fasta_path = human_genome["fasta"]

    # Subset the peaks to speed up the test (88k peaks takes too long for unit test)
    intervals_path = tmp_path / "subset.narrowPeak.gz"
    
    with gzip.open(original_intervals_path, "rt") as f_in, gzip.open(intervals_path, "wt") as f_out:
        count = 0
        for line in f_in:
            f_out.write(line)
            count += 1
            if count >= 2000:
                break

    # Generate config to get chrom_sizes and fold setup
    config = create_human_genome_config(genome_dir)
    chrom_sizes = config.chrom_sizes

    # Create folds (PeakSampler needs these for splitting)
    folds = create_genome_folds(chrom_sizes, config.fold_type, config.fold_args)

    # Create exclude intervals
    exclude_intervals = get_exclude_intervals(config.exclude_intervals)

    # 2. Initialize PeakSampler
    sampler = PeakSampler(
        intervals_path=intervals_path,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        padded_size=1000, # Common size, arbitrary for test
        folds=folds,
        exclude_intervals=exclude_intervals,
        background_ratio=1.0,
        seed=42
    )

    initial_len = len(sampler)
    
    # Verify size
    # PeakSampler = Positives + Negatives. 
    # Negatives should match Positives count if candidates are sufficient.
    n_positives = len(sampler.positives)
    assert sampler.negatives is not None
    n_negatives = len(sampler.negatives)
    
    assert initial_len == n_positives + n_negatives
    # Allow small discrepancy if matching bins wasn't perfect, but usually it's exact or very close.
    # With 1.0 ratio, n_negatives should be roughly n_positives.
    assert n_negatives > 0
    assert abs(n_negatives - n_positives) < n_positives * 0.1 # Within 10%

    # 3. Test Resampling
    # Capture state before resampling
    intervals_before = list(sampler)
    
    # Resample
    sampler.resample(seed=123)
    
    intervals_after = list(sampler)
    new_len = len(sampler)
    
    # Verify length is maintained
    # print(f"Resampling change: {initial_len} -> {new_len} (delta: {new_len - initial_len})")
    assert new_len == initial_len, f"Sampler length changed after resampling! {initial_len} vs {new_len}"
    
    # Verify negatives changed
    # We can check overlap of the sets.
    # Positives are static. Negatives are dynamic.
    # Since we have 50% positives, about 50% of intervals should be identical (the positives).
    # The negatives should be largely different (random selection from large candidate pool).
    
    set_before = set((i.chrom, i.start, i.end) for i in intervals_before)
    set_after = set((i.chrom, i.start, i.end) for i in intervals_after)
    
    intersection = set_before.intersection(set_after)
    overlap_count = len(intersection)
    
    # We expect overlap to be exactly n_positives (since they are constant)
    # Plus potentially some chance overlap in negatives (unlikely to be 0 if pool is small, but pool is 10x)
    # So overlap should be close to n_positives.
    
    # Check that we have *at least* n_positives overlap
    assert overlap_count >= n_positives
    
    # Check that we have *some* difference (negatives changed)
    # i.e. overlap is not 100%
    assert overlap_count < new_len, "Resampling produced identical intervals (negatives didn't change)"

    # 4. Test Splitting
    train, val, test = sampler.split_folds()
    
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    
    # Verify sum
    # Note: ComplexityMatchedSampler regenerates candidates in the split folds,
    # so the matching success rate might vary slightly (some bins might find 0 candidates).
    total_split = n_train + n_val + n_test
    
    # We should not lose many intervals. Positives are constant. Negatives might fluctuate.
    # Allow 1% tolerance
    assert abs(total_split - new_len) <= max(1, new_len * 0.01)
    
    # Verify disjointness (sanity check)
    # Note: Interval objects might be equal but different instances.
    # We check coordinates.
    train_intervals = set((i.chrom, i.start, i.end) for i in train)
    val_intervals = set((i.chrom, i.start, i.end) for i in val)
    test_intervals = set((i.chrom, i.start, i.end) for i in test)
    
    assert train_intervals.isdisjoint(val_intervals)
    assert train_intervals.isdisjoint(test_intervals)
    assert val_intervals.isdisjoint(test_intervals)

