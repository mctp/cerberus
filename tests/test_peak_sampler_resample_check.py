
from pathlib import Path

from interlap import InterLap

from cerberus.samplers import ComplexityMatchedSampler, PeakSampler


# Create a dummy fasta file
def create_dummy_fasta(path):
    with open(path, "w") as f:
        f.write(">chr1\n")
        f.write("ACGT" * 1000 + "\n") # 4000 bp

# Create a dummy peaks file
def create_dummy_peaks(path):
    with open(path, "w") as f:
        # Create 20 peaks to cover train and val regions
        for i in range(20):
            f.write(f"chr1\t{i*200}\t{i*200+100}\tpeak_{i}\t1000\t.\t50\t100\t-1\t-1\n")

def test_peak_sampler_resampling_behavior(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    peaks_path = tmp_path / "peaks.bed"
    create_dummy_fasta(fasta_path)
    create_dummy_peaks(peaks_path)
    
    chrom_sizes = {"chr1": 4000}
    folds = [
        {"chr1": InterLap([(0, 2000)])}, # Train
        {"chr1": InterLap([(2000, 3000)])}, # Val
        {"chr1": InterLap([(3000, 4000)])} # Test
    ]
    # Note: IntervalSampler centers intervals. Peaks are length 100.
    # peak 0: 0-100. Mid 50. Padded 20 -> 40-60.
    
    # We use a small padded size
    padded_size = 20
    
    # Initialize PeakSampler
    sampler = PeakSampler(
        intervals_path=peaks_path,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        padded_size=padded_size,
        folds=folds,
        exclude_intervals=None,
        background_ratio=1.0, # 1 background per peak
        min_candidates=50, # Force enough candidates
        candidate_oversample_factor=5.0,
        seed=42
    )
    
    # Split folds
    # test_fold=2, val_fold=1 -> Train is fold 0
    train_sampler, val_sampler, test_sampler = sampler.split_folds(test_fold=2, val_fold=1)
    
    # --- Check Train Resampling ---
    
    # 1. Get initial state
    train_intervals_1 = list(train_sampler)
    [iv for iv in train_intervals_1 if "peak" not in str(iv)] # We can't distinguish easily by name unless we track it.
    # Actually, PeakSampler mixes them.
    # Positives are from IntervalSampler. Negatives are from ComplexityMatchedSampler -> RandomSampler.
    # RandomSampler intervals don't have special names, they are just intervals.
    # But positives from BED usually don't carry the name into the Interval object unless parsed?
    # Interval object: chrom, start, end, strand.
    # Let's rely on exact equality.
    
    print(f"Train size: {len(train_sampler)}")
    
    # 2. Resample
    train_sampler.resample(seed=123)
    train_intervals_2 = list(train_sampler)
    
    # 3. Verify changes
    # The set of intervals should be different (specifically the background ones)
    assert set(str(iv) for iv in train_intervals_1) != set(str(iv) for iv in train_intervals_2)
    print("Verified: Train sampler intervals changed after resample.")
    
    # --- Check Val Stability (Simulating no resample call) ---
    
    val_intervals_1 = list(val_sampler)
    
    # Simulate "Epoch 2" for Val - i.e. we do NOT call resample
    val_intervals_2 = list(val_sampler)
    
    assert set(str(iv) for iv in val_intervals_1) == set(str(iv) for iv in val_intervals_2)
    print("Verified: Val sampler intervals stable without resample call.")
    
    # --- Check Val Resampling Capability ---
    # If we DID call resample, it SHOULD change (proving it's capable, just not called)
    val_sampler.resample(seed=999)
    val_intervals_3 = list(val_sampler)
    
    # Note: With small sample size and random selection, there's a tiny chance of collision, 
    # but with enough candidates and ratio, it should differ.
    assert set(str(iv) for iv in val_intervals_1) != set(str(iv) for iv in val_intervals_3)
    print("Verified: Val sampler capable of resampling.")
    
    # --- Check Static Candidate Pool ---
    # We want to verify the candidate pool (RandomSampler) inside ComplexityMatchedSampler is NOT regenerated.
    # This is internal implementation detail, but we can check if the underlying candidates change.
    # Accessing internals:
    # PeakSampler (MultiSampler) .samplers[1] is Negatives (ComplexityMatchedSampler)
    # But we split them.
    # train_sampler is MultiSampler. samplers[1] is ComplexityMatchedSampler.
    
    train_neg = train_sampler.samplers[1]
    assert isinstance(train_neg, ComplexityMatchedSampler)
    
    pool_1 = list(train_neg.candidate_sampler)
    
    train_sampler.resample(seed=555)
    
    pool_2 = list(train_neg.candidate_sampler)
    
    # The pool should be identical (indices might be accessed differently, but the list of intervals in random sampler is fixed)
    # Wait, RandomSampler stores `_intervals`.
    assert len(pool_1) == len(pool_2)
    assert set(str(iv) for iv in pool_1) == set(str(iv) for iv in pool_2)
    print("Verified: Candidate pool is static (RandomSampler not re-run).")

if __name__ == "__main__":
    # Manually run if executed as script
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_peak_sampler_resampling_behavior(Path(tmp))
