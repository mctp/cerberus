
from cerberus.samplers import IntervalSampler
from cerberus.genome import create_genome_folds

def test_sampler_split_folds_overlap(tmp_path):
    # Create a mock BED file
    bed_path = tmp_path / "test.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t100\t200\n") # Fold 0
        f.write("chr2\t100\t200\n") # Fold 1
        
    chrom_sizes = {'chr1': 1000, 'chr2': 1000}
    
    # Init sampler with 2 folds
    # chr1 -> Fold 0 (size 1000)
    # chr2 -> Fold 1 (size 1000)
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 2})
    sampler = IntervalSampler(
        file_path=bed_path,
        chrom_sizes=chrom_sizes,
        padded_size=100,
        exclude_intervals={},
        folds=folds
    )
    
    # Test 1: test_fold=None, val_fold=0
    train, val, test = sampler.split_folds(test_fold=None, val_fold=0)
    assert len(test) == 0
    assert len(val) == 1
    assert val[0].chrom == "chr1"
    assert len(train) == 1
    assert train[0].chrom == "chr2"
    
    # Test 2: Overlap test_fold=0, val_fold=0
    train, val, test = sampler.split_folds(test_fold=0, val_fold=0)
    assert len(test) == 1
    assert test[0].chrom == "chr1"
    
    assert len(val) == 1
    assert val[0].chrom == "chr1"
    
    assert len(train) == 1
    assert train[0].chrom == "chr2"
    
    # Verify exact object identity or equality
    assert test[0] == val[0]
    
    # Test 3: Disjoint test_fold=0, val_fold=1
    train, val, test = sampler.split_folds(test_fold=0, val_fold=1)
    assert len(test) == 1
    assert test[0].chrom == "chr1"
    
    assert len(val) == 1
    assert val[0].chrom == "chr2"
    
    assert len(train) == 0
