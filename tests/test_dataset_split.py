from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_folds, GenomeConfig
from cerberus.config import DataConfig, SamplerConfig, FoldArgs, IntervalSamplerArgs
from cerberus.samplers import IntervalSampler

def test_sampler_split_folds(tmp_path):
    # Create a mock BED file
    bed_path = tmp_path / "test.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t100\t200\n")
        f.write("chr1\t300\t400\n")
        f.write("chr2\t100\t200\n")
        f.write("chr3\t100\t200\n")

    chrom_sizes = {'chr1': 1000, 'chr2': 1000, 'chr3': 1000}

    # Init sampler with 3 folds
    folds = create_genome_folds(chrom_sizes, fold_type="chrom_partition", fold_args={"k": 3})
    sampler = IntervalSampler(
        file_path=bed_path,
        chrom_sizes=chrom_sizes,
        padded_size=100,
        exclude_intervals={},
        folds=folds
    )

    assert sampler.folds is not None
    assert len(sampler.folds) == 3

    # Split
    train_sampler, val_sampler, test_sampler = sampler.split_folds(test_fold=0, val_fold=1)

    # Check lengths
    total = len(train_sampler) + len(val_sampler) + len(test_sampler)
    assert total == 4

    # Check that they are mutually exclusive
    all_intervals = set()
    for s in [train_sampler, val_sampler, test_sampler]:
        for interval in s:
            assert id(interval) not in all_intervals
            all_intervals.add(id(interval))

def test_dataset_split_folds(tmp_path):
    # Mock configs
    bed_path = tmp_path / "test.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t100\t200\n")
        f.write("chr2\t100\t200\n")

    chrom_sizes = {'chr1': 1000, 'chr2': 1000}
    genome_config = GenomeConfig.model_construct(
        name='hg38',
        fasta_path=tmp_path / "genome.fa",
        allowed_chroms=['chr1', 'chr2'],
        chrom_sizes=chrom_sizes,
        exclude_intervals={},
        fold_type='chrom_partition',
        fold_args=FoldArgs.model_construct(k=2, test_fold=None, val_fold=None),
    )

    # Create dummy fasta
    with open(tmp_path / "genome.fa", "w") as f:
        f.write(">chr1\n" + "A"*1000 + "\n")
        f.write(">chr2\n" + "T"*1000 + "\n")

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=1,
        output_bin_size=1,
        encoding='one_hot',
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type='interval',
        padded_size=100,
        sampler_args=IntervalSamplerArgs.model_construct(intervals_path=bed_path),
    )

    dataset = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None)

    # Split using dataset method
    train_ds, val_ds, test_ds = dataset.split_folds(test_fold=0, val_fold=1)

    # Verify shared extractor
    assert train_ds.sequence_extractor is dataset.sequence_extractor
    assert val_ds.sequence_extractor is dataset.sequence_extractor
    assert test_ds.sequence_extractor is dataset.sequence_extractor

    # Verify exclude mask sharing
    assert train_ds.exclude_intervals is dataset.exclude_intervals

    # Verify split correctness (basic check)
    # 2 folds, 2 chromosomes. Each fold gets 1 chrom.
    # test_fold=0 -> chr1
    # val_fold=1 -> chr2
    # train -> empty

    # Note: depends on create_folds sorting.
    # chr1 and chr2 equal size.
    # chr1 -> fold 0
    # chr2 -> fold 1

    assert len(test_ds) == 1
    assert len(val_ds) == 1
    assert len(train_ds) == 0
