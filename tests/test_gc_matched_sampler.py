import pytest
from pathlib import Path
import random
from cerberus.samplers import create_sampler, RandomSampler, ComplexityMatchedSampler
from cerberus.sequence import calculate_gc_content
from cerberus.config import (
    SamplerConfig, ComplexityMatchedSamplerArgs, RandomSamplerArgs,
    IntervalSamplerArgs,
)

@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("GC" * 500 + "\n")
        f.write(">chr2\n")
        f.write("AT" * 500 + "\n")
        f.write(">chr3\n")
        f.write("ACGT" * 250 + "\n")

    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))

    return fasta_path

def test_calculate_gc_content():
    assert calculate_gc_content("GCGC") == 1.0
    assert calculate_gc_content("ATAT") == 0.0
    assert calculate_gc_content("ACGT") == 0.5
    assert calculate_gc_content("NNNN") == 0.0
    assert calculate_gc_content("GCN") == 1.0

def test_random_sampler():
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    padded_size = 100
    num_intervals = 50
    exclude_intervals = {}
    folds = []

    sampler = RandomSampler(chrom_sizes, padded_size, num_intervals, folds, exclude_intervals, seed=42)

    assert len(sampler) == num_intervals
    for interval in sampler:
        assert interval.chrom in chrom_sizes
        assert interval.end - interval.start == padded_size
        assert interval.start >= 0
        assert interval.end <= chrom_sizes[interval.chrom]

def test_complexity_matched_sampler_gc_only(mock_fasta):
    chrom_sizes = {"chr1": 1000, "chr2": 1000, "chr3": 1000}
    padded_size = 10

    target_bed = mock_fasta.parent / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(10):
            f.write(f"chr1\t{i*10}\t{i*10+10}\n")

    config = SamplerConfig.model_construct(
        sampler_type="complexity_matched",
        padded_size=10,
        sampler_args=ComplexityMatchedSamplerArgs.model_construct(
            target_sampler=SamplerConfig.model_construct(
                sampler_type="interval", padded_size=10,
                sampler_args=IntervalSamplerArgs.model_construct(intervals_path=target_bed),
            ),
            candidate_sampler=SamplerConfig.model_construct(
                sampler_type="random", padded_size=10,
                sampler_args=RandomSamplerArgs.model_construct(num_intervals=300),
            ),
            bins=10, candidate_ratio=1.0, metrics=["gc"],
        ),
    )

    sampler = create_sampler(
        config, chrom_sizes, [], {}, fasta_path=mock_fasta
    )

    assert isinstance(sampler, ComplexityMatchedSampler)
    assert len(sampler) == 10

    count_chr1 = 0
    count_chr2 = 0
    count_chr3 = 0

    for interval in sampler:
        if interval.chrom == "chr1": count_chr1 += 1
        elif interval.chrom == "chr2": count_chr2 += 1
        elif interval.chrom == "chr3": count_chr3 += 1

    print(f"Counts: chr1={count_chr1}, chr2={count_chr2}, chr3={count_chr3}")

    assert count_chr1 == 10
    assert count_chr2 == 0
    assert count_chr3 == 0
