import numpy as np
import pytest
from interlap import InterLap

from cerberus.config import SamplerConfig
from cerberus.samplers import (
    ComplexityMatchedSampler,
    Interval,
    ListSampler,
    create_sampler,
)


@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"

    rng = np.random.default_rng(42)
    bases = np.array(['A', 'C', 'G', 'T'])

    def generate_seq(length, p):
        return "".join(rng.choice(bases, size=length, p=p))

    seq1 = generate_seq(2000, [0.1, 0.4, 0.4, 0.1])
    seq2 = generate_seq(2000, [0.4, 0.1, 0.1, 0.4])
    seq3 = "AC" * 1000

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq1 + "\n")
        f.write(">chr2\n")
        f.write(seq2 + "\n")
        f.write(">chr3\n")
        f.write(seq3 + "\n")

    import pyfaidx
    pyfaidx.Faidx(str(fasta_path))

    return fasta_path

def test_complexity_matched_sampler(mock_fasta):
    chrom_sizes = {"chr1": 2000, "chr2": 2000, "chr3": 2000}

    target_bed = mock_fasta.parent / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(20):
            start = i * 50
            f.write(f"chr1\t{start}\t{start+50}\n")

    config = SamplerConfig.model_construct(
        sampler_type="complexity_matched",
        padded_size=50,
        sampler_args={
            "target_sampler": SamplerConfig.model_construct(
                sampler_type="interval", padded_size=50,
                sampler_args={"intervals_path": target_bed},
            ),
            "candidate_sampler": SamplerConfig.model_construct(
                sampler_type="random", padded_size=50,
                sampler_args={"num_intervals": 1000},
            ),
            "bins": 5,
            "candidate_ratio": 1.0,
            "metrics": ["gc", "dust", "cpg"],
        },
    )

    sampler = create_sampler(
        config, chrom_sizes, [], {}, fasta_path=mock_fasta, seed=42
    )

    assert isinstance(sampler, ComplexityMatchedSampler)
    assert len(sampler) == 20

    chr1_count = sum(1 for iv in sampler if iv.chrom == "chr1")
    chr2_count = sum(1 for iv in sampler if iv.chrom == "chr2")
    chr3_count = sum(1 for iv in sampler if iv.chrom == "chr3")

    print(f"Counts: chr1={chr1_count}, chr2={chr2_count}, chr3={chr3_count}")

    assert chr1_count > 15
    assert chr2_count < 5
    assert chr3_count < 5

def test_complexity_matched_exclusions(mock_fasta):
    chrom_sizes = {"chr1": 2000}

    target_intervals = [Interval("chr1", 0, 100, "+")]
    target_sampler = ListSampler(intervals=target_intervals, chrom_sizes=chrom_sizes)

    candidate_intervals = [
        Interval("chr1", 100, 200, "+"),
        Interval("chr1", 200, 300, "+")
    ]
    candidate_sampler = ListSampler(intervals=candidate_intervals, chrom_sizes=chrom_sizes)

    exclude_intervals = {"chr1": InterLap()}
    exclude_intervals["chr1"].add((200, 300))

    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        exclude_intervals=exclude_intervals,
        candidate_ratio=10.0,
        bins=1,
        metrics=["gc"],
        seed=42
    )

    for interval in sampler:
        assert interval.start != 200
        assert interval.start == 100
