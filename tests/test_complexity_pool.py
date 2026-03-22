import numpy as np
import pytest

from cerberus.samplers import ComplexityMatchedSampler, IntervalSampler, RandomSampler


@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    rng = np.random.default_rng(42)
    bases = np.array(["A", "C", "G", "T"])

    # Simple genome
    seq = "".join(rng.choice(bases, size=5000))

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq + "\n")

    import pyfaidx

    pyfaidx.Faidx(str(fasta_path))
    return fasta_path


class MockRandomSampler(RandomSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resample_calls = 0

    def resample(self, seed=None):
        self.resample_calls += 1
        super().resample(seed)


def test_pool_reuse(mock_fasta, tmp_path):
    chrom_sizes = {"chr1": 5000}
    padded_size = 50

    # Create target bed
    target_bed = tmp_path / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(10):
            start = i * 100
            f.write(f"chr1\t{start}\t{start + 50}\n")

    target_sampler = IntervalSampler(target_bed, chrom_sizes, padded_size)

    # Candidate sampler (Random)
    # 10 targets -> need ~10 negatives.
    # Provide 100 candidates (10x pool).
    candidate_sampler = MockRandomSampler(
        chrom_sizes, padded_size, num_intervals=100, generate_on_init=False
    )

    # Init ComplexityMatchedSampler
    # generate_on_init=True will trigger first resample -> first pool init.
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        candidate_ratio=1.0,
        generate_on_init=True,
        metrics=["gc"],
    )

    # Verify candidate_sampler was resampled once (during pool initialization)
    assert candidate_sampler.resample_calls == 1
    assert sampler._initialized is True

    # Resample again (epoch 2)
    sampler.resample()

    # Verify candidate_sampler was NOT resampled again
    assert candidate_sampler.resample_calls == 1

    # Resample again (epoch 3)
    sampler.resample()
    assert candidate_sampler.resample_calls == 1

    # Verify we are getting intervals
    items = list(sampler)
    assert len(items) == 10


def test_lazy_pool_initialization(mock_fasta, tmp_path):
    chrom_sizes = {"chr1": 5000}
    padded_size = 50
    target_bed = tmp_path / "target.bed"
    with open(target_bed, "w") as f:
        f.write("chr1\t0\t50\n")
    target_sampler = IntervalSampler(target_bed, chrom_sizes, padded_size)
    candidate_sampler = MockRandomSampler(
        chrom_sizes, padded_size, num_intervals=100, generate_on_init=False
    )

    # Init with generate_on_init=False
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        generate_on_init=False,
    )

    # Should NOT have initialized pool yet
    assert not sampler._initialized
    assert candidate_sampler.resample_calls == 0

    # First resample
    sampler.resample()

    # Should now be initialized
    assert sampler._initialized
    assert candidate_sampler.resample_calls == 1
