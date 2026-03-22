import pytest

from cerberus.samplers import ComplexityMatchedSampler


@pytest.fixture
def mock_fasta(tmp_path):
    fasta_path = tmp_path / "genome.fa"

    # chr1: All 'G' (100% GC)
    seq1 = "G" * 2000

    # chr2: All 'A' (0% GC)
    seq2 = "A" * 2000

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq1 + "\n")
        f.write(">chr2\n")
        f.write(seq2 + "\n")

    import pyfaidx

    pyfaidx.Faidx(str(fasta_path))

    return fasta_path


def test_complexity_sampler_padding(mock_fasta):
    """
    Test that ComplexityMatchedSampler fills the quota even if no matching candidates exist.
    """
    chrom_sizes = {"chr1": 2000, "chr2": 2000}

    # Targets: 10 intervals on chr1 (100% GC)
    target_bed = mock_fasta.parent / "target.bed"
    with open(target_bed, "w") as f:
        for i in range(10):
            start = i * 50
            f.write(f"chr1\t{start}\t{start + 50}\n")

    # Candidates: Only allow chr2 (0% GC) via explicit regions or file
    # We'll use RandomSampler restricted to chr2

    # We want strict mismatch.
    # Target (chr1) -> 100% GC.
    # Candidate (chr2) -> 0% GC.
    # Bins: 5. 100% falls in bin 4. 0% falls in bin 0.
    # So bin 4 will have 10 targets, 0 candidates.

    {
        "sampler_type": "complexity_matched",
        "padded_size": 50,
        "sampler_args": {
            "target_sampler": {
                "type": "interval",
                "args": {"intervals_path": str(target_bed)},
            },
            "candidate_sampler": {
                "type": "random",
                "args": {
                    "num_intervals": 100  # Plenty of candidates, but WRONG GC
                },
            },
            "bins": 5,
            "candidate_ratio": 1.0,
            "metrics": ["gc"],  # Only GC for simplicity
        },
    }

    # Create samplers manually to control exclusions
    from interlap import InterLap

    from cerberus.samplers import IntervalSampler, RandomSampler

    # 1. Target Sampler (No exclusions, reads from chr1)
    target_sampler = IntervalSampler(
        file_path=target_bed, chrom_sizes=chrom_sizes, padded_size=50
    )
    assert len(target_sampler) == 10

    # 2. Candidate Sampler (Exclude chr1, forces chr2)
    exclude_chr1 = {"chr1": InterLap([(0, 2000)])}
    candidate_sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=50,
        num_intervals=100,
        exclude_intervals=exclude_chr1,
        seed=42,
    )

    # 3. ComplexityMatchedSampler
    sampler = ComplexityMatchedSampler(
        target_sampler=target_sampler,
        candidate_sampler=candidate_sampler,
        fasta_path=mock_fasta,
        chrom_sizes=chrom_sizes,
        bins=5,
        candidate_ratio=1.0,
        metrics=["gc"],
        seed=42,
    )

    # 1. Length should be maintained (10 targets * 1.0 = 10)
    # Without the fix, this would be 0 because of mismatch.
    print(f"Sampler length: {len(sampler)}")
    assert len(sampler) == 10

    # 2. Content should be from chr2 (fallback candidates)
    # Since only chr2 is available for candidates, all fallback items must be chr2.
    chr2_count = sum(1 for iv in sampler if iv.chrom == "chr2")
    assert chr2_count == 10
