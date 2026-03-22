import pytest
import torch

from cerberus.genome import create_genome_config
from cerberus.interval import Interval
from cerberus.sequence import InMemorySequenceExtractor, SequenceExtractor, encode_dna


def test_genome_from_fasta_human_supported(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text(">chr1\nACGT\n>chr2\nACGTACGT")

    fai_path = tmp_path / "test.fa.fai"
    fai_content = "chr1\t4\t6\t4\t5\nchr2\t8\t19\t8\t9\n"
    fai_path.write_text(fai_content)

    # Human should work
    genome = create_genome_config(
        name="test_genome", fasta_path=fasta_path, species="human"
    )

    assert genome.name == "test_genome"
    assert genome.chrom_sizes is not None
    assert "chr1" in genome.chrom_sizes
    assert "chr2" in genome.chrom_sizes


def test_genome_from_fasta_unsupported_species(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")
    (tmp_path / "test.fa.fai").touch()

    with pytest.raises(NotImplementedError, match="Species 'dog' is not supported"):
        create_genome_config(name="test_genome", fasta_path=fasta_path, species="dog")


def test_genome_from_fasta_allowed_chroms(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")

    fai_path = tmp_path / "test.fa.fai"
    fai_content = "chr1\t100\t0\t0\t0\nchr2\t200\t0\t0\t0\nchr3\t300\t0\t0\t0\n"
    fai_path.write_text(fai_content)

    # Only keep chr1 and chr3
    allowed = ["chr1", "chr3"]
    genome = create_genome_config(
        "test", fasta_path, species="human", allowed_chroms=allowed
    )

    assert genome.chrom_sizes is not None
    assert len(genome.chrom_sizes) == 2
    assert "chr1" in genome.chrom_sizes
    assert "chr3" in genome.chrom_sizes
    assert "chr2" not in genome.chrom_sizes


def test_genome_human_defaults(tmp_path):
    # Verify that species="human" automatically applies standard chromosomes
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")
    fai_path = tmp_path / "test.fa.fai"

    lines = [f"chr{c}\t100\t0\t0\t0" for c in range(1, 23)]
    lines.extend(
        [
            "chrX\t100\t0\t0\t0",
            "chrY\t100\t0\t0\t0",
            "chrM\t100\t0\t0\t0",
            "chrUn\t100\t0\t0\t0",
        ]
    )
    fai_path.write_text("\n".join(lines))

    genome = create_genome_config("test", fasta_path, species="human")

    assert genome.chrom_sizes is not None
    assert len(genome.chrom_sizes) == 24
    assert "chrM" not in genome.chrom_sizes
    assert "chrUn" not in genome.chrom_sizes
    assert "chr1" in genome.chrom_sizes
    assert "chrX" in genome.chrom_sizes


def test_genome_sorting_human(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")
    fai_path = tmp_path / "test.fa.fai"

    lines = [
        "chrM\t100\t0\t0\t0",
        "chr10\t100\t0\t0\t0",
        "chr1\t100\t0\t0\t0",
        "chrY\t100\t0\t0\t0",
        "chrUn\t100\t0\t0\t0",
        "chr2\t100\t0\t0\t0",
        "chrX\t100\t0\t0\t0",
    ]
    fai_path.write_text("\n".join(lines))

    allowed = ["chrM", "chr10", "chr1", "chrY", "chrUn", "chr2", "chrX"]

    genome = create_genome_config(
        "test", fasta_path, species="human", allowed_chroms=allowed
    )

    assert genome.chrom_sizes is not None
    keys = list(genome.chrom_sizes.keys())

    # Expected human sort order
    expected = ["chr1", "chr2", "chr10", "chrX", "chrY", "chrM", "chrUn"]
    assert keys == expected


def test_genome_mouse_defaults(tmp_path):
    # Verify that species="mouse" automatically applies standard mouse chromosomes (1-19, X, Y)
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")
    fai_path = tmp_path / "test.fa.fai"

    # Create FAI with mouse and some extras
    lines = [
        f"chr{c}\t100\t0\t0\t0" for c in range(1, 23)
    ]  # 1-22 (human range, so 20-22 are extra for mouse)
    lines.extend(["chrX\t100\t0\t0\t0", "chrY\t100\t0\t0\t0", "chrM\t100\t0\t0\t0"])
    fai_path.write_text("\n".join(lines))

    genome = create_genome_config("test", fasta_path, species="mouse")

    assert genome.chrom_sizes is not None
    # 1-19 + X + Y = 21 chromosomes
    assert len(genome.chrom_sizes) == 21
    assert "chr1" in genome.chrom_sizes
    assert "chr19" in genome.chrom_sizes
    assert "chrX" in genome.chrom_sizes
    assert "chrY" in genome.chrom_sizes
    assert "chrM" not in genome.chrom_sizes
    assert "chr20" not in genome.chrom_sizes


def test_genome_sorting_mouse(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text("")
    fai_path = tmp_path / "test.fa.fai"

    lines = [
        "chrM\t100\t0\t0\t0",
        "chr19\t100\t0\t0\t0",
        "chr1\t100\t0\t0\t0",
        "chrY\t100\t0\t0\t0",
        "chr20\t100\t0\t0\t0",
        "chrX\t100\t0\t0\t0",
    ]
    fai_path.write_text("\n".join(lines))

    allowed = ["chrM", "chr19", "chr1", "chrY", "chr20", "chrX"]

    genome = create_genome_config(
        "test", fasta_path, species="mouse", allowed_chroms=allowed
    )

    assert genome.chrom_sizes is not None
    keys = list(genome.chrom_sizes.keys())

    # Expected mouse sort order: 1, 19, X, Y, M, 20 (other)
    expected = ["chr1", "chr19", "chrX", "chrY", "chrM", "chr20"]
    assert keys == expected


def test_encode_dna():
    seq = "ACGTN"
    encoded = encode_dna(seq)
    assert encoded.shape == (4, 5)
    # A -> 0 -> [1,0,0,0]
    assert torch.all(encoded[:, 0] == torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # C -> 1 -> [0,1,0,0]
    assert torch.all(encoded[:, 1] == torch.tensor([0.0, 1.0, 0.0, 0.0]))
    # G -> 2 -> [0,0,1,0]
    assert torch.all(encoded[:, 2] == torch.tensor([0.0, 0.0, 1.0, 0.0]))
    # T -> 3 -> [0,0,0,1]
    assert torch.all(encoded[:, 3] == torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # N -> -1 -> [0,0,0,0]
    assert torch.all(encoded[:, 4] == torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_sequence_extractor(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n>chr2\nGGGG")

    extractor = SequenceExtractor(fasta_path)

    # Test valid extraction
    interval = Interval("chr1", 0, 4)
    encoded = extractor.extract(interval)
    assert encoded.shape == (4, 4)
    # ACGT
    assert torch.all(encoded[:, 0] == torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Test different region
    interval = Interval("chr1", 4, 8)
    encoded = extractor.extract(interval)
    assert encoded.shape == (4, 4)
    # ACGT
    assert torch.all(encoded[:, 0] == torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Test padding (truncation by file end) - now expect truncated output
    # chr2 has 4 Gs. Request 5.
    interval = Interval("chr2", 0, 5)
    encoded = extractor.extract(interval)
    assert encoded.shape == (4, 4)


def test_in_memory_sequence_extractor(tmp_path):
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text(">chr1\nACGTACGT\n")  # Length 8

    extractor = InMemorySequenceExtractor(fasta_path)

    # Normal extraction
    interval = Interval("chr1", 0, 4)
    out = extractor.extract(interval)  # ACGT
    assert out.shape == (4, 4)
    assert torch.all(out[:, 0] == torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Out of bounds right (truncated)
    interval = Interval("chr1", 6, 10)  # 2 bases valid (GT), 2 out.
    out = extractor.extract(interval)
    assert out.shape == (4, 2)
    # 6: G -> col 0
    assert torch.all(out[:, 0] == torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # Out of bounds left (truncated) - checking handling of start < 0
    # Note: IntervalSampler guarantees start >= 0, so this is technically invalid input.
    # Python slicing negative start wraps to end.
    # cached[:, -2:2] -> cached[:, 6:2] -> empty.
    interval = Interval("chr1", -2, 2)
    out = extractor.extract(interval)
    # Expect empty because negative index wraps
    assert out.shape == (4, 0)
