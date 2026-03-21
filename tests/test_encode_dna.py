"""Tests for encode_dna against the original LUT-scatter implementation."""

import numpy as np
import pytest
import torch

from cerberus.sequence import encode_dna


# ---------------------------------------------------------------------------
# Gold-standard reference: the original LUT-scatter implementation
# ---------------------------------------------------------------------------

def _create_mapping(encoding: str) -> np.ndarray:
    mapping = np.zeros(256, dtype=np.int8) - 1
    for i, base in enumerate(encoding):
        mapping[ord(base)] = i
    return mapping


_ENCODING_MAPPINGS = {"ACGT": _create_mapping("ACGT"), "AGCT": _create_mapping("AGCT")}


def encode_dna_reference(sequence: str, encoding: str = "ACGT") -> torch.Tensor:
    """Original LUT-scatter encode_dna kept as gold-standard reference."""
    sequence = sequence.upper()
    encoding = encoding.upper()
    mapping = _ENCODING_MAPPINGS[encoding]
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    indices = mapping[seq_bytes]
    one_hot = np.zeros((4, len(sequence)), dtype=np.float32)
    valid_mask = indices >= 0
    one_hot[indices[valid_mask], np.where(valid_mask)[0]] = 1.0
    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASES = "ACGT"
ENCODINGS = ["ACGT", "AGCT"]


def _random_dna(length: int, rng: np.random.RandomState, ambiguity_rate: float = 0.05) -> str:
    """Generate a random DNA string with occasional ambiguous bases."""
    bases = list(BASES) + ["N"]
    weights = [(1.0 - ambiguity_rate) / 4] * 4 + [ambiguity_rate]
    chars = rng.choice(bases, size=length, p=weights)
    return "".join(chars)


@pytest.fixture()
def random_sequences() -> list[str]:
    """10 random sequences of length 50-100."""
    rng = np.random.RandomState(12345)
    lengths = rng.randint(50, 101, size=10)
    return [_random_dna(int(l), rng) for l in lengths]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncodeDnaCorrectness:
    """Verify encode_dna matches the gold-standard reference on random inputs."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_random_sequences(self, random_sequences: list[str], encoding: str) -> None:
        for seq in random_sequences:
            result = encode_dna(seq, encoding=encoding)
            expected = encode_dna_reference(seq, encoding=encoding)
            assert result.shape == expected.shape, (
                f"Shape mismatch for len={len(seq)}: {result.shape} vs {expected.shape}"
            )
            assert torch.equal(result, expected), (
                f"Output mismatch for encoding={encoding}, seq={seq[:20]}..."
            )

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_case_insensitivity(self, encoding: str) -> None:
        seq = "AcGtNaCgT"
        assert torch.equal(encode_dna(seq, encoding), encode_dna_reference(seq, encoding))

    def test_all_n(self) -> None:
        result = encode_dna("NNNNN")
        assert result.shape == (4, 5)
        assert (result == 0).all()

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_single_bases(self, encoding: str) -> None:
        for base in "ACGTN":
            result = encode_dna(base, encoding=encoding)
            expected = encode_dna_reference(base, encoding=encoding)
            assert result.shape == (4, 1)
            assert torch.equal(result, expected)

    def test_empty_string(self) -> None:
        result = encode_dna("")
        expected = encode_dna_reference("")
        assert result.shape == (4, 0)
        assert torch.equal(result, expected)

    def test_dtype_and_contiguity(self, random_sequences: list[str]) -> None:
        for seq in random_sequences:
            result = encode_dna(seq)
            assert result.dtype == torch.float32
            assert result.is_contiguous()

    def test_one_hot_rows_sum_le_one(self, random_sequences: list[str]) -> None:
        """Each position should have at most one channel active."""
        for seq in random_sequences:
            result = encode_dna(seq)
            col_sums = result.sum(dim=0)
            assert (col_sums <= 1.0).all()

    def test_unsupported_encoding(self) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding"):
            encode_dna("ACGT", encoding="XYZW")
