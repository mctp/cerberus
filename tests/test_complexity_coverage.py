"""Coverage tests for cerberus.complexity — untested code paths."""
import pytest
import numpy as np
from cerberus.complexity import (
    calculate_dust_score,
    calculate_log_cpg_ratio,
    get_bin_index,
)


# ---------------------------------------------------------------------------
# calculate_dust_score edge cases
# ---------------------------------------------------------------------------

class TestDustScoreEdgeCases:

    def test_k_less_than_1_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            calculate_dust_score("ACGT", k=0)

    def test_k_greater_than_5_raises(self):
        with pytest.raises(ValueError, match="k must be <= 5"):
            calculate_dust_score("ACGT", k=6)

    def test_k_equals_1(self):
        score = calculate_dust_score("AAAA", k=1)
        assert score >= 0.0

    def test_short_sequence_returns_zero(self):
        """Sequence shorter than k returns 0.0."""
        score = calculate_dust_score("AC", k=3)
        assert score == 0.0

    def test_repetitive_sequence_high_score(self):
        """Highly repetitive sequence should have high DUST score."""
        score = calculate_dust_score("A" * 100, k=3, normalize=True)
        assert score > 0.5

    def test_non_normalized(self):
        score = calculate_dust_score("ACGTACGT", k=3, normalize=False)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# calculate_log_cpg_ratio edge cases
# ---------------------------------------------------------------------------

class TestLogCpgRatioEdgeCases:

    def test_length_less_than_2_normalized(self):
        result = calculate_log_cpg_ratio("A", normalize=True)
        assert result == 0.5

    def test_length_less_than_2_not_normalized(self):
        result = calculate_log_cpg_ratio("A", normalize=False)
        assert result == 0.0

    def test_empty_string_normalized(self):
        result = calculate_log_cpg_ratio("", normalize=True)
        assert result == 0.5

    def test_cpg_rich_sequence(self):
        """CpG-rich sequence should have ratio > 0.5 when normalized."""
        seq = "CGCGCGCGCGCGCGCG"
        result = calculate_log_cpg_ratio(seq, normalize=True)
        assert result > 0.5

    def test_cpg_depleted_sequence(self):
        """Sequence with G and C but no CpG dinucleotides should have ratio < 0.5."""
        # Has C and G but no CG dinucleotide (G always before C, never C before G)
        seq = "GCAGCAGCAGCAGCAGCAGCA"
        result = calculate_log_cpg_ratio(seq, normalize=True)
        assert result < 0.5


# ---------------------------------------------------------------------------
# get_bin_index edge cases
# ---------------------------------------------------------------------------

class TestGetBinIndex:

    def test_nan_values_returns_none(self):
        row = np.array([0.5, np.nan, 0.3])
        result = get_bin_index(row, bins=10)
        assert result is None

    def test_all_nan_returns_none(self):
        row = np.array([np.nan, np.nan])
        result = get_bin_index(row, bins=5)
        assert result is None

    def test_valid_values(self):
        row = np.array([0.5, 0.3])
        result = get_bin_index(row, bins=10)
        assert result == (5, 3)

    def test_boundary_values_clamped(self):
        row = np.array([1.0, 0.0])
        result = get_bin_index(row, bins=10)
        # 1.0 * 10 = 10, clamped to 9
        assert result == (9, 0)

    def test_negative_values_clamped(self):
        row = np.array([-0.1])
        result = get_bin_index(row, bins=10)
        assert result == (0,)
