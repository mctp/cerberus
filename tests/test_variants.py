from pathlib import Path

import pyfaidx
import pytest
import torch

from cerberus.interval import Interval
from cerberus.output import (
    FactorizedProfileCountOutput,
    ProfileCountOutput,
    ProfileLogRates,
)
from cerberus.variants import (
    Variant,
    _interval_to_region,
    compute_variant_effects,
    generate_variants,
    load_variants,
    load_vcf,
    variant_to_ref_alt,
)

FIXTURES = Path(__file__).parent / "data" / "fixtures"
VCF_PATH = FIXTURES / "test_variants.vcf"
VCF_GZ_PATH = FIXTURES / "test_variants.vcf.gz"
TSV_PATH = FIXTURES / "test_variants.tsv"
TSV_NOID_PATH = FIXTURES / "test_variants_noid.tsv"
TSV_0BASED_PATH = FIXTURES / "test_variants_0based.tsv"
FASTA_PATH = FIXTURES / "test_variants.fa"


# ── Construction ──────────────────────────────────────────────────────


class TestConstruction:
    def test_snp(self):
        v = Variant("chr1", 100, "A", "G")
        assert v.chrom == "chr1"
        assert v.pos == 100
        assert v.ref == "A"
        assert v.alt == "G"
        assert v.id == "."
        assert v.info == {}

    def test_deletion(self):
        v = Variant("chr1", 100, "ACGT", "A")
        assert v.ref == "ACGT"
        assert v.alt == "A"

    def test_insertion(self):
        v = Variant("chr1", 100, "A", "ACGT")
        assert v.ref == "A"
        assert v.alt == "ACGT"

    def test_with_id(self):
        v = Variant("chr1", 100, "A", "G", id="rs12345")
        assert v.id == "rs12345"

    def test_with_info(self):
        info = {"AF": 0.05, "DP": 100, "DB": True}
        v = Variant("chr1", 100, "A", "G", info=info)
        assert v.info["AF"] == 0.05
        assert v.info["DP"] == 100
        assert v.info["DB"] is True

    def test_info_default_not_shared(self):
        """Each Variant gets its own empty dict, not a shared mutable default."""
        v1 = Variant("chr1", 100, "A", "G")
        v2 = Variant("chr1", 200, "C", "T")
        assert v1.info is not v2.info


# ── Validation ────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_ref_raises(self):
        with pytest.raises(ValueError, match="ref allele must be non-empty"):
            Variant("chr1", 100, "", "A")

    def test_empty_alt_raises(self):
        with pytest.raises(ValueError, match="alt allele must be non-empty"):
            Variant("chr1", 100, "A", "")

    def test_negative_pos_raises(self):
        with pytest.raises(ValueError, match="pos must be non-negative"):
            Variant("chr1", -1, "A", "G")

    def test_ref_equals_alt_raises(self):
        with pytest.raises(ValueError, match="ref and alt alleles must differ"):
            Variant("chr1", 100, "A", "A")

    def test_pos_zero_is_valid(self):
        v = Variant("chr1", 0, "A", "G")
        assert v.pos == 0


# ── Frozen ────────────────────────────────────────────────────────────


class TestFrozen:
    def test_cannot_mutate_pos(self):
        v = Variant("chr1", 100, "A", "G")
        with pytest.raises(AttributeError):
            v.pos = 200  # type: ignore[misc]

    def test_cannot_mutate_ref(self):
        v = Variant("chr1", 100, "A", "G")
        with pytest.raises(AttributeError):
            v.ref = "C"  # type: ignore[misc]

    def test_cannot_mutate_id(self):
        v = Variant("chr1", 100, "A", "G")
        with pytest.raises(AttributeError):
            v.id = "rs1"  # type: ignore[misc]

    def test_hashable(self):
        """Frozen dataclasses are hashable and can be used in sets."""
        v1 = Variant("chr1", 100, "A", "G")
        v2 = Variant("chr1", 100, "A", "G")
        v3 = Variant("chr1", 100, "A", "T")
        assert hash(v1) == hash(v2)
        assert v1 == v2
        assert {v1, v2, v3} == {v1, v3}


# ── Coordinate properties ────────────────────────────────────────────


class TestCoordinates:
    def test_end_snp(self):
        v = Variant("chr1", 100, "A", "G")
        assert v.end == 101  # half-open: [100, 101)

    def test_end_deletion(self):
        v = Variant("chr1", 100, "ACGT", "A")
        assert v.end == 104  # half-open: [100, 104)

    def test_end_insertion(self):
        v = Variant("chr1", 100, "A", "ACGT")
        assert v.end == 101  # ref footprint is still [100, 101)

    def test_ref_len(self):
        assert Variant("chr1", 0, "ACGT", "A").ref_len == 4
        assert Variant("chr1", 0, "A", "G").ref_len == 1

    def test_alt_len(self):
        assert Variant("chr1", 0, "A", "ACGT").alt_len == 4
        assert Variant("chr1", 0, "A", "G").alt_len == 1

    def test_ref_center_snp(self):
        v = Variant("chr1", 100, "A", "G")
        # (100 + 101) // 2 = 100
        assert v.ref_center == 100

    def test_ref_center_even_deletion(self):
        v = Variant("chr1", 100, "ACGT", "A")
        # (100 + 104) // 2 = 102
        assert v.ref_center == 102

    def test_ref_center_odd_deletion(self):
        v = Variant("chr1", 100, "ACG", "A")
        # (100 + 103) // 2 = 101
        assert v.ref_center == 101

    def test_ref_center_insertion(self):
        v = Variant("chr1", 100, "A", "ACGTACGT")
        # Ref footprint is [100, 101), center = 100
        assert v.ref_center == 100


# ── Variant type classification ──────────────────────────────────────


class TestClassification:
    def test_snp(self):
        v = Variant("chr1", 100, "A", "G")
        assert v.is_snp is True
        assert v.is_insertion is False
        assert v.is_deletion is False
        assert v.size_change == 0

    def test_insertion(self):
        v = Variant("chr1", 100, "A", "ACGT")
        assert v.is_snp is False
        assert v.is_insertion is True
        assert v.is_deletion is False
        assert v.size_change == 3

    def test_deletion(self):
        v = Variant("chr1", 100, "ACGT", "A")
        assert v.is_snp is False
        assert v.is_insertion is False
        assert v.is_deletion is True
        assert v.size_change == -3

    def test_mnp(self):
        """Multi-nucleotide polymorphism: same length, not SNP."""
        v = Variant("chr1", 100, "AC", "GT")
        assert v.is_snp is False
        assert v.is_insertion is False
        assert v.is_deletion is False
        assert v.size_change == 0

    def test_complex_insertion(self):
        """Ref has >1 base but alt is longer — still an insertion."""
        v = Variant("chr1", 100, "AC", "ACGT")
        assert v.is_insertion is True
        assert v.size_change == 2

    def test_complex_deletion(self):
        """Alt has >1 base but ref is longer — still a deletion."""
        v = Variant("chr1", 100, "ACGT", "AC")
        assert v.is_deletion is True
        assert v.size_change == -2


# ── to_interval ──────────────────────────────────────────────────────


class TestToInterval:
    def test_snp_interval(self):
        v = Variant("chr1", 100, "A", "G")
        iv = v.to_interval()
        assert iv == Interval("chr1", 100, 101)
        assert len(iv) == 1

    def test_deletion_interval(self):
        v = Variant("chr1", 100, "ACGT", "A")
        iv = v.to_interval()
        assert iv == Interval("chr1", 100, 104)
        assert len(iv) == 4

    def test_insertion_interval(self):
        """Insertion footprint in the reference is just the anchor base(s)."""
        v = Variant("chr1", 100, "A", "ACGT")
        iv = v.to_interval()
        assert iv == Interval("chr1", 100, 101)
        assert len(iv) == 1

    def test_interval_strand_is_default(self):
        v = Variant("chr1", 100, "A", "G")
        assert v.to_interval().strand == "+"


# ── String representation ────────────────────────────────────────────


class TestStringRepresentation:
    def test_str_snp(self):
        v = Variant("chr1", 100, "A", "G")
        assert str(v) == "chr1:100:A>G"

    def test_str_deletion(self):
        v = Variant("chr1", 100, "ACGT", "A")
        assert str(v) == "chr1:100:ACGT>A"

    def test_str_insertion(self):
        v = Variant("chr1", 100, "A", "ACGT")
        assert str(v) == "chr1:100:A>ACGT"

    def test_from_str_snp(self):
        v = Variant.from_str("chr1:100:A>G")
        assert v == Variant("chr1", 100, "A", "G")

    def test_from_str_deletion(self):
        v = Variant.from_str("chr1:100:ACGT>A")
        assert v == Variant("chr1", 100, "ACGT", "A")

    def test_from_str_roundtrip(self):
        original = Variant("chrX", 50000, "AC", "ACGT", id="rs999")
        # from_str does not preserve id — only core fields
        parsed = Variant.from_str(str(original))
        assert parsed.chrom == original.chrom
        assert parsed.pos == original.pos
        assert parsed.ref == original.ref
        assert parsed.alt == original.alt
        assert parsed.id == "."  # default, not preserved

    def test_from_str_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid variant string"):
            Variant.from_str("chr1-100-A-G")

    def test_from_str_missing_alleles(self):
        with pytest.raises(ValueError, match="Invalid variant string"):
            Variant.from_str("chr1:100")

    def test_from_str_propagates_validation(self):
        """from_str triggers __post_init__ validation."""
        with pytest.raises(ValueError, match="ref and alt alleles must differ"):
            Variant.from_str("chr1:100:A>A")


# ── Equality ─────────────────────────────────────────────────────────


class TestEquality:
    def test_equal_variants(self):
        v1 = Variant("chr1", 100, "A", "G")
        v2 = Variant("chr1", 100, "A", "G")
        assert v1 == v2

    def test_different_pos(self):
        v1 = Variant("chr1", 100, "A", "G")
        v2 = Variant("chr1", 101, "A", "G")
        assert v1 != v2

    def test_different_alt(self):
        v1 = Variant("chr1", 100, "A", "G")
        v2 = Variant("chr1", 100, "A", "T")
        assert v1 != v2

    def test_different_id_still_not_equal(self):
        """id is part of the dataclass fields, so different id means not equal."""
        v1 = Variant("chr1", 100, "A", "G", id="rs1")
        v2 = Variant("chr1", 100, "A", "G", id="rs2")
        assert v1 != v2

    def test_same_with_info(self):
        """info dicts are compared by value."""
        v1 = Variant("chr1", 100, "A", "G", info={"AF": 0.5})
        v2 = Variant("chr1", 100, "A", "G", info={"AF": 0.5})
        assert v1 == v2


# ── variant_to_ref_alt ───────────────────────────────────────────────

# test_variants.fa:
#   chr1: ACGTACGTACGT... (104 bp, repeating ACGT)
#   chr2: TTTTTTTTTTAAAAAAAAAA (20 bp)


@pytest.fixture()
def fasta():
    return pyfaidx.Fasta(str(FASTA_PATH))


class TestVariantToRefAlt:
    """Tests for variant_to_ref_alt using tests/data/fixtures/test_variants.fa.

    chr1 = "ACGTACGT..." (104 bp repeating ACGT pattern)
        pos % 4: 0=A, 1=C, 2=G, 3=T
    chr2 = "TTTTTTTTTTAAAAAAAAAA" (20 bp)
    """

    def test_snp_shapes(self, fasta):
        """Both tensors are (4, input_len)."""
        # pos 48 = 'A' (48 % 4 = 0)
        v = Variant("chr1", 48, "A", "G")
        ref_t, alt_t, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)
        assert ref_t.dtype == torch.float32

    def test_snp_interval(self, fasta):
        """Returned interval matches the ref window centered on ref_center."""
        # pos 48, ref_center = 48, window = [48-10, 48+10) = [38, 58)
        v = Variant("chr1", 48, "A", "G")
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert iv == Interval("chr1", 38, 58)
        assert len(iv) == 20

    def test_snp_only_one_position_differs(self, fasta):
        """For a SNP, ref and alt tensors differ at exactly one position."""
        # pos 48 = 'A' → alt 'G'
        v = Variant("chr1", 48, "A", "G")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        diff = (ref_t != alt_t).any(dim=0)
        assert diff.sum().item() == 1
        # variant at pos 48, window starts at 38, offset = 48-38 = 10
        diff_pos = diff.nonzero(as_tuple=True)[0].item()
        assert diff_pos == 10

    def test_snp_alt_has_correct_base(self, fasta):
        """The alt tensor has the correct base at the variant position."""
        # pos 48 = 'A', alt = 'G'
        v = Variant("chr1", 48, "A", "G")
        _, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        # offset within window = 48 - 38 = 10
        # G is channel index 2 in ACGT encoding
        assert alt_t[2, 10].item() == 1.0
        assert alt_t[0, 10].item() == 0.0  # A channel should be 0

    def test_deletion_shapes(self, fasta):
        """Deletion produces correct shapes."""
        # pos 48: ref=ACGT (4bp), alt=A (1bp) → 3bp deletion
        v = Variant("chr1", 48, "ACGT", "A")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)

    def test_deletion_interval_centered_on_ref_center(self, fasta):
        """Deletion window is centered on midpoint of deleted span."""
        v = Variant("chr1", 48, "ACGT", "A")
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        # ref_center = (48 + 52) // 2 = 50
        assert iv == Interval("chr1", 40, 60)

    def test_insertion_shapes(self, fasta):
        """Insertion produces correct shapes."""
        # pos 50 = 'G', ref=G, alt=GGGG → 3bp insertion
        v = Variant("chr1", 50, "G", "GGGG")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)

    def test_insertion_interval_centered_on_ref_center(self, fasta):
        """Insertion window is centered on the anchor base."""
        # pos 50 = 'G', ref=G (1bp), ref_center = 50
        v = Variant("chr1", 50, "G", "GGGG")
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert iv == Interval("chr1", 40, 60)

    def test_insertion_alt_contains_inserted_bases(self, fasta):
        """The alt sequence actually contains the inserted allele."""
        # pos 48 = 'A', insert ACCC
        v = Variant("chr1", 48, "A", "ACCC")
        _, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        nuc = "ACGT"
        alt_str = ""
        for i in range(alt_t.shape[1]):
            idx = int(alt_t[:, i].argmax().item())
            alt_str += nuc[idx]
        assert "ACCC" in alt_str

    def test_ref_allele_mismatch_raises(self, fasta):
        """ValueError if FASTA doesn't match variant.ref."""
        # pos 48 in chr1 ACGT pattern is 'A', not 'T'
        v = Variant("chr1", 48, "T", "G")
        with pytest.raises(ValueError, match="Ref allele mismatch"):
            variant_to_ref_alt(v, fasta, input_len=20)

    def test_window_out_of_bounds_left_raises(self, fasta):
        """ValueError if window extends past chromosome start."""
        # pos 2 = 'G', window [2-10, 2+10) = [-8, 12)
        v = Variant("chr1", 2, "G", "A")
        with pytest.raises(ValueError, match="extends beyond"):
            variant_to_ref_alt(v, fasta, input_len=20)

    def test_window_out_of_bounds_right_raises(self, fasta):
        """ValueError if window extends past chromosome end."""
        # chr1 is 104bp; pos 100 = 'A', window [90,110) > 104
        v = Variant("chr1", 100, "A", "G")
        with pytest.raises(ValueError, match="extends beyond"):
            variant_to_ref_alt(v, fasta, input_len=20)

    def test_deletion_extended_context_out_of_bounds_raises(self, fasta):
        """ValueError if deletion requires FASTA context beyond chromosome."""
        # chr2 is 20bp, pos 10 = 'A', ref=AAAAAAAAAA (10bp), alt=A → 9bp del
        # ref_center = (10 + 20) // 2 = 15, window [5, 25) > chr2 len=20
        v = Variant("chr2", 10, "AAAAAAAAAA", "A")
        with pytest.raises(ValueError, match="extends beyond"):
            variant_to_ref_alt(v, fasta, input_len=20)

    def test_unknown_chrom_raises(self, fasta):
        """ValueError if chromosome not in FASTA."""
        v = Variant("chrZ", 50, "A", "G")
        with pytest.raises(ValueError, match="not found in FASTA"):
            variant_to_ref_alt(v, fasta, input_len=20)

    def test_snp_ref_tensor_matches_fasta(self, fasta):
        """Ref tensor encodes the exact FASTA sequence."""
        # pos 48 = 'A'
        v = Variant("chr1", 48, "A", "G")
        ref_t, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        expected_seq = str(fasta["chr1"][iv.start : iv.end]).upper()
        nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, base in enumerate(expected_seq):
            assert ref_t[nuc_map[base], i].item() == 1.0

    def test_mnp_substitution(self, fasta):
        """MNP (multi-nucleotide polymorphism) replaces multiple bases."""
        # pos 48-49 in chr1 is "AC", replace with "GG"
        v = Variant("chr1", 48, "AC", "GG")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == alt_t.shape
        diff = (ref_t != alt_t).any(dim=0)
        assert diff.sum().item() == 2  # exactly 2 positions differ

    def test_even_deletion_symmetric_trim(self, fasta):
        """Even-length deletion produces correct shapes."""
        # pos 48: ref=AC (2bp), alt=A (1bp) → 1bp deletion
        v = Variant("chr1", 48, "AC", "A")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)

    def test_large_insertion_symmetric_trim(self, fasta):
        """Large insertion trims from both ends."""
        # pos 48 = 'A', ref=A, alt=AGGGGGG → size_change=+6
        v = Variant("chr1", 48, "A", "AGGGGGG")
        ref_t, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)

    # -- Interval / size consistency --

    def test_snp_interval_length_matches_input_len(self, fasta):
        """Returned interval length always equals input_len."""
        v = Variant("chr1", 48, "A", "G")
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert len(iv) == 20

    def test_deletion_interval_length_matches_input_len(self, fasta):
        """Deletion: interval length equals input_len, not ref-allele-adjusted."""
        v = Variant("chr1", 48, "ACGT", "A")  # 3bp deletion
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert len(iv) == 20

    def test_insertion_interval_length_matches_input_len(self, fasta):
        """Insertion: interval length equals input_len."""
        v = Variant("chr1", 50, "G", "GGGG")  # 3bp insertion
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert len(iv) == 20

    def test_interval_contains_variant(self, fasta):
        """The returned interval must fully contain the variant ref footprint."""
        v = Variant("chr1", 48, "ACGT", "A")  # ref spans [48, 52)
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert iv.start <= v.pos
        assert iv.end >= v.end

    def test_interval_contains_variant_insertion(self, fasta):
        """Insertion: interval contains the anchor base."""
        v = Variant("chr1", 50, "G", "GGGG")  # ref spans [50, 51)
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert iv.start <= v.pos
        assert iv.end >= v.end

    def test_tensors_always_match_input_len(self, fasta):
        """Both tensors always have shape (4, input_len) regardless of variant type."""
        cases = [
            Variant("chr1", 48, "A", "G"),          # SNP
            Variant("chr1", 48, "AC", "GG"),         # MNP
            Variant("chr1", 48, "ACGT", "A"),        # deletion
            Variant("chr1", 48, "A", "AGGGGGG"),     # insertion
            Variant("chr1", 48, "ACGT", "GGGGG"),    # delins (longer alt)
            Variant("chr1", 48, "ACGTACGT", "GG"),   # delins (shorter alt)
        ]
        for v in cases:
            ref_t, alt_t, iv = variant_to_ref_alt(v, fasta, input_len=20)
            assert ref_t.shape == (4, 20), f"ref shape wrong for {v}"
            assert alt_t.shape == (4, 20), f"alt shape wrong for {v}"
            assert len(iv) == 20, f"interval length wrong for {v}"

    # -- Delins (multi-base ref AND multi-base alt, different lengths) --

    def test_delins_shorter_alt(self, fasta):
        """Delins where alt is shorter than ref (net deletion).

        pos 48: ref=ACGTACGT (8bp), alt=GG (2bp) → size_change=-6
        """
        v = Variant("chr1", 48, "ACGTACGT", "GG")
        assert v.size_change == -6
        ref_t, alt_t, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)
        # ref_center = (48 + 56) // 2 = 52
        assert iv.start <= v.pos
        assert iv.end >= v.end

    def test_delins_longer_alt(self, fasta):
        """Delins where alt is longer than ref (net insertion).

        pos 48: ref=ACGT (4bp), alt=GGGGGGG (7bp) → size_change=+3
        """
        v = Variant("chr1", 48, "ACGT", "GGGGGGG")
        assert v.size_change == 3
        ref_t, alt_t, iv = variant_to_ref_alt(v, fasta, input_len=20)
        assert ref_t.shape == (4, 20)
        assert alt_t.shape == (4, 20)
        assert iv.start <= v.pos
        assert iv.end >= v.end

    def test_delins_alt_present_in_sequence(self, fasta):
        """Delins: alt allele actually appears in the alt tensor.

        pos 48: ref=ACGT (4bp), alt=TTT (3bp) → size_change=-1
        """
        v = Variant("chr1", 48, "ACGT", "TTT")
        _, alt_t, _ = variant_to_ref_alt(v, fasta, input_len=20)
        nuc = "ACGT"
        alt_str = ""
        for i in range(alt_t.shape[1]):
            idx = int(alt_t[:, i].argmax().item())
            alt_str += nuc[idx]
        assert "TTT" in alt_str

    def test_delins_ref_center(self, fasta):
        """Delins: window is centered on midpoint of the multi-base ref."""
        # pos 48: ref=ACGTACGT (8bp), ref spans [48, 56)
        # ref_center = (48 + 56) // 2 = 52
        v = Variant("chr1", 48, "ACGTACGT", "GG")
        _, _, iv = variant_to_ref_alt(v, fasta, input_len=20)
        expected_center = 52
        assert iv.start == expected_center - 10
        assert iv.end == expected_center + 10


# ── _interval_to_region ──────────────────────────────────────────────


class TestIntervalToRegion:
    def test_basic(self):
        iv = Interval("chr1", 0, 100)
        assert _interval_to_region(iv) == "chr1:1-100"

    def test_nonzero_start(self):
        iv = Interval("chr1", 999, 2000)
        assert _interval_to_region(iv) == "chr1:1000-2000"

    def test_single_base(self):
        iv = Interval("chr1", 42, 43)
        assert _interval_to_region(iv) == "chr1:43-43"


# ── load_vcf ─────────────────────────────────────────────────────────


class TestLoadVcf:
    """Tests for load_vcf using tests/data/fixtures/test_variants.vcf(.gz).

    The test VCF contains 7 records:
      chr1:100 rs001  A>G    PASS         (SNP)
      chr1:200 rs002  C>T    PASS         (SNP)
      chr1:300 rs003  ACGT>A PASS         (deletion)
      chr1:400 rs004  A>ACGT PASS         (insertion)
      chr1:500 .      G>A    LowQual      (fails filter)
      chr1:600 rs006  A>G,T  PASS         (multi-allelic — skipped)
      chr2:100 rs007  T>C    PASS         (SNP, different chrom)
    """

    def test_load_all_pass(self):
        """Default pass_only=True: yields 5 PASS biallelic records."""
        variants = list(load_vcf(VCF_PATH))
        assert len(variants) == 5

    def test_coordinates_are_zero_based(self):
        """VCF POS is 1-based; Variant.pos should be 0-based."""
        variants = list(load_vcf(VCF_PATH))
        first = variants[0]
        assert first.pos == 99  # VCF POS=100 → 0-based 99
        assert first.chrom == "chr1"
        assert first.ref == "A"
        assert first.alt == "G"
        assert first.id == "rs001"

    def test_variant_types(self):
        """Check that SNPs, deletions, and insertions are parsed correctly."""
        variants = list(load_vcf(VCF_PATH))
        snp = variants[0]       # chr1:100 A>G
        deletion = variants[2]  # chr1:300 ACGT>A
        insertion = variants[3] # chr1:400 A>ACGT

        assert snp.is_snp
        assert deletion.is_deletion
        assert deletion.ref == "ACGT"
        assert deletion.alt == "A"
        assert insertion.is_insertion
        assert insertion.ref == "A"
        assert insertion.alt == "ACGT"

    def test_pass_only_false(self):
        """pass_only=False includes the LowQual record."""
        variants = list(load_vcf(VCF_PATH, pass_only=False))
        # 5 PASS biallelic + 1 LowQual = 6 (multi-allelic still skipped)
        assert len(variants) == 6
        low_qual = [v for v in variants if v.pos == 499]  # VCF POS=500
        assert len(low_qual) == 1

    def test_multi_allelic_skipped(self):
        """Multi-allelic records (>1 ALT) are silently skipped."""
        # pass_only=False to not confuse with filter skipping
        variants = list(load_vcf(VCF_PATH, pass_only=False))
        # rs006 at chr1:600 has ALT=G,T — should not appear
        at_600 = [v for v in variants if v.pos == 599]
        assert len(at_600) == 0

    def test_info_fields_none(self):
        """Default info_fields=None: info dict is empty."""
        variants = list(load_vcf(VCF_PATH))
        assert variants[0].info == {}

    def test_info_fields_captured(self):
        """Requested INFO fields are captured into info dict."""
        variants = list(load_vcf(VCF_PATH, info_fields=["AF", "DP"]))
        first = variants[0]
        assert pytest.approx(first.info["AF"], abs=1e-6) == 0.05
        assert first.info["DP"] == 100

    def test_info_fields_missing_key(self):
        """Requesting a non-existent INFO field doesn't error, just absent."""
        variants = list(load_vcf(VCF_PATH, info_fields=["NONEXISTENT"]))
        assert "NONEXISTENT" not in variants[0].info

    def test_region_with_interval(self):
        """Region filtering with a cerberus Interval (0-based half-open)."""
        # chr2 has one variant at VCF POS=100
        region = Interval("chr2", 0, 500000)
        variants = list(load_vcf(VCF_GZ_PATH, region=region))
        assert len(variants) == 1
        assert variants[0].chrom == "chr2"
        assert variants[0].pos == 99

    def test_region_with_string(self):
        """Region filtering with a tabix-style string (1-based inclusive)."""
        variants = list(load_vcf(VCF_GZ_PATH, region="chr2:1-500000"))
        assert len(variants) == 1
        assert variants[0].chrom == "chr2"

    def test_region_excludes_other_chrom(self):
        """Region on chr2 should not return chr1 variants."""
        region = Interval("chr2", 0, 500000)
        variants = list(load_vcf(VCF_GZ_PATH, region=region))
        assert all(v.chrom == "chr2" for v in variants)

    def test_region_narrow(self):
        """Narrow region includes only overlapping variants."""
        # Only chr1 POS=100 (0-based 99) and POS=200 (0-based 199) should match
        region = Interval("chr1", 90, 210)
        variants = list(load_vcf(VCF_GZ_PATH, region=region))
        assert len(variants) == 2
        assert variants[0].pos == 99
        assert variants[1].pos == 199

    def test_is_generator(self):
        """load_vcf returns a generator, not a list."""
        result = load_vcf(VCF_PATH)
        assert hasattr(result, "__next__")

    def test_missing_id_becomes_dot(self):
        """Records with no ID (.) get id='.'."""
        variants = list(load_vcf(VCF_PATH, pass_only=False))
        no_id = [v for v in variants if v.pos == 499]
        assert len(no_id) == 1
        assert no_id[0].id == "."

    def test_different_chroms(self):
        """Variants from different chromosomes are both yielded."""
        variants = list(load_vcf(VCF_PATH))
        chroms = {v.chrom for v in variants}
        assert chroms == {"chr1", "chr2"}


# ── load_variants ───────────────────────────────────────────────────


class TestLoadVariants:
    """Tests for load_variants using tests/data/fixtures/test_variants*.tsv.

    test_variants.tsv (1-based, with id column):
      chr1:100 rs001 A>G, chr1:200 rs002 C>T, chr1:300 rs003 ACGT>A,
      chr1:400 rs004 A>ACGT, chr2:100 rs007 T>C

    test_variants_noid.tsv (1-based, no id column):
      chr1:100 A>G, chr1:200 C>T

    test_variants_0based.tsv (0-based, with id column):
      chr1:99 rs001 A>G, chr1:199 rs002 C>T, chr2:99 rs007 T>C
    """

    def test_load_all(self):
        """Loads all 5 variants from the TSV."""
        variants = list(load_variants(TSV_PATH))
        assert len(variants) == 5

    def test_coordinates_are_zero_based(self):
        """1-based TSV pos is converted to 0-based Variant.pos."""
        variants = list(load_variants(TSV_PATH))
        first = variants[0]
        assert first.pos == 99  # TSV pos=100 → 0-based 99
        assert first.chrom == "chr1"
        assert first.ref == "A"
        assert first.alt == "G"
        assert first.id == "rs001"

    def test_variant_types(self):
        """SNPs, deletions, and insertions are parsed correctly."""
        variants = list(load_variants(TSV_PATH))
        snp = variants[0]
        deletion = variants[2]
        insertion = variants[3]

        assert snp.is_snp
        assert deletion.is_deletion
        assert deletion.ref == "ACGT"
        assert deletion.alt == "A"
        assert insertion.is_insertion
        assert insertion.ref == "A"
        assert insertion.alt == "ACGT"

    def test_id_column_captured(self):
        """The id column is captured into Variant.id."""
        variants = list(load_variants(TSV_PATH))
        assert variants[0].id == "rs001"
        assert variants[4].id == "rs007"

    def test_no_id_column(self):
        """Without an id column, Variant.id defaults to '.'."""
        variants = list(load_variants(TSV_NOID_PATH))
        assert len(variants) == 2
        assert variants[0].id == "."
        assert variants[1].id == "."

    def test_zero_based_flag(self):
        """zero_based=True skips the subtraction."""
        variants = list(load_variants(TSV_0BASED_PATH, zero_based=True))
        assert variants[0].pos == 99
        assert variants[1].pos == 199
        assert variants[2].pos == 99

    def test_zero_based_matches_default(self):
        """0-based file with zero_based=True gives same result as 1-based default."""
        from_1based = list(load_variants(TSV_PATH))
        from_0based = list(load_variants(TSV_0BASED_PATH, zero_based=True))
        # Compare just the first variant (rs001)
        assert from_1based[0].pos == from_0based[0].pos
        assert from_1based[0].chrom == from_0based[0].chrom
        assert from_1based[0].ref == from_0based[0].ref
        assert from_1based[0].alt == from_0based[0].alt

    def test_different_chroms(self):
        """Variants from different chromosomes are both yielded."""
        variants = list(load_variants(TSV_PATH))
        chroms = {v.chrom for v in variants}
        assert chroms == {"chr1", "chr2"}

    def test_is_generator(self):
        """load_variants returns a generator, not a list."""
        result = load_variants(TSV_PATH)
        assert hasattr(result, "__next__")

    def test_comment_lines_skipped(self, tmp_path):
        """Lines starting with # in the body are skipped."""
        f = tmp_path / "commented.tsv"
        f.write_text("chrom\tpos\tref\talt\n# this is a comment\nchr1\t100\tA\tG\n")
        variants = list(load_variants(f))
        assert len(variants) == 1

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in the body are skipped."""
        f = tmp_path / "blanks.tsv"
        f.write_text("chrom\tpos\tref\talt\nchr1\t100\tA\tG\n\nchr1\t200\tC\tT\n")
        variants = list(load_variants(f))
        assert len(variants) == 2

    def test_hash_header(self, tmp_path):
        """Header starting with '#' is accepted (leading '#' stripped)."""
        f = tmp_path / "hash_header.tsv"
        f.write_text("#chrom\tpos\tref\talt\nchr1\t100\tA\tG\n")
        variants = list(load_variants(f))
        assert len(variants) == 1
        assert variants[0].pos == 99

    def test_missing_column_raises(self, tmp_path):
        """Missing required columns raise ValueError."""
        f = tmp_path / "bad.tsv"
        f.write_text("chrom\tpos\tref\n")
        with pytest.raises(ValueError, match="Missing required column"):
            list(load_variants(f))

    def test_columns_any_order(self, tmp_path):
        """Columns can appear in any order."""
        f = tmp_path / "reordered.tsv"
        f.write_text("alt\tref\tpos\tchrom\nchr1\tG\tA\t100\n")
        # With this reordering: alt=chr1, ref=G, pos=A, chrom=100 — wrong!
        # Let me fix the test data to match the column order
        f.write_text("alt\tref\tpos\tchrom\nG\tA\t100\tchr1\n")
        variants = list(load_variants(f))
        assert len(variants) == 1
        assert variants[0].chrom == "chr1"
        assert variants[0].pos == 99
        assert variants[0].ref == "A"
        assert variants[0].alt == "G"

    def test_extra_columns_ignored(self, tmp_path):
        """Extra columns beyond the required ones are silently ignored."""
        f = tmp_path / "extra.tsv"
        f.write_text("chrom\tpos\tref\talt\tAF\tDP\nchr1\t100\tA\tG\t0.05\t100\n")
        variants = list(load_variants(f))
        assert len(variants) == 1
        assert variants[0] == Variant("chr1", 99, "A", "G")


# ── generate_variants ────────────────────────────────────────────────


class TestGenerateVariants:
    """Tests for generate_variants saturation mutagenesis generator."""

    @pytest.fixture()
    def fasta(self):
        fa = pyfaidx.Fasta(str(FASTA_PATH))
        yield fa
        fa.close()

    # -- SNVs only (default) --

    def test_snv_count(self, fasta):
        """4bp interval on repeating ACGT → 4 positions × 3 alts = 12 SNVs."""
        iv = Interval("chr1", 0, 4)
        variants = list(generate_variants(iv, fasta))
        assert len(variants) == 12

    def test_snv_ref_alleles(self, fasta):
        """Ref alleles match the FASTA sequence."""
        iv = Interval("chr1", 0, 4)  # ACGT
        variants = list(generate_variants(iv, fasta))
        # Group by position
        by_pos = {}
        for v in variants:
            by_pos.setdefault(v.pos, []).append(v)
        assert all(v.ref == "A" for v in by_pos[0])
        assert all(v.ref == "C" for v in by_pos[1])
        assert all(v.ref == "G" for v in by_pos[2])
        assert all(v.ref == "T" for v in by_pos[3])

    def test_snv_alt_alleles(self, fasta):
        """Each position has exactly 3 non-ref alt alleles."""
        iv = Interval("chr1", 0, 1)  # A
        variants = list(generate_variants(iv, fasta))
        alts = {v.alt for v in variants}
        assert alts == {"C", "G", "T"}

    def test_snv_all_are_snps(self, fasta):
        iv = Interval("chr1", 0, 8)
        for v in generate_variants(iv, fasta):
            assert v.is_snp

    def test_positional_order(self, fasta):
        """Variants are yielded in positional order."""
        iv = Interval("chr1", 0, 8)
        variants = list(generate_variants(iv, fasta))
        positions = [v.pos for v in variants]
        assert positions == sorted(positions)

    def test_is_generator(self, fasta):
        result = generate_variants(Interval("chr1", 0, 4), fasta)
        assert hasattr(result, "__next__")

    # -- Deletions --

    def test_deletions_size_1(self, fasta):
        """max_indel_size=1 adds 1 deletion per position (if room in fetched seq)."""
        iv = Interval("chr1", 0, 4)  # ACGT — fetched seq is 4bp
        variants = list(generate_variants(iv, fasta, max_indel_size=1))
        deletions = [v for v in variants if v.is_deletion]
        # pos 0: AC→A, pos 1: CG→C, pos 2: GT→G
        # pos 3 (T): needs pos 4 which is past the fetched sequence end → no deletion
        assert len(deletions) == 3
        for d in deletions:
            assert len(d.ref) == 2
            assert len(d.alt) == 1
            assert d.alt == d.ref[0]  # anchor base

    def test_deletions_size_2(self, fasta):
        """max_indel_size=2 adds up to 2 deletions per position."""
        iv = Interval("chr1", 0, 4)  # ACGT — fetched seq is 4bp
        variants = list(generate_variants(iv, fasta, max_indel_size=2))
        deletions = [v for v in variants if v.is_deletion]
        # pos 0: del 1 (AC→A), del 2 (ACG→A) = 2
        # pos 1: del 1 (CG→C), del 2 (CGT→C) = 2
        # pos 2: del 1 (GT→G), del 2 needs pos 5 → out of bounds = 1
        # pos 3: del 1 needs pos 5 → out of bounds = 0
        assert len(deletions) == 5

    def test_deletion_at_sequence_end(self, fasta):
        """Deletion that would extend past the fetched sequence is not yielded."""
        # chr1 is 104bp; interval ends at 104
        iv = Interval("chr1", 102, 104)  # last 2 bases: GT
        variants = list(generate_variants(iv, fasta, max_indel_size=2))
        deletions = [v for v in variants if v.is_deletion]
        # pos 102 (G): del 1 (GT→G) ok, del 2 would need pos 104 → out of bounds
        # pos 103 (T): del 1 would need pos 104 → out of bounds
        assert len(deletions) == 1

    # -- Insertions --

    def test_insertions_size_1(self, fasta):
        """max_indel_size=1 adds 4 insertions per position (4 possible bases)."""
        iv = Interval("chr1", 0, 2)  # AC
        variants = list(generate_variants(iv, fasta, max_indel_size=1))
        insertions = [v for v in variants if v.is_insertion]
        # 2 positions × 4 inserted bases = 8
        assert len(insertions) == 8
        for ins in insertions:
            assert len(ins.alt) == 2  # anchor + 1 inserted base
            assert ins.alt[0] == ins.ref  # anchor preserved

    def test_insertions_size_2(self, fasta):
        """max_indel_size=2 adds 4 + 16 insertions per position."""
        iv = Interval("chr1", 0, 1)  # A
        variants = list(generate_variants(iv, fasta, max_indel_size=2))
        insertions = [v for v in variants if v.is_insertion]
        # size 1: 4 sequences, size 2: 4^2=16 sequences → 20
        assert len(insertions) == 20

    # -- Combined counts --

    def test_combined_count(self, fasta):
        """SNVs + indels at max_indel_size=1."""
        iv = Interval("chr1", 0, 4)  # ACGT, fetched seq is 4bp
        variants = list(generate_variants(iv, fasta, max_indel_size=1))
        snvs = [v for v in variants if v.is_snp]
        dels = [v for v in variants if v.is_deletion]
        ins = [v for v in variants if v.is_insertion]
        assert len(snvs) == 12   # 4 × 3
        assert len(dels) == 3    # 3 positions can delete (last can't)
        assert len(ins) == 16    # 4 × 4
        assert len(variants) == 31

    # -- Edge cases --

    def test_empty_interval(self, fasta):
        iv = Interval("chr1", 50, 50)
        assert list(generate_variants(iv, fasta)) == []

    def test_single_position(self, fasta):
        iv = Interval("chr1", 0, 1)  # A
        variants = list(generate_variants(iv, fasta))
        assert len(variants) == 3
        assert all(v.pos == 0 for v in variants)

    def test_bad_chrom_raises(self, fasta):
        with pytest.raises(ValueError, match="not found in FASTA"):
            list(generate_variants(Interval("chrX", 0, 10), fasta))

    def test_negative_indel_size_raises(self, fasta):
        with pytest.raises(ValueError, match="non-negative"):
            list(generate_variants(Interval("chr1", 0, 4), fasta, max_indel_size=-1))

    def test_interval_clamped_to_chrom_length(self, fasta):
        """Interval extending past chrom end is clamped, not an error."""
        # chr1 is 104bp; interval asks for 200
        iv = Interval("chr1", 100, 200)
        variants = list(generate_variants(iv, fasta))
        # Only 4 positions (100..103), each with 3 SNVs
        assert len(variants) == 12

    def test_chr2_homopolymer(self, fasta):
        """chr2 is TTTTTTTTTTAAAAAAAAAA — T region has only 3 alts per pos."""
        iv = Interval("chr2", 0, 3)  # TTT
        variants = list(generate_variants(iv, fasta))
        assert len(variants) == 9  # 3 × 3
        assert all(v.ref == "T" for v in variants)


# ── compute_variant_effects ──────────────────────────────────────────


class TestComputeVariantEffects:
    """Tests for compute_variant_effects with synthetic ModelOutput objects."""

    def _make_profile_count(
        self, logits: torch.Tensor, log_counts: torch.Tensor
    ) -> ProfileCountOutput:
        return ProfileCountOutput(logits=logits, log_counts=log_counts)

    # -- Basic functionality --

    def test_identical_outputs_zero_effects(self):
        """Identical ref and alt → SAD=0, log_fc=0, JSD=0, pearson=1."""
        logits = torch.randn(2, 1, 100)
        log_counts = torch.tensor([[3.0]])
        ref = self._make_profile_count(logits, log_counts)
        alt = self._make_profile_count(logits.clone(), log_counts.clone())

        effects = compute_variant_effects(ref, alt)

        assert effects["sad"].abs().max().item() == pytest.approx(0.0, abs=1e-5)
        assert effects["log_fc"].abs().max().item() == pytest.approx(0.0, abs=1e-5)
        assert effects["jsd"].abs().max().item() == pytest.approx(0.0, abs=1e-5)
        assert effects["pearson"].min().item() == pytest.approx(1.0, abs=1e-4)
        assert effects["max_abs_diff"].abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_batched_shapes(self):
        """Batched (B, C, L) inputs → (B, C) metric shapes."""
        B, C, L = 4, 3, 64
        ref = self._make_profile_count(torch.randn(B, C, L), torch.ones(B, C))
        alt = self._make_profile_count(torch.randn(B, C, L), torch.ones(B, C))

        effects = compute_variant_effects(ref, alt)

        assert effects["sad"].shape == (B, C)
        assert effects["log_fc"].shape == (B, C)
        assert effects["jsd"].shape == (B, C)
        assert effects["pearson"].shape == (B, C)
        assert effects["max_abs_diff"].shape == (B, C)

    def test_unbatched_shapes(self):
        """Unbatched (C, L) inputs → (C,) metric shapes."""
        C, L = 2, 64
        ref = self._make_profile_count(torch.randn(C, L), torch.ones(C))
        alt = self._make_profile_count(torch.randn(C, L), torch.ones(C))

        effects = compute_variant_effects(ref, alt)

        assert effects["sad"].shape == (C,)
        assert effects["log_fc"].shape == (C,)
        assert effects["jsd"].shape == (C,)
        assert effects["pearson"].shape == (C,)
        assert effects["max_abs_diff"].shape == (C,)

    # -- Numerical accuracy --

    def test_sad_manual(self):
        """SAD matches manual computation."""
        ref_logits = torch.zeros(1, 1, 4)
        ref_logits[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        alt_logits = torch.zeros(1, 1, 4)
        alt_logits[0, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        log_c = torch.tensor([[2.0]])  # exp(2) ≈ 7.389

        ref = self._make_profile_count(ref_logits, log_c)
        alt = self._make_profile_count(alt_logits, log_c)

        ref_probs = torch.softmax(ref_logits, dim=-1)
        alt_probs = torch.softmax(alt_logits, dim=-1)
        total = torch.exp(log_c)
        expected_sad = ((alt_probs - ref_probs).abs() * total).sum(dim=-1)

        effects = compute_variant_effects(ref, alt)
        assert effects["sad"].item() == pytest.approx(expected_sad.item(), rel=1e-5)

    def test_log_fc_manual(self):
        """Log fold change is simply alt_log_counts - ref_log_counts."""
        ref = self._make_profile_count(
            torch.randn(1, 1, 50), torch.tensor([[3.0]])
        )
        alt = self._make_profile_count(
            torch.randn(1, 1, 50), torch.tensor([[5.0]])
        )
        effects = compute_variant_effects(ref, alt)
        assert effects["log_fc"].item() == pytest.approx(2.0, abs=1e-5)

    def test_jsd_bounds(self):
        """JSD is in [0, ln(2)] for probability distributions."""
        import math

        ref = self._make_profile_count(
            torch.randn(4, 2, 100), torch.ones(4, 2)
        )
        alt = self._make_profile_count(
            torch.randn(4, 2, 100), torch.ones(4, 2)
        )
        effects = compute_variant_effects(ref, alt)
        assert (effects["jsd"] >= -1e-6).all()
        assert (effects["jsd"] <= math.log(2) + 1e-6).all()

    def test_jsd_identical_is_zero(self):
        """JSD of identical distributions is exactly 0."""
        logits = torch.randn(1, 1, 100)
        ref = self._make_profile_count(logits, torch.ones(1, 1))
        alt = self._make_profile_count(logits.clone(), torch.ones(1, 1))
        effects = compute_variant_effects(ref, alt)
        assert effects["jsd"].item() == pytest.approx(0.0, abs=1e-6)

    def test_jsd_maximally_different(self):
        """JSD approaches ln(2) for maximally different distributions."""
        import math

        L = 1000
        ref_logits = torch.full((1, 1, L), -1e6)
        ref_logits[0, 0, 0] = 0.0  # all mass on position 0
        alt_logits = torch.full((1, 1, L), -1e6)
        alt_logits[0, 0, L - 1] = 0.0  # all mass on last position

        ref = self._make_profile_count(ref_logits, torch.ones(1, 1))
        alt = self._make_profile_count(alt_logits, torch.ones(1, 1))
        effects = compute_variant_effects(ref, alt)
        assert effects["jsd"].item() == pytest.approx(math.log(2), rel=1e-3)

    def test_pearson_anticorrelated(self):
        """Perfectly anticorrelated signals → pearson ≈ -1.

        Uses ProfileLogRates so exp(log_rates) preserves the monotone
        relationship without softmax distortion.
        """
        log_rates = torch.log(
            torch.arange(1, 101, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        ref = ProfileLogRates(log_rates=log_rates)
        alt = ProfileLogRates(log_rates=log_rates.flip(-1))
        effects = compute_variant_effects(ref, alt)
        assert effects["pearson"].item() == pytest.approx(-1.0, abs=1e-3)

    def test_max_abs_diff_manual(self):
        """max_abs_diff matches manual computation."""
        ref_logits = torch.zeros(1, 1, 4)
        alt_logits = torch.zeros(1, 1, 4)
        alt_logits[0, 0, 2] = 10.0  # spike at position 2

        log_c = torch.tensor([[2.0]])
        ref = self._make_profile_count(ref_logits, log_c)
        alt = self._make_profile_count(alt_logits, log_c)
        effects = compute_variant_effects(ref, alt)

        # Compute expected manually
        ref_signal = torch.softmax(ref_logits, dim=-1) * torch.exp(log_c).unsqueeze(-1)
        alt_signal = torch.softmax(alt_logits, dim=-1) * torch.exp(log_c).unsqueeze(-1)
        expected = (alt_signal - ref_signal).abs().max(dim=-1).values
        assert effects["max_abs_diff"].item() == pytest.approx(
            expected.item(), rel=1e-4
        )

    # -- SAD non-negative --

    def test_sad_non_negative(self):
        """SAD is always non-negative."""
        ref = self._make_profile_count(torch.randn(8, 2, 100), torch.randn(8, 2))
        alt = self._make_profile_count(torch.randn(8, 2, 100), torch.randn(8, 2))
        effects = compute_variant_effects(ref, alt)
        assert (effects["sad"] >= 0).all()

    # -- ProfileLogRates support --

    def test_log_rates_output(self):
        """compute_variant_effects works with ProfileLogRates."""
        ref = ProfileLogRates(log_rates=torch.randn(2, 1, 50))
        alt = ProfileLogRates(log_rates=torch.randn(2, 1, 50))
        effects = compute_variant_effects(ref, alt)
        assert "sad" in effects
        assert "pearson" in effects
        assert "max_abs_diff" in effects
        # No log_fc or jsd for ProfileLogRates
        assert "log_fc" not in effects
        assert "jsd" not in effects

    # -- Dalmatian FactorizedProfileCountOutput --

    def test_dalmatian_signal_metrics(self):
        """FactorizedProfileCountOutput includes signal_* metrics."""
        B, C, L = 2, 1, 64
        ref = FactorizedProfileCountOutput(
            logits=torch.randn(B, C, L),
            log_counts=torch.ones(B, C),
            bias_logits=torch.randn(B, C, L),
            bias_log_counts=torch.ones(B, C),
            signal_logits=torch.randn(B, C, L),
            signal_log_counts=torch.ones(B, C),
        )
        alt = FactorizedProfileCountOutput(
            logits=torch.randn(B, C, L),
            log_counts=torch.ones(B, C),
            bias_logits=torch.randn(B, C, L),
            bias_log_counts=torch.ones(B, C),
            signal_logits=torch.randn(B, C, L),
            signal_log_counts=torch.ones(B, C),
        )
        effects = compute_variant_effects(ref, alt)

        # Combined metrics present
        assert "sad" in effects
        assert "log_fc" in effects
        assert "jsd" in effects
        # Signal sub-model metrics present
        assert "signal_sad" in effects
        assert "signal_log_fc" in effects
        assert "signal_jsd" in effects
        # Shapes are correct
        assert effects["signal_sad"].shape == (B, C)
        assert effects["signal_log_fc"].shape == (B, C)
        assert effects["signal_jsd"].shape == (B, C)

    def test_dalmatian_identical_signal_zero(self):
        """Identical signal sub-models → signal_sad=0, signal_jsd=0."""
        sig_logits = torch.randn(1, 1, 64)
        sig_log_counts = torch.ones(1, 1)
        ref = FactorizedProfileCountOutput(
            logits=torch.randn(1, 1, 64),
            log_counts=torch.ones(1, 1),
            bias_logits=torch.randn(1, 1, 64),
            bias_log_counts=torch.ones(1, 1),
            signal_logits=sig_logits,
            signal_log_counts=sig_log_counts,
        )
        alt = FactorizedProfileCountOutput(
            logits=torch.randn(1, 1, 64),
            log_counts=torch.ones(1, 1),
            bias_logits=torch.randn(1, 1, 64),
            bias_log_counts=torch.ones(1, 1),
            signal_logits=sig_logits.clone(),
            signal_log_counts=sig_log_counts.clone(),
        )
        effects = compute_variant_effects(ref, alt)
        assert effects["signal_sad"].item() == pytest.approx(0.0, abs=1e-5)
        assert effects["signal_log_fc"].item() == pytest.approx(0.0, abs=1e-5)
        assert effects["signal_jsd"].item() == pytest.approx(0.0, abs=1e-5)
