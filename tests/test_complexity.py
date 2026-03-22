import numpy as np
import pyfaidx
import pytest

from cerberus.complexity import (
    calculate_dust_score,
    calculate_gc_content,
    calculate_log_cpg_ratio,
    compute_intervals_complexity,
)
from cerberus.interval import Interval

# --- Test Data ---
SEQ_A = "AAAAAAAA"  # GC=0.0, DUST high
SEQ_GC = "GCGCGCGC"  # GC=1.0, DUST high
SEQ_MIX = "ACGTACGT" # GC=0.5, DUST lower
SEQ_N = "NNNNNNNN"   # GC=0.0, DUST high (N=4, repeats)

# --- GC Content Tests ---

def test_gc_content_single_str():
    assert calculate_gc_content(SEQ_A) == 0.0
    assert calculate_gc_content(SEQ_GC) == 1.0
    assert calculate_gc_content(SEQ_MIX) == 0.5
    assert calculate_gc_content("N") == 0.0

def test_gc_content_type_error():
    with pytest.raises(TypeError):
        calculate_gc_content(123)  # type: ignore

# --- DUST Score Tests ---

def test_dust_score_single_str():
    # AAAAAAAA, k=3
    # L=8, M=6.
    # kmers: AAA (0,0,0) appears 6 times.
    # score = 6 * 5 / 2 = 15.
    # norm = 15 / 6 = 2.5
    res1 = calculate_dust_score(SEQ_A, k=3, normalize=False)
    assert isinstance(res1, float)
    assert abs(res1 - 2.5) < 1e-6
    
    # ATATATAT, k=3. L=8, M=6.
    # ATA, TAT, ATA, TAT, ATA, TAT
    # ATA: 3 times. TAT: 3 times.
    # score = (3*2/2) + (3*2/2) = 3 + 3 = 6.
    # norm = 6 / 6 = 1.0
    seq_at = "ATATATAT"
    res2 = calculate_dust_score(seq_at, k=3, normalize=False)
    assert isinstance(res2, float)
    assert abs(res2 - 1.0) < 1e-6

def test_dust_score_Ns():
    # NNNNNNNN -> should be treated as repeat of 4,4,4
    # Same score as AAAAAAAA
    score = calculate_dust_score(SEQ_N, k=3, normalize=False)
    assert isinstance(score, float)
    assert abs(score - 2.5) < 1e-6

def test_dust_score_normalize():
    # AAAAAAAA (8) k=3. Raw Score = 2.5.
    # Exp Random = 6/128 = 0.046875. Ratio = 2.5/0.046875 = 53.33. Log = 3.976.
    # Norm = tanh(3.976 / 1.5) = tanh(2.651) = 0.990.
    res = calculate_dust_score("AAAAAAAA", k=3, normalize=True)
    assert isinstance(res, float)
    assert abs(res - 0.990) < 1e-3
    
    # ATATATAT (8). Raw Score = 1.0.
    # Ratio = 1.0/0.046875 = 21.33. Log = 3.06.
    # Norm = tanh(3.06 / 1.5) = tanh(2.04) = 0.967.
    res2 = calculate_dust_score("ATATATAT", k=3, normalize=True)
    assert isinstance(res2, float)
    assert abs(res2 - 0.967) < 1e-3

def test_dust_score_k_value_error():
    with pytest.raises(ValueError, match="k must be <= 5"):
        calculate_dust_score("ACGT", k=6)
    
    with pytest.raises(ValueError, match="k must be >= 1"):
        calculate_dust_score("ACGT", k=0)

def test_dust_score_type_error():
    with pytest.raises(TypeError):
        calculate_dust_score(123)  # type: ignore

def test_dust_comprehensive_normalization():
    """Check DUST score normalization across various complexities."""
    test_cases = [
        ("ACGT", 0.0),       # Unique -> 0
        ("AAAAAAAA", 2.5),   # Repetitive -> High
        ("ATATATAT", 1.0),   # Moderate -> Medium
        ("NNNN", 0.5)        # N repeat (L=4, k=3, M=2. "NNN" count=2. Score=2*1/2=1. Norm=1/2=0.5)
    ]
    
    # Helper to calc expected norm
    def expected_norm_func(raw, seq_len, k=3):
        if seq_len < k: return 0.0
        exp_random = max((seq_len - k + 1) / (2 * 4**k), 1e-9)
        ratio = (raw + 1e-9) / exp_random
        return np.tanh(np.log(ratio) / 1.5)

    for seq, expected_raw in test_cases:
        # Check Unnormalized
        raw = calculate_dust_score(seq, k=3, normalize=False)
        assert abs(raw - expected_raw) < 1e-6, f"Raw mismatch for {seq}"
        
        # Check Normalized
        norm = calculate_dust_score(seq, k=3, normalize=True)
        exp_norm = expected_norm_func(raw, len(seq), k=3)
        # Use max(0, val) as in impl
        exp_norm = max(0.0, exp_norm)
        
        assert abs(norm - exp_norm) < 1e-6, f"Norm mismatch for {seq}"
        assert 0.0 <= norm < 1.0

# --- CpG Ratio Tests ---

def test_cpg_ratio_single():
    """Verify single sequence CpG ratio logic."""
    # Enriched: "CGCG". L=4. C=2, G=2. Exp = 4/4 = 1. Obs=2.
    # Eps=1.0. Ratio = (2+1)/(1+1) = 1.5. Log2(1.5) approx 0.585.
    res = calculate_log_cpg_ratio("CGCG", normalize=False)
    assert isinstance(res, float)
    assert abs(res - 0.585) < 1e-3
    
    # Depleted: "CCTTTTGG". L=8. C=2, G=2. Exp = 4/8 = 0.5. Obs=0 (No CG).
    # Eps=1.0. Ratio = (0+1)/(0.5+1) = 1/1.5 = 0.666. Log2(0.666) approx -0.585.
    res_dep = calculate_log_cpg_ratio("CCTTTTGG", normalize=False)
    assert isinstance(res_dep, float)
    assert abs(res_dep - (-0.585)) < 1e-3
    
    # Neutral / Zero: "AAAA". C=0, G=0. Exp=0. Obs=0. 
    # Eps=1.0. Ratio = 1/1 = 1. Log2(1) = 0.
    res_zero = calculate_log_cpg_ratio("AAAA", normalize=False)
    assert isinstance(res_zero, float)
    assert abs(res_zero - 0.0) < 1e-4

def test_cpg_ratio_normalize():
    # Neutral: "CCGG". L=4. C=2, G=2. Exp=1. Obs=1. Ratio=1. Log2=0. Sigmoid=0.5.
    res = calculate_log_cpg_ratio("CCGG", normalize=True)
    assert isinstance(res, float)
    assert abs(res - 0.5) < 1e-4
    
    # Enriched: CGCG. Log2(1.5) = 0.585.
    # Norm = (tanh(0.585/2) + 1)/2 = (tanh(0.292) + 1)/2 = (0.284 + 1)/2 = 0.642.
    res2 = calculate_log_cpg_ratio("CGCG", normalize=True)
    assert isinstance(res2, float)
    assert abs(res2 - 0.642) < 1e-3
    
    # Depleted: CCTTTTGG. Log2(0.666) = -0.585.
    # Norm = (tanh(-0.292) + 1)/2 = (-0.284 + 1)/2 = 0.358.
    res3 = calculate_log_cpg_ratio("CCTTTTGG", normalize=True)
    assert isinstance(res3, float)
    assert abs(res3 - 0.358) < 1e-3
    
    # Edge case: Short sequence
    short = "A"
    raw_short = calculate_log_cpg_ratio(short, normalize=False)
    norm_short = calculate_log_cpg_ratio(short, normalize=True)
    assert raw_short == 0.0
    assert norm_short == 0.5 # Neutral

def test_cpg_comprehensive_normalization():
    """Check CpG ratio normalization across various enrichments."""
    # Recalculate expectations with epsilon=1.0
    # ACGT: Obs=1, Exp=1*1/4=0.25. Ratio=(1+1)/(0.25+1)=2/1.25=1.6. Log2(1.6)=0.678.
    # AAAA: Obs=0, Exp=0. Ratio=1/1=1. Log2=0.
    # CGCGCGCG: Obs=4, Exp=4*4/8=2. Ratio=(4+1)/(2+1)=5/3=1.666. Log2(1.666)=0.737.
    # CCTTTTGG: Obs=0, Exp=2*2/8=0.5. Ratio=(0+1)/(0.5+1)=0.666. Log2(0.666)=-0.585.
    
    test_cases = [
        ("ACGT", 0.678),
        ("AAAA", 0.0),
        ("CGCGCGCG", 0.737),
        ("CCTTTTGG", -0.585)
    ]
    
    for seq, expected_log_approx in test_cases:
        # Check Unnormalized
        raw = calculate_log_cpg_ratio(seq, normalize=False)
        
        # Verify manual calc matches roughly what we expect
        assert abs(raw - expected_log_approx) < 0.1, f"Raw mismatch for {seq}: {raw} vs {expected_log_approx}"
        
        # Check Normalized
        norm = calculate_log_cpg_ratio(seq, normalize=True)
        
        # Relationship: norm = (tanh(raw/2) + 1) / 2
        expected_norm = (np.tanh(raw / 2) + 1.0) / 2.0
        assert abs(norm - expected_norm) < 1e-6, f"Norm mismatch for {seq}. Raw={raw}, Norm={norm}"
        assert 0.0 <= norm <= 1.0

def test_cpg_ratio_type_error():
    with pytest.raises(TypeError):
        calculate_log_cpg_ratio(123)  # type: ignore

def test_gc_content_all_N():
    """Verify GC content returns 0.0 for sequences with no valid bases."""
    assert calculate_gc_content("NNNN") == 0.0


# --- center_size Tests ---

@pytest.fixture
def center_size_fasta(tmp_path):
    """FASTA with a long chr1 that has distinct GC in center vs flanks."""
    fasta_path = tmp_path / "genome.fa"
    rng = np.random.default_rng(42)

    # Build a 40kb sequence: AT-rich flanks, GC-rich center
    flank_len = 15000
    center_len = 10000
    flank = "".join(rng.choice(["A", "T"], size=flank_len))
    center = "".join(rng.choice(["G", "C"], size=center_len))
    seq = flank + center + flank

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")

    pyfaidx.Faidx(str(fasta_path))
    return fasta_path


def test_compute_intervals_complexity_center_size(center_size_fasta):
    """center_size crops intervals before computing metrics."""
    interval = Interval("chr1", 0, 40000, "+")

    full = compute_intervals_complexity([interval], center_size_fasta)
    center = compute_intervals_complexity(
        [interval], center_size_fasta, center_size=2000
    )

    assert full.shape == center.shape == (1, 3)
    # The center 2kb is GC-rich, the full 40kb is mixed — GC metric should differ
    assert not np.allclose(full, center)


def test_compute_intervals_complexity_center_size_noop_when_smaller(center_size_fasta):
    """center_size larger than interval is a no-op."""
    interval = Interval("chr1", 0, 2000, "+")

    full = compute_intervals_complexity([interval], center_size_fasta)
    center = compute_intervals_complexity(
        [interval], center_size_fasta, center_size=50000
    )

    np.testing.assert_array_equal(full, center)

