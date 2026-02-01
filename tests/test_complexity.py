import pytest
import torch
import numpy as np
from cerberus.complexity import calculate_gc_content, calculate_dust_score, calculate_log_cpg_ratio

# --- Test Data ---
SEQ_A = "AAAAAAAA"  # GC=0.0, DUST high
SEQ_GC = "GCGCGCGC"  # GC=1.0, DUST high
SEQ_MIX = "ACGTACGT" # GC=0.5, DUST lower
SEQ_N = "NNNNNNNN"   # GC=0.0, DUST=0 (if N treated as unique it won't form repeats? No, N is 4. NNN is 4,4,4. So it is repetitive!)
# Wait, if N is 4, then NNN is a repeat of 4,4,4.
# DUST usually filters low complexity. Poly-N is low complexity.
# So "NNNNNNNN" should have high dust score in our implementation.

# --- GC Content Tests ---

def test_gc_content_single_str():
    assert calculate_gc_content(SEQ_A) == 0.0
    assert calculate_gc_content(SEQ_GC) == 1.0
    assert calculate_gc_content(SEQ_MIX) == 0.5
    assert calculate_gc_content("N") == 0.0

def test_gc_content_batch_list():
    batch = [SEQ_A, SEQ_GC, SEQ_MIX]
    results = calculate_gc_content(batch)
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == 0.0
    assert results[1] == 1.0
    assert results[2] == 0.5

def test_gc_content_batch_tensor():
    # ACGT -> 0,1,2,3
    # AAAAAAAA -> 0,0,0...
    t1 = torch.zeros(4, 8)
    t1[0, :] = 1 # All A
    
    t2 = torch.zeros(4, 8)
    t2[1, ::2] = 1 # C
    t2[2, 1::2] = 1 # G
    # GCGCGCGC
    
    batch = torch.stack([t1, t2]) # (2, 4, 8)
    
    results = calculate_gc_content(batch)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0] == 0.0
    assert results[1] == 1.0

# --- DUST Score Tests ---

def test_dust_score_single_str():
    # AAAAAAAA, k=3
    # L=8, M=6.
    # kmers: AAA (0,0,0) appears 6 times.
    # score = 6 * 5 / 2 = 15.
    # norm = 15 / 6 = 2.5
    res1 = calculate_dust_score(SEQ_A, k=3)
    assert isinstance(res1, float)
    assert abs(res1 - 2.5) < 1e-6
    
    # ATATATAT, k=3. L=8, M=6.
    # ATA, TAT, ATA, TAT, ATA, TAT
    # ATA: 3 times. TAT: 3 times.
    # score = (3*2/2) + (3*2/2) = 3 + 3 = 6.
    # norm = 6 / 6 = 1.0
    seq_at = "ATATATAT"
    res2 = calculate_dust_score(seq_at, k=3)
    assert isinstance(res2, float)
    assert abs(res2 - 1.0) < 1e-6

def test_dust_score_batch_consistency():
    batch = [SEQ_A, "ATATATAT", "ACGTACGT"]
    batch_scores = calculate_dust_score(batch, k=3)
    assert isinstance(batch_scores, list)
    
    single_scores = [calculate_dust_score(s, k=3) for s in batch]
    
    assert np.allclose(batch_scores, single_scores)

def test_dust_score_tensor_single():
    # Construct tensor for AAAAAAAA
    t = torch.zeros(4, 8)
    t[0, :] = 1
    
    score = calculate_dust_score(t, k=3)
    assert isinstance(score, float)
    assert abs(score - 2.5) < 1e-6

def test_dust_score_tensor_batch():
    t1 = torch.zeros(4, 8)
    t1[0, :] = 1 # AAAAAAAA
    
    t2 = torch.zeros(4, 8)
    t2[0, ::2] = 1 # A
    t2[3, 1::2] = 1 # T
    # ATATATAT
    
    batch = torch.stack([t1, t2])
    scores = calculate_dust_score(batch, k=3)
    assert isinstance(scores, list)
    
    assert abs(scores[0] - 2.5) < 1e-6
    assert abs(scores[1] - 1.0) < 1e-6

def test_dust_score_Ns():
    # NNNNNNNN -> should be treated as repeat of 4,4,4
    # Same score as AAAAAAAA
    score = calculate_dust_score(SEQ_N, k=3)
    assert isinstance(score, float)
    assert abs(score - 2.5) < 1e-6

def test_gc_content_batch_consistency_large():
    """Verify that batch optimization produces identical results to loop."""
    # Create random sequences
    np.random.seed(42)
    # 100 sequences of length 50
    seqs = []
    for _ in range(100):
        s = "".join(np.random.choice(list("ACGTN"), size=50))
        seqs.append(s)
        
    # Batch (optimized)
    batch_res = calculate_gc_content(seqs)
    assert isinstance(batch_res, list)
    
    # Loop (single)
    loop_res = [calculate_gc_content(s) for s in seqs]
    
    assert np.allclose(batch_res, loop_res)

def test_dust_score_batch_consistency_large():
    """Verify that batch optimization produces identical results to loop."""
    np.random.seed(42)
    seqs = []
    for _ in range(100):
        # Generate some with repeats to be interesting
        motif = "".join(np.random.choice(list("ACGT"), size=5))
        s = motif * 10 
        seqs.append(s)
        
    # Batch (optimized)
    batch_res = calculate_dust_score(seqs, k=3)
    assert isinstance(batch_res, list)
    
    # Loop (single)
    loop_res = [calculate_dust_score(s, k=3) for s in seqs]
    
    assert np.allclose(batch_res, loop_res)

def test_numpy_scalar():
    """Verify handling of 0-d numpy arrays containing a string."""
    arr = np.array("ACGT")
    
    # GC Content
    res = calculate_gc_content(arr)
    assert isinstance(res, float)
    assert res == 0.5
    
    # DUST Score (k=2 for short seq)
    # AC, CG, GT -> all unique. 0 score.
    res_dust = calculate_dust_score(arr, k=2)
    assert isinstance(res_dust, float)
    assert res_dust == 0.0

def test_cpg_ratio_single():
    """Verify single sequence CpG ratio logic."""
    # Enriched: "CGCG". L=4. C=2, G=2. Exp = 4/4 = 1. Obs=2. Ratio=2. Log2=1.0.
    res = calculate_log_cpg_ratio("CGCG")
    assert isinstance(res, float)
    assert abs(res - 1.0) < 1e-4
    
    # Depleted: "CCTTTTGG". L=8. C=2, G=2. Exp = 4/8 = 0.5. Obs=0 (No CG).
    # Ratio ~ 0. Log2 ~ -inf.
    res_dep = calculate_log_cpg_ratio("CCTTTTGG")
    assert isinstance(res_dep, float)
    assert res_dep < -10.0
    
    # Neutral / Zero: "AAAA". C=0, G=0. Exp=0. Obs=0. Ratio ~ 1. Log2 ~ 0.
    res_zero = calculate_log_cpg_ratio("AAAA")
    assert isinstance(res_zero, float)
    assert abs(res_zero - 0.0) < 1e-4

def test_cpg_ratio_batch_consistency():
    """Verify batch optimization produces identical results."""
    batch = ["CGCG", "ACGT", "AAAAAAAA", "CCCCGGGG"]
    batch_res = calculate_log_cpg_ratio(batch)
    assert isinstance(batch_res, list)
    
    single_res = [calculate_log_cpg_ratio(s) for s in batch]
    assert np.allclose(batch_res, single_res)

def test_cpg_ratio_tensor_batch():
    """Verify tensor batch processing."""
    # CGCG -> 1,2,1,2 (One-hot indices 1, 2)
    t1 = torch.zeros(4, 4)
    t1[1, 0] = 1 # C
    t1[2, 1] = 1 # G
    t1[1, 2] = 1 # C
    t1[2, 3] = 1 # G
    
    # AAAA -> 0,0,0,0 (Index 0)
    t2 = torch.zeros(4, 4)
    t2[0, :] = 1
    
    batch = torch.stack([t1, t2])
    res = calculate_log_cpg_ratio(batch)
    assert isinstance(res, list)
    
    # Check values match single string logic
    assert abs(res[0] - 1.0) < 1e-4
    assert abs(res[1] - 0.0) < 1e-4

def test_dust_score_normalize():
    # AAAAAAAA (8) k=3. Raw Score = 2.5.
    # Normalize uses tanh(score).
    # tanh(2.5) approx 0.9866
    res = calculate_dust_score("AAAAAAAA", k=3, normalize=True)
    assert isinstance(res, float)
    assert abs(res - 0.9866) < 1e-3
    
    # ATATATAT (8). Raw Score = 1.0.
    # tanh(1.0) approx 0.7616
    res2 = calculate_dust_score("ATATATAT", k=3, normalize=True)
    assert isinstance(res2, float)
    assert abs(res2 - 0.7616) < 1e-3

def test_cpg_ratio_normalize():
    # Neutral: "CCGG". L=4. C=2, G=2. Exp=1. Obs=1. Ratio=1. Log2=0. Sigmoid=0.5.
    res = calculate_log_cpg_ratio("CCGG", normalize=True)
    assert isinstance(res, float)
    assert abs(res - 0.5) < 1e-4
    
    # Enriched: CGCG. Ratio=2. Log2=1. Sigmoid(1) = 0.731.
    res2 = calculate_log_cpg_ratio("CGCG", normalize=True)
    assert isinstance(res2, float)
    assert res2 > 0.7
    assert res2 < 0.75
    
    # Depleted: Ratio ~ 0. Log2 ~ -inf. Sigmoid(-inf) ~ 0.
    res3 = calculate_log_cpg_ratio("CCTTTTGG", normalize=True)
    assert isinstance(res3, float)
    assert res3 < 0.1
    assert res3 >= 0.0

def test_dust_score_batch_large_k():
    """Test DUST score with k=12, which triggers the fallback to avoid OOM."""
    # A sequence with repeats of length 12
    # "A" * 20. 
    # k=12. L=20. M=9.
    # kmers: "AAAAAAAAAAAA" (length 12). 9 times.
    # All are identical.
    # Score = 9 * 8 / 2 = 36.
    # Normalized by M=9. Score = 4.0.
    seqs = ["A" * 20] * 5
    scores = calculate_dust_score(seqs, k=12)
    assert isinstance(scores, list)
    assert len(scores) == 5
    assert abs(scores[0] - 4.0) < 1e-6

def test_gc_content_all_N():
    """Verify GC content returns 0.0 for sequences with no valid bases."""
    assert calculate_gc_content("NNNN") == 0.0
    assert calculate_gc_content(["NNNN", "ACGT"]) == [0.0, 0.5]
