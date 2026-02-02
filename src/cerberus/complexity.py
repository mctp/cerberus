import math
import numpy as np


def calculate_gc_content(sequence: str) -> float:
    """
    Calculates the GC content of a DNA sequence.

    Note:
        Ambiguous bases (like 'N') are excluded from the calculation.
        The score is relative to the total count of valid bases (A, C, G, T).

    Args:
        sequence: DNA sequence string.

    Returns:
        float: GC content ratio (0.0 to 1.0).
    """
    if not isinstance(sequence, str):
        raise TypeError(f"Input must be str. Got {type(sequence).__name__}")

    seq = sequence.upper()
    gc = seq.count("G") + seq.count("C")
    total = gc + seq.count("A") + seq.count("T")
    return gc / total if total > 0 else 0.0


def calculate_dust_score(
    sequence: str,
    k: int = 3,
    normalize: bool = True,
) -> float:
    """
    Calculates the DUST score for a DNA sequence.

    The DUST score measures sequence complexity based on k-mer repetition.
    Higher score indicates lower complexity (more repetitive).

    Formula: sum(counts * (counts - 1) / 2) / (L - k + 1)

    Args:
        sequence: DNA sequence string.
        k: The length of k-mers to consider. Defaults to 3. Must be <= 5.
        normalize: If True, normalizes score using tanh(score).
                   This maps [0, inf) to [0, 1), spreading out lower scores.

    Returns:
        float: The DUST score.
    """
    if not isinstance(sequence, str):
        raise TypeError(f"Input must be str. Got {type(sequence).__name__}")
        
    if k > 5:
        raise ValueError(f"k must be <= 5. Got {k}.")
        
    if k < 1:
        raise ValueError("k must be >= 1.")

    # Map string to indices for efficient processing
    # A=0, C=1, G=2, T=3, Other=4
    arr_bytes = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    lookup = np.full(256, 4, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    arr = lookup[arr_bytes]

    seq_len = len(arr)
    if seq_len < k:
        return 0.0

    # Efficient k-mer counting using numpy sliding window
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=k)
    powers = 5 ** np.arange(k - 1, -1, -1)
    kmer_indices = np.dot(windows, powers)
    
    # max index is 5^k - 1
    counts = np.bincount(kmer_indices, minlength=5**k)
    score = np.sum(counts * (counts - 1) / 2)
    
    val = float(score / (seq_len - k + 1))
    
    if normalize:
        return math.tanh(math.log(val + 1) / 2)
    return val


def calculate_log_cpg_ratio(
    sequence: str,
    epsilon: float = 1.0,
    normalize: bool = True,
) -> float:
    """
    Calculates the log-transformed Observed/Expected CpG ratio.

    Formula:
        Score = log2( (Obs_Count + eps) / (Exp_Count + eps) )
        Exp_Count = (Count(C) * Count(G)) / Length

    Note:
        The sequence Length used in Exp_Count includes 'N's and other characters.

    Args:
        sequence: DNA sequence string.
        epsilon: Smoothing factor (default: 1e-6).
        normalize: If True, applies scaled tanh transform ((tanh(score)+1)/2)
                   to map result to (0, 1). 0.5 corresponds to neutral.

    Returns:
        float: Log2 CpG ratio.
            0.0  : Neutral (Observed == Expected)
            > 0  : Enriched (CpG Island-like)
            < 0  : Depleted (Methylation suppression)
    """
    if not isinstance(sequence, str):
        raise TypeError(f"Input must be str. Got {type(sequence).__name__}")

    if len(sequence) < 2:
        return 0.5 if normalize else 0.0
        
    s = sequence.upper()
    n_c = s.count('C')
    n_g = s.count('G')
    n_cg = s.count('CG')
    length = len(s)
    
    exp = (n_c * n_g) / length
    ratio = (n_cg + epsilon) / (exp + epsilon)
    val = math.log2(ratio)
    
    if normalize:
        return (math.tanh(val / 2) + 1.0) / 2.0
    return val
