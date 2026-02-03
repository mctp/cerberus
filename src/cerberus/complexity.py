import math
from typing import Iterable
from pathlib import Path
import numpy as np
import pyfaidx

from cerberus.interval import Interval


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
        # Expected random val
        exp_random = max((seq_len - k + 1) / (2 * 4**k), 1e-9)
        # Ratio of Observed / Expected
        ratio = (val + 1e-9) / exp_random
        # Log of ratio. Random ~ 1 -> log(1)=0. Repeat >> 1 -> log(high) -> 1.
        # We use tanh(log(ratio)/1.5) which spreads out the low-to-medium scores.
        # k=1.5 maps ratios ~2.0 to ~0.4, utilizing more of the 0-1 range for actual targets.
        norm_val = math.tanh(math.log(ratio) / 1.5)
        return max(0.0, norm_val)
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
        # Normalize to spread scores. Random DNA (ratio ~1, val ~0) maps to ~0.5.
        # Uses logistic sigmoid: (tanh(val/2) + 1) / 2
        # Maps:
        #   Depleted (val < 0) -> [0, 0.5)
        #   Neutral  (val = 0) -> 0.5
        #   Enriched (val > 0) -> (0.5, 1]
        return (math.tanh(val / 2) + 1) / 2
    return val


def compute_intervals_complexity(
    intervals: Iterable[Interval],
    fasta_path: Path | str,
    metrics: list[str] | None = None
) -> np.ndarray:
    """
    Computes selected complexity metrics for a collection of intervals.

    Args:
        intervals: Iterable of Interval objects.
        fasta_path: Path to the genome FASTA file.
        metrics: List of metrics to compute. Options: 'gc', 'dust', 'cpg'.
                 If None, computes all.

    Returns:
        np.ndarray: A (N, M) array where columns correspond to the requested metrics in order.
    """
    if metrics is None:
        metrics = ["gc", "dust", "cpg"]
        
    valid_metrics = {"gc", "dust", "cpg"}
    for m in metrics:
        if m not in valid_metrics:
            raise ValueError(f"Invalid metric: {m}. Options: {valid_metrics}")

    results = []
    fasta = pyfaidx.Fasta(str(fasta_path))

    for interval in intervals:
        try:
            if interval.chrom in fasta:
                # pyfaidx expects 0-based [start:end]
                seq_obj = fasta[interval.chrom][interval.start : interval.end]
                seq = str(seq_obj)
                
                row = []
                for m in metrics:
                    if m == "gc":
                        row.append(calculate_gc_content(seq))
                    elif m == "dust":
                        row.append(calculate_dust_score(seq, normalize=True))
                    elif m == "cpg":
                        row.append(calculate_log_cpg_ratio(seq, normalize=True))
                results.append(row)
            else:
                results.append([np.nan] * len(metrics))
        except Exception:
            results.append([np.nan] * len(metrics))

    return np.array(results, dtype=np.float32)
