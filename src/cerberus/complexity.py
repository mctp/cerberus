from typing import Sequence, List
import numpy as np
import torch


def calculate_gc_content(
    sequence: str | torch.Tensor | Sequence[str] | np.ndarray,
) -> float | List[float]:
    """
    Calculates the GC content of a sequence or batch of sequences.

    Note:
        Ambiguous bases (like 'N') are excluded from the calculation.
        The score is relative to the total count of valid bases (A, C, G, T).

    Args:
        sequence: Input data. Can be:
            - str: Single DNA sequence.
            - torch.Tensor: One-hot encoded sequence(s).
                - Shape (4, L) for single sequence.
                - Shape (B, 4, L) for batch.
            - Sequence[str]: Batch of DNA sequences (e.g., list of strings).
            - np.ndarray: Batch of DNA sequences.

    Returns:
        float or List[float]: GC content ratio (0.0 to 1.0).
    """
    # Handle torch.Tensor
    if isinstance(sequence, torch.Tensor):
        # Single: (4, L)
        if sequence.ndim == 2 and sequence.shape[0] == 4:
            gc = sequence[1, :].sum() + sequence[2, :].sum()
            total = sequence.sum()
            return (gc / total).item() if total > 0 else 0.0
        
        # Batch: (B, 4, L)
        elif sequence.ndim == 3 and sequence.shape[1] == 4:
            # shape (B, 4, L)
            # Sum over L (dim 2)
            gc_counts = sequence[:, 1, :].sum(dim=1) + sequence[:, 2, :].sum(dim=1)
            total_counts = sequence.sum(dim=(1, 2))
            
            # Avoid division by zero
            # mask where total is > 0
            res = torch.zeros_like(total_counts)
            mask = total_counts > 0
            res[mask] = gc_counts[mask] / total_counts[mask]
            return res.tolist()
        
        else:
            raise ValueError(
                f"Unexpected tensor shape {sequence.shape}. Expected (4, L) or (B, 4, L)."
            )

    # Handle single string
    if isinstance(sequence, str):
        return _calculate_gc_single(sequence)

    # Handle numpy scalar (0-d array)
    if isinstance(sequence, np.ndarray) and sequence.ndim == 0:
        return _calculate_gc_single(str(sequence))

    # Handle batch of strings
    if isinstance(sequence, (list, tuple, np.ndarray)):
        if len(sequence) == 0:
            return []
            
        # Optimization for equal length strings
        first_len = len(sequence[0])
        if all(len(s) == first_len for s in sequence):
            return _calculate_gc_batch_equal_len(sequence, first_len)
        else:
            return [_calculate_gc_single(s) for s in sequence]

    raise TypeError(
        "Input must be str, torch.Tensor, or Sequence[str]. Got "
        f"{type(sequence).__name__}"
    )


def calculate_dust_score(
    sequence: str | torch.Tensor | Sequence[str] | np.ndarray,
    k: int = 3,
    normalize: bool = False,
) -> float | List[float]:
    """
    Calculates the DUST score for a sequence or a batch of sequences.

    The DUST score measures sequence complexity based on k-mer repetition.
    Higher score indicates lower complexity (more repetitive).

    Formula: sum(counts * (counts - 1) / 2) / (L - k + 1)

    Args:
        sequence: Input data. Can be:
            - str: Single DNA sequence.
            - torch.Tensor: One-hot encoded sequence(s).
                - Shape (4, L) for single sequence.
                - Shape (B, 4, L) for batch.
            - Sequence[str]: Batch of DNA sequences (e.g., list of strings).
            - np.ndarray: Batch of DNA sequences.
        k: The length of k-mers to consider. Defaults to 3.
        normalize: If True, normalizes score using tanh(score).
                   This maps [0, inf) to [0, 1), spreading out lower scores.

    Returns:
        float or list[float]: The DUST score(s).
    """
    # Handle torch.Tensor
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim == 2 and sequence.shape[0] == 4:
            # (4, L) -> Single
            sums = sequence.sum(dim=0).cpu().numpy()
            indices = sequence.argmax(dim=0).cpu().numpy().astype(np.int8)
            indices[sums == 0] = 4  # Treat no-hot as N
            return _calculate_dust_score_indices(indices, k, normalize)

        elif sequence.ndim == 3 and sequence.shape[1] == 4:
            # (B, 4, L) -> Batch
            sums = sequence.sum(dim=1).cpu().numpy()
            indices = sequence.argmax(dim=1).cpu().numpy().astype(np.int8)
            indices[sums == 0] = 4
            return _calculate_dust_score_batch_indices(indices, k, normalize)
        else:
             raise ValueError(
                f"Unexpected tensor shape {sequence.shape}. Expected (4, L) or (B, 4, L)."
            )

    # Handle single string
    if isinstance(sequence, str):
        indices = _string_to_indices(sequence)
        return _calculate_dust_score_indices(indices, k, normalize)

    # Handle numpy scalar (0-d array)
    if isinstance(sequence, np.ndarray) and sequence.ndim == 0:
        indices = _string_to_indices(str(sequence))
        return _calculate_dust_score_indices(indices, k, normalize)

    # Handle batch of strings
    if isinstance(sequence, (list, tuple, np.ndarray)):
        if len(sequence) == 0:
            return []

        first_len = len(sequence[0])
        if all(len(s) == first_len for s in sequence):
            indices_batch = _batch_strings_to_indices(sequence, first_len)
            return _calculate_dust_score_batch_indices(indices_batch, k, normalize)
        else:
            return [
                _calculate_dust_score_indices(_string_to_indices(s), k, normalize)
                for s in sequence
            ]

    raise TypeError(
        "Input must be str, torch.Tensor, or Sequence[str]. Got "
        f"{type(sequence).__name__}"
    )


def calculate_log_cpg_ratio(
    sequence: str | torch.Tensor | Sequence[str] | np.ndarray,
    epsilon: float = 1e-6,
    normalize: bool = False,
) -> float | List[float]:
    """
    Calculates the log-transformed Observed/Expected CpG ratio.

    Formula:
        Score = log2( (Obs_Count + eps) / (Exp_Count + eps) )
        Exp_Count = (Count(C) * Count(G)) / Length

    Note:
        The sequence Length used in Exp_Count includes 'N's and other characters.
        This means 'N's effectively dilute the expected CpG count.

    Args:
        sequence: Input data. Can be:
            - str: Single DNA sequence.
            - torch.Tensor: One-hot encoded sequence(s).
                - Shape (4, L) for single sequence.
                - Shape (B, 4, L) for batch.
            - Sequence[str]: Batch of DNA sequences (e.g., list of strings).
            - np.ndarray: Batch of DNA sequences.
        epsilon: Smoothing factor (default: 1e-6).
        normalize: If True, applies scaled tanh transform ((tanh(score)+1)/2)
                   to map result to (0, 1). 0.5 corresponds to neutral.

    Returns:
        float or List[float]: Log2 CpG ratio.
            0.0  : Neutral (Observed == Expected)
            > 0  : Enriched (CpG Island-like)
            < 0  : Depleted (Methylation suppression)
    """
    # Handle torch.Tensor
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim == 2 and sequence.shape[0] == 4:
            # Single: (4, L)
            is_c = sequence[1, :]
            is_g = sequence[2, :]
            n_c = is_c.sum().item()
            n_g = is_g.sum().item()
            
            # n_cg: C at i, G at i+1
            n_cg = (is_c[:-1] * is_g[1:]).sum().item()
            length = sequence.shape[1]
            
            if length < 2: return 0.5 if normalize else 0.0
            
            exp = (n_c * n_g) / length
            ratio = (n_cg + epsilon) / (exp + epsilon)
            val = float(np.log2(ratio))
            return (float(np.tanh(val)) + 1.0) / 2.0 if normalize else val
            
        elif sequence.ndim == 3 and sequence.shape[1] == 4:
            # Batch: (B, 4, L)
            _, _, L = sequence.shape
            if L < 2:
                default = 0.5 if normalize else 0.0
                return [default] * sequence.shape[0]
            
            n_c = sequence[:, 1, :].sum(dim=1)
            n_g = sequence[:, 2, :].sum(dim=1)
            
            # n_cg: elementwise mult of shifted
            pairs = sequence[:, 1, :-1] * sequence[:, 2, 1:]
            n_cg = pairs.sum(dim=1)
            
            exp = (n_c * n_g) / L
            ratio = (n_cg + epsilon) / (exp + epsilon)
            vals = torch.log2(ratio)
            return ((torch.tanh(vals) + 1.0) / 2.0 if normalize else vals).tolist()
            
        else:
            raise ValueError(
                f"Unexpected tensor shape {sequence.shape}. Expected (4, L) or (B, 4, L)."
            )

    # Handle single string
    if isinstance(sequence, str):
        return _calculate_cpg_single(sequence, epsilon, normalize)

    # Handle numpy scalar
    if isinstance(sequence, np.ndarray) and sequence.ndim == 0:
        return _calculate_cpg_single(str(sequence), epsilon, normalize)

    # Handle batch of strings
    if isinstance(sequence, (list, tuple, np.ndarray)):
        if len(sequence) == 0:
            return []
            
        first_len = len(sequence[0])
        if all(len(s) == first_len for s in sequence):
            return _calculate_cpg_batch_equal_len(sequence, first_len, epsilon, normalize)
        else:
            return [_calculate_cpg_single(s, epsilon, normalize) for s in sequence]

    raise TypeError(
        "Input must be str, torch.Tensor, or Sequence[str]. Got "
        f"{type(sequence).__name__}"
    )


# --- Helper Functions ---

def _calculate_gc_single(sequence: str) -> float:
    seq = sequence.upper()
    gc = seq.count("G") + seq.count("C")
    total = gc + seq.count("A") + seq.count("T")
    return gc / total if total > 0 else 0.0


def _calculate_gc_batch_equal_len(
    sequences: Sequence[str] | np.ndarray, length: int
) -> List[float]:
    # Flatten and map to 1 (GC) or 0 (AT) or -1 (N/Other)
    N = len(sequences)
    flat_str = "".join(sequences).upper()
    arr_bytes = np.frombuffer(flat_str.encode("ascii"), dtype=np.uint8)
    
    # Map: A(65)->0, C(67)->1, G(71)->1, T(84)->0
    # Everything else -> -1 (don't count) or 0 (count as total?) 
    # Logic in _calculate_gc_single counts only ACGT for total.
    
    lookup_gc = np.zeros(256, dtype=np.int8)
    lookup_gc[ord('C')] = 1
    lookup_gc[ord('G')] = 1
    
    lookup_valid = np.zeros(256, dtype=np.int8)
    lookup_valid[ord('A')] = 1
    lookup_valid[ord('C')] = 1
    lookup_valid[ord('G')] = 1
    lookup_valid[ord('T')] = 1
    
    is_gc = lookup_gc[arr_bytes].reshape(N, length)
    is_valid = lookup_valid[arr_bytes].reshape(N, length)
    
    gc_counts = np.sum(is_gc, axis=1)
    total_counts = np.sum(is_valid, axis=1)
    
    # Avoid div by zero
    result = np.zeros(N, dtype=np.float64)
    mask = total_counts > 0
    result[mask] = gc_counts[mask] / total_counts[mask]
    
    return result.tolist()


def _string_to_indices(sequence: str) -> np.ndarray:
    """Maps DNA string to integers (A=0, C=1, G=2, T=3, Other=4)."""
    arr_bytes = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    lookup = np.full(256, 4, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    return lookup[arr_bytes]


def _batch_strings_to_indices(
    sequences: Sequence[str] | np.ndarray, length: int
) -> np.ndarray:
    """Maps batch of DNA strings to 2D integer array."""
    N = len(sequences)
    flat_str = "".join(sequences).upper()
    arr_bytes = np.frombuffer(flat_str.encode("ascii"), dtype=np.uint8)
    
    lookup = np.full(256, 4, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    
    mapped = lookup[arr_bytes]
    return mapped.reshape(N, length)


def _calculate_dust_score_indices(
    arr: np.ndarray, k: int, normalize: bool = False
) -> float:
    seq_len = len(arr)
    if seq_len < k:
        return 0.0

    if k < 12:
        windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=k)
        powers = 5 ** np.arange(k - 1, -1, -1)
        kmer_indices = np.dot(windows, powers)
        counts = np.bincount(kmer_indices)
        score = np.sum(counts * (counts - 1) / 2)
    else:
        kmers = [tuple(arr[i : i + k]) for i in range(seq_len - k + 1)]
        _, counts = np.unique(kmers, axis=0, return_counts=True)
        score = np.sum(counts * (counts - 1) / 2)
    
    val = float(score / (seq_len - k + 1))
    
    if normalize:
        return float(np.tanh(val))
    return val


def _calculate_dust_score_batch_indices(
    mapped: np.ndarray, k: int, normalize: bool = False
) -> List[float]:
    N, L = mapped.shape
    if L < k:
        return [0.0] * N

    # For large k, the bincount method below would require allocating an array
    # of size N * 5^k. For k=12, this is ~244M * N, which will OOM.
    # Fallback to per-sequence processing which handles large k safely.
    if k >= 12:
        return [
            _calculate_dust_score_indices(mapped[i], k, normalize)
            for i in range(N)
        ]

    windows = np.lib.stride_tricks.sliding_window_view(mapped, window_shape=k, axis=1)
    powers = 5 ** np.arange(k - 1, -1, -1)
    kmer_indices = np.dot(windows, powers)

    M = L - k + 1
    max_val = 5**k

    row_offsets = np.arange(N) * max_val
    flat_indices = kmer_indices + row_offsets[:, None]

    total_counts = np.bincount(flat_indices.ravel(), minlength=N * max_val)
    counts_matrix = total_counts.reshape(N, max_val)

    scores = np.sum(counts_matrix * (counts_matrix - 1) / 2, axis=1)
    vals = scores / M
    
    if normalize:
        vals = np.tanh(vals)
            
    return vals.tolist()


def _calculate_cpg_single(
    sequence: str, epsilon: float, normalize: bool = False
) -> float:
    if len(sequence) < 2:
        return 0.5 if normalize else 0.0
    s = sequence.upper()
    n_c = s.count('C')
    n_g = s.count('G')
    n_cg = s.count('CG')
    length = len(s)
    
    exp = (n_c * n_g) / length
    ratio = (n_cg + epsilon) / (exp + epsilon)
    val = float(np.log2(ratio))
    return (float(np.tanh(val)) + 1.0) / 2.0 if normalize else val


def _calculate_cpg_batch_equal_len(
    sequences: Sequence[str] | np.ndarray, length: int, epsilon: float, normalize: bool = False
) -> List[float]:
    if length < 2:
        default = 0.5 if normalize else 0.0
        return [default] * len(sequences)
        
    indices = _batch_strings_to_indices(sequences, length)
    is_c = (indices == 1)
    is_g = (indices == 2)
    
    n_c = is_c.sum(axis=1)
    n_g = is_g.sum(axis=1)
    
    pairs = is_c[:, :-1] & is_g[:, 1:]
    n_cg = pairs.sum(axis=1)
    
    exp = (n_c * n_g) / length
    ratio = (n_cg + epsilon) / (exp + epsilon)
    vals = np.log2(ratio)
    
    if normalize:
        vals = (np.tanh(vals) + 1.0) / 2.0
        
    return vals.tolist()
