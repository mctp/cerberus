from pathlib import Path
from typing import Protocol, Iterable
import numpy as np
import pyfaidx
import torch

from cerberus.interval import Interval
from cerberus.complexity import calculate_gc_content


def _create_mapping(encoding: str) -> np.ndarray:
    """Creates a lookup array for DNA encoding."""
    mapping = np.zeros(256, dtype=np.int8) - 1  # Default -1
    for i, base in enumerate(encoding):
        mapping[ord(base)] = i
    return mapping


_ENCODING_MAPPINGS = {"ACGT": _create_mapping("ACGT"), "AGCT": _create_mapping("AGCT")}


def encode_dna(sequence: str, encoding: str = "ACGT") -> torch.Tensor:
    """
    One-hot encodes a DNA sequence string into a tensor.

    This function maps characters in the input sequence to channel indices based on the specified encoding.
    Supported encodings:
    - 'ACGT': A=0, C=1, G=2, T=3
    - 'AGCT': A=0, G=1, C=2, T=3
    
    The resulting tensor is 1-hot encoded.

    Args:
        sequence: Input DNA sequence (string). Case-insensitive.
        encoding: Channel order, e.g. 'ACGT' or 'AGCT'. Defaults to 'ACGT'.

    Returns:
        torch.Tensor: A float32 tensor of shape (4, Length).
                      Channels correspond to the bases in the order specified by `encoding`.
    
    Raises:
        ValueError: If the encoding is not supported.
    """
    sequence = sequence.upper()
    encoding = encoding.upper()

    if encoding not in _ENCODING_MAPPINGS:
        raise ValueError(
            f"Unsupported encoding '{encoding}'. Supported: {list(_ENCODING_MAPPINGS.keys())}"
        )

    mapping = _ENCODING_MAPPINGS[encoding]

    # Convert string to ascii bytes
    # Note: this assumes ASCII compatible encoding
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)

    # Map to indices
    indices = mapping[seq_bytes]

    # Create one hot (4, L) directly for contiguous memory
    one_hot = np.zeros((4, len(sequence)), dtype=np.float32)

    # Valid indices (>= 0)
    valid_mask = indices >= 0

    # Fill using advanced indexing
    # indices[valid_mask] gives row indices (channels)
    # np.where(valid_mask)[0] gives column indices (positions)
    one_hot[indices[valid_mask], np.where(valid_mask)[0]] = 1.0

    return torch.from_numpy(one_hot)


def compute_intervals_gc(intervals: Iterable[Interval], fasta_path: Path | str) -> list[float]:
    """
    Computes GC content for a collection of intervals using a FASTA file.

    Args:
        intervals: Iterable of Interval objects.
        fasta_path: Path to the genome FASTA file.

    Returns:
        List of GC content values (floats).
    """
    gc_values = []
    fasta = pyfaidx.Fasta(str(fasta_path))

    for interval in intervals:
        try:
            # pyfaidx expects 0-based [start:end]
            seq_obj = fasta[interval.chrom][interval.start : interval.end]
            seq = str(seq_obj)
            gc_values.append(calculate_gc_content(seq))
        except Exception:
            gc_values.append(0.0)

    return gc_values


class BaseSequenceExtractor(Protocol):
    """
    Protocol for sequence extractors.
    
    Classes implementing this protocol must provide an `extract` method that returns
    a one-hot encoded DNA sequence tensor for a given genomic interval.
    """
    def extract(self, interval: Interval) -> torch.Tensor: ...


class SequenceExtractor(BaseSequenceExtractor):
    """
    Extracts DNA sequence from a FASTA file on-the-fly.
    
    This extractor keeps the FASTA file open (lazily) and reads specific intervals as requested.
    It is memory-efficient but may be slower than in-memory extraction for random access patterns.
    """
    
    def __init__(self, fasta_path: Path | str, encoding: str = "ACGT"):
        """
        Args:
            fasta_path: Path to the FASTA file.
            encoding: Channel order (e.g., 'ACGT'). Defaults to 'ACGT'.
        """
        self.fasta_path = Path(fasta_path)
        self.encoding = encoding
        self.fasta = None

    def __getstate__(self):
        """Pickle support: exclude file handles to allow safe multiprocessing."""
        state = self.__dict__.copy()
        state["fasta"] = None
        return state

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts one-hot encoded sequence for the given interval.

        Args:
            interval: Genomic interval specifying chromosome, start, and end.

        Returns:
            torch.Tensor: Tensor of shape (4, Length), where Length = interval.end - interval.start.
            
        Raises:
            ValueError: If the chromosome is not found in the FASTA file.
        """
        # Lazy load (fork safety via lazy init + __getstate__)
        if self.fasta is None:
            self.fasta = pyfaidx.Fasta(str(self.fasta_path))

        # pyfaidx handles 0-based coordinates if using slicing on the chrom object
        # but standard usage is fasta[chrom][start:end]
        # Check if chrom exists
        if interval.chrom not in self.fasta:
            raise ValueError(f"Chromosome {interval.chrom} not found in FASTA.")

        # Extract sequence
        # pyfaidx.Fasta slicing is [start:end] 0-based, half-open
        seq_obj = self.fasta[interval.chrom][interval.start : interval.end]
        sequence = seq_obj.seq  # type: ignore

        # We assume the interval is valid and within bounds (guaranteed by Sampler)
        # If pyfaidx truncates (unexpected), encode_dna will return a shorter tensor.
        return encode_dna(sequence, self.encoding)


class InMemorySequenceExtractor(BaseSequenceExtractor):
    """
    Extracts DNA sequence from memory.
    
    This extractor loads the entire FASTA content into memory (RAM) upon initialization.
    It provides faster random access at the cost of high memory usage.
    
    Data is stored as uint8 and shared across processes to reduce overhead when using PyTorch DataLoaders.
    """

    def __init__(self, fasta_path: Path | str, encoding: str = "ACGT"):
        """
        Args:
            fasta_path: Path to the FASTA file.
            encoding: Channel order (e.g., 'ACGT'). Defaults to 'ACGT'.
        """
        self.fasta_path = Path(fasta_path)
        self.encoding = encoding
        self._cache: dict[str, torch.Tensor] = {}
        self._load()

    def _load(self):
        """Loads the entire genome into memory."""
        fasta = pyfaidx.Fasta(str(self.fasta_path))
        for chrom in fasta.keys():
            # Read full sequence
            # pyfaidx behaves like dict
            seq = str(fasta[chrom])
            # Encode and store as uint8 to save memory
            encoded = encode_dna(seq, self.encoding)
            encoded = encoded.to(dtype=torch.uint8)
            # Invoke share_memory_() on the tensor (from torch.Tensor).
            # This moves the underlying storage to shared memory (if not already),
            # allowing the tensor to be shared across processes (e.g. DataLoader workers)
            # without copying the data, preventing memory explosion.
            encoded.share_memory_()
            self._cache[chrom] = encoded

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts one-hot encoded sequence for the given interval from memory.

        Args:
            interval: Genomic interval.

        Returns:
            torch.Tensor: Tensor of shape (4, Length) as float32.
            
        Raises:
            ValueError: If the chromosome is not found in the memory cache.
        """
        if interval.chrom not in self._cache:
            raise ValueError(f"Chromosome {interval.chrom} not found in memory cache.")

        cached = self._cache[interval.chrom]

        # We assume the interval is valid and within bounds (guaranteed by Sampler)
        extracted = cached[:, interval.start : interval.end]

        return extracted.float()
