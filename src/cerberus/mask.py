from pathlib import Path
from typing import Protocol
import numpy as np
import pybigtools
import torch
import gzip
from interlap import InterLap
from cerberus.interval import Interval


class BaseMaskExtractor(Protocol):
    """
    Protocol for mask extractors.
    
    Classes implementing this protocol must provide an `extract` method that returns
    a binary mask tensor for a given genomic interval.
    """
    def extract(self, interval: Interval) -> torch.Tensor: ...


class BigBedMaskExtractor(BaseMaskExtractor):
    """
    Extracts binary masks from BigBed files on-the-fly.
    
    This extractor handles multiple channels (one BigBed file per channel).
    It reads entries overlapping the requested interval and constructs a binary mask
    (1.0 where a BigBed entry exists, 0.0 otherwise).
    """
    def __init__(self, bigbed_paths: dict[str, Path]):
        """
        Extracts binary mask from BigBed files for the given interval.

        Args:
            bigbed_paths: Dictionary mapping channel names to BigBed file paths.
                          e.g., {'TSS': Path('tss.bb'), 'Enhancer': Path('enhancers.bb')}
        """
        self.bigbed_paths = bigbed_paths
        self.channels = sorted(bigbed_paths.keys())
        self._bigbed_files = None

    def _load(self):
        """Lazy loader for BigBed handles."""
        self._bigbed_files = {}
        for name in self.channels:
            path = str(self.bigbed_paths[name])
            try:
                self._bigbed_files[name] = pybigtools.open(path)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to open BigBed file {path}: {e}")

    def __getstate__(self):
        """Pickle support: exclude file handles."""
        state = self.__dict__.copy()
        state["_bigbed_files"] = None
        return state

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts binary mask for the given interval.

        Args:
            interval: Genomic interval specifying chromosome, start, and end.

        Returns:
            torch.Tensor: Float32 tensor of shape (Channels, Length).
                          Values are 1.0 (overlap) or 0.0 (no overlap).
        """
        if self._bigbed_files is None:
            self._load()

        start = interval.start
        end = interval.end
        length = end - start

        extracted_values = []
        for name in self.channels:
            vals = np.zeros(length, dtype=np.float32)
            # We guaranteed _bigbed_files is not None and populated in _load()
            bb = self._bigbed_files[name]  # type: ignore

            try:
                # Use records iterator for sub-region queries to build the mask manually
                entries = bb.records(interval.chrom, start, end)
                if entries:
                    for entry in entries:
                        h_start, h_end = entry[0], entry[1]

                        # Clip to query interval
                        s = max(0, h_start - start)
                        e = min(length, h_end - start)

                        if s < e:
                            vals[s:e] = 1.0

            except RuntimeError:
                # Chromosome not found or read error -> zeros
                pass
            except Exception:
                pass

            extracted_values.append(vals)

        return torch.from_numpy(np.stack(extracted_values))


class InMemoryBigBedMaskExtractor(BaseMaskExtractor):
    """
    Extracts binary masks from BigBed files loaded into memory.
    
    Pre-converts BigBed entries into dense boolean arrays (stored as float32 tensors)
    for the entire genome. This allows for very fast random access.
    """
    def __init__(self, bigbed_paths: dict[str, Path]):
        """
        In-memory version of BigBedMaskExtractor. Pre-loads entire chromosomes.

        Args:
            bigbed_paths: Dictionary mapping channel names to BigBed file paths.
        """
        self.channels = sorted(bigbed_paths.keys())
        self._cache = {}  # channel -> chrom -> tensor

        for name in self.channels:
            path = str(bigbed_paths[name])
            try:
                bb = pybigtools.open(path)  # type: ignore
                self._cache[name] = {}

                # Get chrom sizes
                chroms = bb.chroms()  # dict[str, int]

                for chrom, size in chroms.items():
                    # Fetch values (coverage)
                    vals = bb.values(chrom, 0, size)
                    # Convert to binary mask
                    arr = (vals > 0).astype(np.float32)
                    tensor = torch.from_numpy(arr)

                    # Share memory
                    tensor.share_memory_()
                    self._cache[name][chrom] = tensor

            except Exception as e:
                raise RuntimeError(f"Failed to load BigBed file {path}: {e}")

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts mask for the given interval from memory.

        Args:
            interval: Genomic interval.

        Returns:
            torch.Tensor: Float32 tensor of shape (Channels, Length).
        """
        extracted_values = []
        for name in self.channels:
            chrom_data = self._cache[name].get(interval.chrom)

            if chrom_data is None:
                # Chrom missing from BigBed but requested
                length = interval.end - interval.start
                vals = torch.zeros(length, dtype=torch.float32)
            else:
                # Assume interval is valid (checked by Sampler)
                vals = chrom_data[interval.start : interval.end]

            extracted_values.append(vals)

        return torch.stack(extracted_values)


class BedMaskExtractor(BaseMaskExtractor):
    """
    Extracts binary masks from text-based BED files.
    
    Loads the entire BED file into memory using InterLap for efficient range queries.
    Suitable for sparse interval data like peaks.
    """
    def __init__(self, bed_paths: dict[str, Path]):
        """
        Args:
            bed_paths: Dictionary mapping channel names to BED file paths.
        """
        self.bed_paths = bed_paths
        self.channels = sorted(bed_paths.keys())
        self._interlaps = {} # channel -> chrom -> InterLap

        for name in self.channels:
            path = Path(self.bed_paths[name])
            self._interlaps[name] = self._load_bed(path)

    def _load_bed(self, path: Path) -> dict[str, InterLap]:
        """Loads BED file into a dictionary of InterLaps (one per chromosome)."""
        intervals_by_chrom = {}
        
        try:
            if path.suffix == ".gz":
                f = gzip.open(path, "rt")
            else:
                f = open(path, "r")
                
            with f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("#", "track", "browser")):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                        
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    
                    if chrom not in intervals_by_chrom:
                        intervals_by_chrom[chrom] = []
                    
                    # InterLap expects closed intervals. BED is [start, end).
                    # Store as [start, end-1].
                    if end > start:
                        intervals_by_chrom[chrom].append((start, end - 1))
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load BED file {path}: {e}")
            
        # Build InterLap objects
        interlaps = {}
        for chrom, intervals in intervals_by_chrom.items():
            interlaps[chrom] = InterLap()
            interlaps[chrom].update(intervals)
            
        return interlaps

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts binary mask for the given interval.

        Args:
            interval: Genomic interval.

        Returns:
            torch.Tensor: Float32 tensor of shape (Channels, Length).
        """
        start = interval.start
        end = interval.end
        length = end - start
        
        extracted_values = []
        for name in self.channels:
            vals = np.zeros(length, dtype=np.float32)
            
            chrom_interlap = self._interlaps[name].get(interval.chrom)
            
            if chrom_interlap is not None:
                # Query with closed interval [start, end-1]
                # If interval is empty (start >= end), skip
                if end > start:
                    overlaps = chrom_interlap.find((start, end - 1))
                    
                    for o in overlaps:
                        h_start, h_end_inclusive = o[0], o[1]
                        h_end = h_end_inclusive + 1
                        
                        # Clip to query interval
                        s = max(0, h_start - start)
                        e = min(length, h_end - start)
                        
                        if s < e:
                            vals[s:e] = 1.0
            
            extracted_values.append(vals)
            
        return torch.from_numpy(np.stack(extracted_values))
