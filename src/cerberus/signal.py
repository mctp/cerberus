from pathlib import Path
from typing import Protocol
import logging
import numpy as np
import pybigtools
import torch
from cerberus.interval import Interval
from cerberus.mask import BigBedMaskExtractor, InMemoryBigBedMaskExtractor, BedMaskExtractor

logger = logging.getLogger(__name__)


class BaseSignalExtractor(Protocol):
    """
    Protocol for signal extractors.
    
    Classes implementing this protocol must provide an `extract` method that returns
    a signal tensor (usually coverage or fold-change) for a given genomic interval.
    """
    def extract(self, interval: Interval) -> torch.Tensor: ...


class SignalExtractor(BaseSignalExtractor):
    """
    Extracts raw signal from BigWig files on-the-fly.

    This extractor handles multiple channels (one BigWig file per channel).
    It reads specific intervals as requested, padding with zeros if necessary.
    
    No binning, logging or resizing is performed here; those are handled by transforms.
    """
    def __init__(self, bigwig_paths: dict[str, Path]):
        """
        Args:
            bigwig_paths: Dictionary mapping channel names to BigWig file paths.
                          e.g., {'H3K4me3': Path('h3k4me3.bw'), 'DNase': Path('dnase.bw')}
        """
        self.bigwig_paths = bigwig_paths
        self.channels = sorted(bigwig_paths.keys())  # Ensure consistent order
        self._bigwig_files = None

    def _load(self):
        """Lazy loader for BigWig handles."""
        logger.debug(f"Lazy-loading {len(self.channels)} BigWig file(s)...")
        self._bigwig_files = {}
        for name in self.channels:
            path = str(self.bigwig_paths[name])
            try:
                self._bigwig_files[name] = pybigtools.open(path)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to open BigWig file {path}: {e}")

    def __getstate__(self):
        """Pickle support: exclude file handles to allow safe multiprocessing."""
        state = self.__dict__.copy()
        state["_bigwig_files"] = None
        return state

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts signal for the given interval.

        Args:
            interval: Genomic interval specifying chromosome, start, and end.

        Returns:
            torch.Tensor: Float32 tensor of shape (Channels, Length).
                          Channels are ordered alphabetically by their keys in `bigwig_paths`.
        """
        # Lazy load (fork safety via lazy init + __getstate__)
        if self._bigwig_files is None:
            self._load()

        start = interval.start
        end = interval.end
        length = end - start

        # Extract signals
        extracted_values = []
        for name in self.channels:
            # We guaranteed _bigwig_files is not None and populated in _load()
            bw = self._bigwig_files[name]  # type: ignore
            try:
                # pybigtools.values(chrom, start, end)
                vals = bw.values(interval.chrom, start, end)
                vals = np.array(vals, dtype=np.float32)
                vals = np.nan_to_num(vals)

                # If truncated, pad with zeros
                if len(vals) < length:
                    pad_len = length - len(vals)
                    vals = np.pad(vals, (0, pad_len), constant_values=0)

            except RuntimeError:
                # Chromosome not found or other read error -> zeros
                logger.debug(f"Chrom {interval.chrom} not found in BigWig '{name}', returning zeros")
                vals = np.zeros(length, dtype=np.float32)
            except (Exception, BaseException) as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                # pyo3_runtime.PanicException (Rust panics from bigtools) inherits from
                # BaseException, not Exception, so it must be caught here explicitly.
                logger.debug(f"Error reading BigWig '{name}' at {interval}, returning zeros")
                vals = np.zeros(length, dtype=np.float32)

            extracted_values.append(vals)

        # Stack: (Channels, Length)
        signal_tensor = np.stack(extracted_values)
        return torch.from_numpy(signal_tensor)


class UniversalExtractor(BaseSignalExtractor):
    """
    Intelligently routes input channels to the appropriate extractor based on file extension.
    Supports BigWig (.bw), BigBed (.bb), and BED (.bed) files.
    """
    def __init__(self, paths: dict[str, Path], in_memory: bool = False):
        self.paths = paths
        self.channels = sorted(paths.keys())
        self.in_memory = in_memory
        
        # Group channels by type
        self.bw_paths = {}
        self.bb_paths = {}
        self.bed_paths = {}
        
        for name in self.channels:
            path = Path(paths[name])
            suffix = path.suffix.lower()
            name_str = str(name)
            
            if suffix in ('.bw', '.bigwig'):
                self.bw_paths[name_str] = path
            elif suffix in ('.bb', '.bigbed'):
                self.bb_paths[name_str] = path
            elif suffix in ('.bed', '.bed.gz', '.gz'): 
                self.bed_paths[name_str] = path
            else:
                # Default to BigWig if unknown
                self.bw_paths[name_str] = path

        logger.debug(f"UniversalExtractor routing: {len(self.bw_paths)} BigWig, {len(self.bb_paths)} BigBed, {len(self.bed_paths)} BED")
        self.extractors = {}

        if self.bw_paths:
            if self.in_memory:
                self.extractors['bw'] = InMemorySignalExtractor(self.bw_paths)
            else:
                self.extractors['bw'] = SignalExtractor(self.bw_paths)
                
        if self.bb_paths:
            if self.in_memory:
                self.extractors['bb'] = InMemoryBigBedMaskExtractor(self.bb_paths)
            else:
                self.extractors['bb'] = BigBedMaskExtractor(self.bb_paths)
                
        if self.bed_paths:
            # BedMaskExtractor is always in-memory (InterLap)
            self.extractors['bed'] = BedMaskExtractor(self.bed_paths)

    def extract(self, interval: Interval) -> torch.Tensor:
        # We need to return channels in sorted order of self.channels
        
        # Collect results from all extractors
        results = {}
        if 'bw' in self.extractors:
            results['bw'] = self.extractors['bw'].extract(interval) # (N_bw, L)
            
        if 'bb' in self.extractors:
            results['bb'] = self.extractors['bb'].extract(interval)
            
        if 'bed' in self.extractors:
            results['bed'] = self.extractors['bed'].extract(interval)
            
        # Helper to map channel name back to result tensor index
        bw_keys = sorted(self.bw_paths.keys())
        bb_keys = sorted(self.bb_paths.keys())
        bed_keys = sorted(self.bed_paths.keys())
        
        final_tensors = []
        
        for name in self.channels:
            if name in self.bw_paths:
                idx = bw_keys.index(name)
                final_tensors.append(results['bw'][idx])
            elif name in self.bb_paths:
                idx = bb_keys.index(name)
                final_tensors.append(results['bb'][idx])
            elif name in self.bed_paths:
                idx = bed_keys.index(name)
                final_tensors.append(results['bed'][idx])
                
        return torch.stack(final_tensors)


class InMemorySignalExtractor(BaseSignalExtractor):
    """
    Extracts signal from BigWig files loaded into memory.

    Pre-loads entire chromosomes into shared memory tensors for fast access.
    Best for smaller genomes or when sufficient RAM is available.
    """
    def __init__(self, bigwig_paths: dict[str, Path]):
        """
        Args:
            bigwig_paths: Dictionary mapping channel names to BigWig file paths.
        """
        self.channels = sorted(bigwig_paths.keys())
        self._cache = {}  # channel -> chrom -> tensor

        logger.info(f"Loading {len(self.channels)} BigWig file(s) into memory...")
        for name in self.channels:
            path = str(bigwig_paths[name])
            try:
                bw = pybigtools.open(path)  # type: ignore
                self._cache[name] = {}

                # Get chrom sizes
                chroms = bw.chroms()

                for chrom, size in chroms.items():
                    vals = bw.values(chrom, 0, size)
                    arr = np.array(vals, dtype=np.float32)
                    arr = np.nan_to_num(arr)
                    tensor = torch.from_numpy(arr)
                    # Invoke share_memory_() on the tensor (from torch.Tensor).
                    # This moves the underlying storage to shared memory (if not already),
                    # allowing the tensor to be shared across processes (e.g. DataLoader workers)
                    # without copying the data, preventing memory explosion.
                    tensor.share_memory_()
                    self._cache[name][chrom] = tensor

            except Exception as e:
                raise RuntimeError(f"Failed to load BigWig file {path}: {e}")

    def extract(self, interval: Interval) -> torch.Tensor:
        """
        Extracts signal for the given interval from memory.

        Args:
            interval: Genomic interval.

        Returns:
            torch.Tensor: Float32 tensor of shape (Channels, Length).
        """
        extracted_values = []
        for name in self.channels:
            chrom_data = self._cache[name].get(interval.chrom)

            if chrom_data is None:
                # Chrom missing from BigWig but requested
                length = interval.end - interval.start
                vals = torch.zeros(length, dtype=torch.float32)
            else:
                # Assume interval is valid (checked by Sampler)
                vals = chrom_data[interval.start : interval.end]

            extracted_values.append(vals)

        return torch.stack(extracted_values)
