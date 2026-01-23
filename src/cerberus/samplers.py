from typing import Iterator, Protocol
from pathlib import Path
import gzip
import random
from collections import defaultdict
import numpy as np
import pyfaidx
from interlap import InterLap
from .interval import Interval
from .config import SamplerConfig
from .exclude import is_excluded
from .sequence import calculate_gc_content


class Sampler(Protocol):
    """
    Protocol for data samplers.
    
    A sampler defines a set of genomic intervals (samples) and supports:
    - Iteration over intervals.
    - Indexed access.
    - Length reporting.
    - K-fold splitting.
    - Resampling (e.g., for epoch-based randomization).
    """
    def __iter__(self) -> Iterator[Interval]:
        ...
    def __len__(self) -> int:
        ...
    def __getitem__(self, idx: int) -> Interval:
        ...
    def resample(self, seed: int | None = None) -> None:
        ...
    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["Sampler", "Sampler", "Sampler"]:
        ...
    def resolve_interval(self, query: str | tuple | Interval) -> Interval:
        ...


class BaseSampler(Sampler):
    """
    Base class for samplers providing common functionality.
    
    Handles subsetting and basic iteration/access methods.
    """
    _intervals: list[Interval]
    folds: list[dict[str, InterLap]]
    chrom_sizes: dict[str, int]
    exclude_intervals: dict[str, InterLap]

    def __iter__(self) -> Iterator[Interval]:
        for interval in self._intervals:
            yield interval

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def _subset(self, indices: list[int]) -> "SubsetSampler":
        return SubsetSampler(
            [self._intervals[i] for i in indices],
            self.folds,
            self.exclude_intervals,
        )

    def is_excluded(self, chrom: str, start: int, end: int) -> bool:
        """Checks if a region overlaps with the exclusion intervals."""
        return is_excluded(self.exclude_intervals, chrom, start, end)

    def resample(self, seed: int | None = None) -> None:
        """No-op by default."""
        pass

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple[Sampler, Sampler, Sampler]:
        """
        Split the sampler into train, validation, and test sets using K-fold strategy.
        Uses pre-computed fold intervals.
        
        Args:
            test_fold: Index of the fold to use for testing.
            val_fold: Index of the fold to use for validation.
            
        Returns:
            Tuple of (train_sampler, val_sampler, test_sampler).
        """
        folds = self.folds

        test_fold_intervals = folds[test_fold] if test_fold is not None else {}
        val_fold_intervals = folds[val_fold] if val_fold is not None else {}

        train_indices = []
        val_indices = []
        test_indices = []

        def is_in_fold(fold_intervals: dict[str, InterLap], chrom: str, start: int, end: int) -> bool:
            if chrom not in fold_intervals:
                return False
            # Check overlap: InterLap stores closed intervals [start, end]
            # Interval uses [start, end) -> [start, end-1]
            return (start, end - 1) in fold_intervals[chrom]

        for i, interval in enumerate(self._intervals):
            in_test = is_in_fold(test_fold_intervals, interval.chrom, interval.start, interval.end)
            in_val = is_in_fold(val_fold_intervals, interval.chrom, interval.start, interval.end)

            if in_test:
                test_indices.append(i)
            
            if in_val:
                val_indices.append(i)
            
            if not in_test and not in_val:
                train_indices.append(i)

        return (
            self._subset(train_indices),
            self._subset(val_indices),
            self._subset(test_indices),
        )

    def resolve_interval(self, query: str | tuple | Interval) -> Interval:
        """
        Resolves a query into an Interval object.

        Supported formats:
        - Interval object: returned as is.
        - string: "chrom:start-end" (e.g. "chr1:100-200")
        - tuple: (chrom, start, end) (e.g. ("chr1", 100, 200))
        """
        if isinstance(query, Interval):
            return query

        if isinstance(query, str):
            try:
                chrom, coords = query.split(":")
                start, end = map(int, coords.split("-"))
                return Interval(chrom, start, end)
            except ValueError:
                raise ValueError(f"Invalid interval string format: {query}. Expected 'chrom:start-end'.")

        if isinstance(query, (tuple, list)):
            if len(query) < 3:
                raise ValueError(f"Invalid interval tuple: {query}. Expected (chrom, start, end).")
            return Interval(str(query[0]), int(query[1]), int(query[2]))

        raise TypeError(f"Unsupported interval query type: {type(query)}")


class DummySampler(BaseSampler):
    """
    A dummy sampler that does not contain any intervals but provides interval resolution.
    Used when no sampler config is provided.
    """
    def __init__(self, chrom_sizes: dict[str, int]):
        self.chrom_sizes = chrom_sizes
        self.exclude_intervals = {}
        self.folds = []
        self._intervals = []

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        raise NotImplementedError("DummySampler does not support indexing.")


class SubsetSampler(BaseSampler):
    """
    A sampler that wraps a subset of intervals from another sampler.
    Returned by `split_folds`.
    """
    def __init__(
        self,
        intervals: list[Interval],
        folds: list[dict[str, InterLap]],
        exclude_intervals: dict[str, InterLap],
    ):
        self._intervals = intervals
        self.folds = folds
        self.exclude_intervals = exclude_intervals


class RandomSampler(BaseSampler):
    """
    Samples random intervals from the genome, respecting exclusions.
    """

    def __init__(
        self,
        chrom_sizes: dict[str, int],
        padded_size: int,
        num_intervals: int,
        exclude_intervals: dict[str, InterLap] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
        seed: int | None = None,
    ):
        self.chrom_sizes = chrom_sizes
        self.padded_size = padded_size
        self.num_intervals = num_intervals
        self.exclude_intervals = exclude_intervals if exclude_intervals is not None else {}
        self.folds = folds if folds is not None else []
        self._intervals: list[Interval] = []
        self.rng = random.Random(seed)
        self._generate_intervals()

    def _generate_intervals(self):
        chroms = list(self.chrom_sizes.keys())
        weights = [self.chrom_sizes[c] for c in chroms]

        count = 0
        max_attempts = self.num_intervals * 100  # Safety break
        attempts = 0

        while count < self.num_intervals and attempts < max_attempts:
            attempts += 1
            # Weighted choice
            chrom = self.rng.choices(chroms, weights=weights, k=1)[0]
            size = self.chrom_sizes[chrom]

            if size < self.padded_size:
                continue

            start = self.rng.randint(0, size - self.padded_size)
            end = start + self.padded_size

            if not self.is_excluded(chrom, start, end):
                self._intervals.append(Interval(chrom, start, end, "+"))
                count += 1

        if count < self.num_intervals:
            print(
                f"Warning: RandomSampler could only generate {count}/{self.num_intervals} "
                f"intervals after {attempts} attempts."
            )


class GCMatchedSampler(BaseSampler):
    """
    Selects candidates from a candidate_sampler that match the GC content distribution
    of a target_sampler.
    """

    def __init__(
        self,
        target_sampler: Sampler,
        candidate_sampler: Sampler,
        fasta_path: Path | str,
        chrom_sizes: dict[str, int],
        exclude_intervals: dict[str, InterLap],
        folds: list[dict[str, InterLap]] | None = None,
        bins: int = 100,
        match_ratio: float = 1.0,
        seed: int | None = None,
    ):
        self.target_sampler = target_sampler
        self.candidate_sampler = candidate_sampler
        self.fasta_path = Path(fasta_path)
        self.chrom_sizes = chrom_sizes
        self.exclude_intervals = exclude_intervals
        self.folds = folds if folds is not None else []
        self.bins = bins
        self.match_ratio = match_ratio
        self.rng = random.Random(seed)

        # Pre-compute GC content
        self.target_gc = self._compute_gc(self.target_sampler)
        self.candidate_gc = self._compute_gc(self.candidate_sampler)

        self._indices: list[int] = []  # Indices into candidate_sampler
        self.resample()

    def _compute_gc(self, sampler: Sampler) -> list[float]:
        gc_values = []
        fasta = pyfaidx.Fasta(str(self.fasta_path))

        for interval in sampler:
            try:
                # pyfaidx expects 0-based [start:end]
                # interval.start is 0-based, interval.end is exclusive 0-based?
                # Interval class uses 0-based start, 1-based end (BED style)
                # pyfaidx slicing [start:end] is also 0-based half-open.
                seq_obj = fasta[interval.chrom][interval.start : interval.end]
                seq = str(seq_obj)
                gc_values.append(calculate_gc_content(seq))
            except Exception:
                gc_values.append(0.0)

        return gc_values

    def resample(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)

        # 1. Bin target GC
        target_hist = defaultdict(int)
        for gc in self.target_gc:
            bin_idx = min(int(gc * self.bins), self.bins - 1)
            target_hist[bin_idx] += 1

        # 2. Group candidate indices by bin
        candidate_bins = defaultdict(list)
        for idx, gc in enumerate(self.candidate_gc):
            bin_idx = min(int(gc * self.bins), self.bins - 1)
            candidate_bins[bin_idx].append(idx)

        self._indices = []

        # 3. Match
        for bin_idx, count in target_hist.items():
            needed = int(count * self.match_ratio)
            candidates = candidate_bins[bin_idx]

            if not candidates:
                continue

            if len(candidates) >= needed:
                # Sample without replacement
                selected = self.rng.sample(candidates, needed)
            else:
                # Sample with replacement
                selected = self.rng.choices(candidates, k=needed)

            self._indices.extend(selected)

        self.rng.shuffle(self._indices)

    def __iter__(self) -> Iterator[Interval]:
        for idx in self._indices:
            yield self.candidate_sampler[idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Interval:
        real_idx = self._indices[idx]
        return self.candidate_sampler[real_idx]

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["GCMatchedSampler", "GCMatchedSampler", "GCMatchedSampler"]:
        
        target_splits = self.target_sampler.split_folds(test_fold, val_fold)
        candidate_splits = self.candidate_sampler.split_folds(test_fold, val_fold)

        return (
            GCMatchedSampler(
                target_splits[0],
                candidate_splits[0],
                self.fasta_path,
                self.chrom_sizes,
                self.exclude_intervals,
                self.folds,
                self.bins,
                self.match_ratio,
                self.rng.randint(0, 10000),
            ),
            GCMatchedSampler(
                target_splits[1],
                candidate_splits[1],
                self.fasta_path,
                self.chrom_sizes,
                self.exclude_intervals,
                self.folds,
                self.bins,
                self.match_ratio,
                self.rng.randint(0, 10000),
            ),
            GCMatchedSampler(
                target_splits[2],
                candidate_splits[2],
                self.fasta_path,
                self.chrom_sizes,
                self.exclude_intervals,
                self.folds,
                self.bins,
                self.match_ratio,
                self.rng.randint(0, 10000),
            ),
        )


class IntervalSampler(BaseSampler):
    """
    Samples from a list of genomic intervals provided in a file.
    
    Supports BED and narrowPeak formats.
    """
    def __init__(
        self,
        file_path: Path,
        chrom_sizes: dict[str, int],
        padded_size: int,
        exclude_intervals: dict[str, InterLap] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
    ):
        """
        Args:
            file_path: Path to the interval file (BED/narrowPeak).
            chrom_sizes: Dictionary of chromosome sizes.
            padded_size: Desired length of the intervals.
            exclude_intervals: Dictionary of regions to exclude.
            folds: Fold definitions for cross-validation.
        """
        self.file_path = Path(file_path)
        self.padded_size = padded_size
        self.chrom_sizes = chrom_sizes
        self.exclude_intervals = exclude_intervals if exclude_intervals is not None else {}
        self.folds = folds if folds is not None else []
        self._intervals: list[Interval] = []
        self._load()
        self._filter_excludes()

    def _load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Check extension for narrowPeak
        name = self.file_path.name
        if name.endswith(".narrowPeak") or name.endswith(".narrowPeak.gz"):
            self._load_narrowPeak()
        else:
            self._load_bed()

    def _read_file(self) -> Iterator[list[str]]:
        if self.file_path.suffix == ".gz":
            with gzip.open(self.file_path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("#", "track", "browser")):
                        continue
                    yield line.split()
        else:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("#", "track", "browser")):
                        continue
                    yield line.split()

    def _apply_centering(
        self, start: int, end: int, center: int | None = None
    ) -> tuple[int, int]:
        if self.padded_size is not None:
            if center is None:
                center = (start + end) // 2

            # Start is center - padded_size/2
            start = center - self.padded_size // 2
            end = start + self.padded_size
        return start, end

    def _is_valid(self, chrom: str, start: int, end: int) -> bool:
        if start < 0:
            return False
        if chrom not in self.chrom_sizes:
            return False
        if end > self.chrom_sizes[chrom]:
            return False

        return True

    def _load_bed(self):
        for parts in self._read_file():
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid BED line: expected at least 3 columns, got {len(parts)}. Line: {' '.join(parts)}"
                )

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            strand = "+"
            if len(parts) >= 6:
                s = parts[5]
                if s in ("+", "-"):
                    strand = s

            start, end = self._apply_centering(start, end)

            if not self._is_valid(chrom, start, end):
                continue

            self._intervals.append(Interval(chrom, start, end, strand))

    def _load_narrowPeak(self):
        for parts in self._read_file():
            if len(parts) < 10:
                raise ValueError(
                    f"Invalid narrowPeak line: expected at least 10 columns, got {len(parts)}. Line: {' '.join(parts)}"
                )

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            # narrowPeak strand is col 6 (idx 5)
            strand = "+"
            s = parts[5]
            if s in ("+", "-"):
                strand = s

            # Summit in col 10 (idx 9)
            summit_offset = int(parts[9])

            center = None
            if summit_offset != -1:
                center = start + summit_offset

            start, end = self._apply_centering(start, end, center)

            if not self._is_valid(chrom, start, end):
                continue

            self._intervals.append(Interval(chrom, start, end, strand))

    def _filter_excludes(self):
        if not self.exclude_intervals or not self._intervals:
            return

        indices_to_remove = set()

        for i, interval in enumerate(self._intervals):
            if self.is_excluded(interval.chrom, interval.start, interval.end):
                indices_to_remove.add(i)

        if indices_to_remove:
            self._intervals = [
                interval
                for i, interval in enumerate(self._intervals)
                if i not in indices_to_remove
            ]


class SlidingWindowSampler(BaseSampler):
    """
    Generates samples by sliding a window across the genome.
    
    Useful for genome-wide prediction tasks.
    """
    def __init__(
        self,
        chrom_sizes: dict[str, int],
        padded_size: int,
        stride: int,
        exclude_intervals: dict[str, InterLap] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
    ):
        """
        Args:
            chrom_sizes: Dictionary of chromosome sizes.
            padded_size: Size of the window.
            stride: Step size.
            exclude_intervals: Dictionary of regions to exclude.
            folds: Fold definitions.
        """
        self.chrom_sizes = chrom_sizes
        self.padded_size = padded_size
        self.stride = stride
        self.exclude_intervals = exclude_intervals if exclude_intervals is not None else {}
        self.folds = folds if folds is not None else []
        self._intervals: list[Interval] = []
        self._generate_intervals()

    def _generate_intervals(self):
        # Sort chromosomes to ensure deterministic order
        for chrom in sorted(self.chrom_sizes.keys()):
            size = self.chrom_sizes[chrom]
            for start in range(0, size - self.padded_size + 1, self.stride):
                end = start + self.padded_size
                if not self.is_excluded(chrom, start, end):
                    self._intervals.append(Interval(chrom, start, end, "+"))


class MultiSampler(BaseSampler):
    """
    Combines multiple samplers with optional scaling factors.
    Allows for mixing peaks and negatives with specific ratios.
    """

    def __init__(
        self,
        samplers: list[Sampler],
        chrom_sizes: dict[str, int],
        exclude_intervals: dict[str, InterLap],
        scaling_factors: list[float] | None = None,
    ):
        """
        Args:
            samplers: List of Sampler instances.
            chrom_sizes: Dictionary of chromosome sizes.
            exclude_intervals: Dictionary of excluded regions.
            scaling_factors: List of floats, one per sampler.
                             - 1.0: Use all samples.
                             - < 1.0: Subsample (e.g., 0.5 uses 50%).
                                    Implemented as random sampling WITHOUT replacement.
                             - > 1.0: Oversample (e.g., 2.0 duplicates samples).
                                    Implemented as random sampling WITH replacement.
        """
        self.samplers = samplers
        self.chrom_sizes = chrom_sizes
        self.exclude_intervals = exclude_intervals
        self.scaling_factors = (
            scaling_factors if scaling_factors is not None else [1.0] * len(samplers)
        )

        if len(self.samplers) != len(self.scaling_factors):
            raise ValueError("Number of samplers must match number of scaling factors")

        self._indices: list[tuple[int, int]] = []  # List of (sampler_idx, interval_idx)
        self.resample()

    def resample(self, seed: int | None = None):
        """
        Regenerates the list of indices based on scaling factors.
        Call this at the start of each epoch to get a fresh random subset for subsampled inputs.
        Args:
            seed: Optional random seed for reproducible or rank-specific sampling.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        self._indices = []
        for i, sampler in enumerate(self.samplers):
            n_total = len(sampler)
            scaling = self.scaling_factors[i]
            n_sample = int(n_total * scaling)

            if n_sample == 0:
                continue

            if scaling <= 1.0:
                # Subsample without replacement
                indices = rng.choice(n_total, n_sample, replace=False)
            else:
                # Oversample with replacement
                indices = rng.choice(n_total, n_sample, replace=True)

            for idx in indices:
                self._indices.append((i, int(idx)))

        # Shuffle the mixed indices
        rng.shuffle(self._indices)

    def __iter__(self) -> Iterator[Interval]:
        for sampler_idx, interval_idx in self._indices:
            yield self.samplers[sampler_idx][interval_idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Interval:
        sampler_idx, interval_idx = self._indices[idx]
        return self.samplers[sampler_idx][interval_idx]

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["MultiSampler", "MultiSampler", "MultiSampler"]:
        """
        Splits each sub-sampler and returns new MultiSamplers composed of the splits.
        """
        train_samplers, val_samplers, test_samplers = [], [], []

        for sampler in self.samplers:
            train, val, test = sampler.split_folds(test_fold, val_fold)
            train_samplers.append(train)
            val_samplers.append(val)
            test_samplers.append(test)

        return (
            MultiSampler(
                train_samplers,
                self.chrom_sizes,
                self.exclude_intervals,
                self.scaling_factors,
            ),
            MultiSampler(
                val_samplers,
                self.chrom_sizes,
                self.exclude_intervals,
                self.scaling_factors,
            ),
            MultiSampler(
                test_samplers,
                self.chrom_sizes,
                self.exclude_intervals,
                self.scaling_factors,
            ),
        )


def create_sampler(
    config: dict | SamplerConfig,
    chrom_sizes: dict[str, int],
    exclude_intervals: dict[str, InterLap],
    folds: list[dict[str, InterLap]],
    fasta_path: Path | str | None = None,
) -> Sampler:
    """
    Factory function to create a Sampler from a config.
    
    Args:
        config: Sampler configuration dictionary.
        chrom_sizes: Chromosome sizes dictionary.
        exclude_intervals: Dictionary of excluded regions.
        folds: List of fold definitions.
        fasta_path: Path to the genome FASTA file (required for GCMatchedSampler).

    Scaling Config (for MultiSampler):
        The `scaling` parameter in sub-sampler config supports:
        - float: Direct ratio (e.g., 1.0, 0.5).
        - "min": Matches the count of the smallest sampler.
        - "max": Matches the count of the largest sampler.
        - "count:<int>": Sets a fixed number of samples.
        
    Returns:
        Sampler: An instantiated sampler object.
        
    Raises:
        ValueError: If sampler_type is unsupported.
    """
    sampler_type = config["sampler_type"]
    sampler_args = config["sampler_args"]
    padded_size = config["padded_size"]

    if sampler_type == "interval":
        file_path = sampler_args["intervals_path"]

        return IntervalSampler(
            file_path=file_path,
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            exclude_intervals=exclude_intervals,
            folds=folds,
        )

    elif sampler_type == "sliding_window":
        return SlidingWindowSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            stride=sampler_args["stride"],
            exclude_intervals=exclude_intervals,
            folds=folds,
        )

    elif sampler_type == "random":
        num_intervals = sampler_args["num_intervals"]
        return RandomSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            num_intervals=num_intervals,
            exclude_intervals=exclude_intervals,
            folds=folds,
        )

    elif sampler_type == "dummy":
        return DummySampler(chrom_sizes=chrom_sizes)

    elif sampler_type == "gc_matched":
        if fasta_path is None:
            raise ValueError("GCMatchedSampler requires 'fasta_path' to be provided.")

        target_conf = sampler_args["target_sampler"]
        candidate_conf = sampler_args["candidate_sampler"]

        # Recursive creation
        target_full_conf = {
            "sampler_type": target_conf["type"],
            "sampler_args": target_conf["args"],
            "padded_size": padded_size,
        }
        candidate_full_conf = {
            "sampler_type": candidate_conf["type"],
            "sampler_args": candidate_conf["args"],
            "padded_size": padded_size,
        }

        target_sampler = create_sampler(
            target_full_conf, chrom_sizes, exclude_intervals, folds, fasta_path
        )
        candidate_sampler = create_sampler(
            candidate_full_conf, chrom_sizes, exclude_intervals, folds, fasta_path
        )

        return GCMatchedSampler(
            target_sampler=target_sampler,
            candidate_sampler=candidate_sampler,
            fasta_path=fasta_path,
            chrom_sizes=chrom_sizes,
            exclude_intervals=exclude_intervals,
            folds=folds,
            bins=sampler_args.get("bins", 100),
            match_ratio=sampler_args.get("match_ratio", 1.0),
        )

    elif sampler_type == "multi":
        samplers = []
        raw_scalings = []

        for sub_config in sampler_args["samplers"]:
            # Construct sub-config inheriting top-level values
            sub_sampler_type = sub_config["type"]
            sub_sampler_args = sub_config["args"]
            scaling = sub_config.get("scaling", 1.0)

            child_config = {
                "sampler_type": sub_sampler_type,
                "sampler_args": sub_sampler_args,
                "padded_size": padded_size,
            }

            sub_sampler = create_sampler(
                child_config, chrom_sizes, exclude_intervals, folds=folds, fasta_path=fasta_path
            )
            samplers.append(sub_sampler)
            raw_scalings.append(scaling)

        # Resolve scaling factors
        scaling_factors = []
        sampler_lengths = [len(s) for s in samplers]
        
        # Calculate reference lengths
        non_zero_lengths = [l for l in sampler_lengths if l > 0]
        min_len = min(non_zero_lengths) if non_zero_lengths else 0
        max_len = max(sampler_lengths) if sampler_lengths else 0

        for i, scaling in enumerate(raw_scalings):
            current_len = sampler_lengths[i]
            
            if isinstance(scaling, (int, float)):
                scaling_factors.append(float(scaling))
            elif isinstance(scaling, str):
                if current_len == 0:
                    scaling_factors.append(0.0)
                    continue

                if scaling == "min":
                    scaling_factors.append(min_len / current_len)
                elif scaling == "max":
                    scaling_factors.append(max_len / current_len)
                elif scaling.startswith("count:"):
                    try:
                        target = int(scaling.split(":")[1])
                        scaling_factors.append(target / current_len)
                    except ValueError:
                        raise ValueError(f"Invalid scaling format '{scaling}'. Expected 'count:<int>'.")
                else:
                    raise ValueError(f"Unsupported scaling string '{scaling}'. Supported: 'min', 'max', 'count:<int>'.")
            else:
                raise ValueError(f"Unsupported scaling type {type(scaling)}. Expected int, float, or str.")

        return MultiSampler(
            samplers=samplers,
            chrom_sizes=chrom_sizes,
            exclude_intervals=exclude_intervals,
            scaling_factors=scaling_factors,
        )

    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")
