import copy
import gzip
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
from interlap import InterLap

from .complexity import compute_hist, compute_intervals_complexity, get_bin_index
from .config import SamplerConfig
from .exclude import is_excluded
from .interval import Interval

logger = logging.getLogger(__name__)


def _select_from_bins(
    target_hist: dict[tuple[int, ...], int],
    candidate_bins: dict[tuple[int, ...], list[int]],
    candidate_ratio: float,
    rng: random.Random,
) -> list[int]:
    """
    Helper to select indices from candidate bins to match target histogram counts.
    """
    selected_indices = []
    for bin_idx, count in target_hist.items():
        needed = int(count * candidate_ratio)
        candidates = candidate_bins.get(bin_idx, [])

        if not candidates:
            continue

        if len(candidates) >= needed:
            selected = rng.sample(candidates, needed)
        else:
            selected = rng.choices(candidates, k=needed)

        selected_indices.extend(selected)
    return selected_indices


def generate_sub_seeds(seed: int, n: int) -> list[int]:
    """
    Generates n independent seeds from a master seed.
    """
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]


def match_bin_counts(
    target_metrics: list[float] | np.ndarray,
    candidate_metrics: list[float] | np.ndarray,
    bins: int,
    candidate_ratio: float,
    rng: random.Random,
) -> list[int]:
    """
    Selects indices from candidate_metrics to match the distribution of target_metrics.

    This function bins the metrics (1D or ND) and selects candidates to match
    the count of targets in each bin, scaled by candidate_ratio.

    Args:
        target_metrics: Metric values for target samples. Shape (N,) or (N, D).
        candidate_metrics: Metric values for candidate samples. Shape (M,) or (M, D).
        bins: Number of bins per dimension. Values are assumed to be in [0, 1].
        candidate_ratio: Ratio of candidates to targets per bin.
        rng: Random number generator.

    Returns:
        List of selected indices from candidate_metrics.
    """
    # Ensure numpy arrays
    t_metrics = np.asarray(target_metrics)
    c_metrics = np.asarray(candidate_metrics)

    # Handle 1D case by reshaping to (N, 1)
    if t_metrics.ndim == 1:
        t_metrics = t_metrics.reshape(-1, 1)
    if c_metrics.ndim == 1:
        c_metrics = c_metrics.reshape(-1, 1)

    # 1. Bin targets
    target_hist = defaultdict(int)
    for row in t_metrics:
        bin_idx = get_bin_index(row, bins)
        if bin_idx is not None:
            target_hist[bin_idx] += 1

    # 2. Bin candidates
    candidate_bins = defaultdict(list)
    for i, row in enumerate(c_metrics):
        bin_idx = get_bin_index(row, bins)
        if bin_idx is not None:
            candidate_bins[bin_idx].append(i)

    # 3. Match
    selected_indices = _select_from_bins(
        target_hist, candidate_bins, candidate_ratio, rng
    )
    rng.shuffle(selected_indices)
    return selected_indices


def partition_intervals_by_fold(
    intervals: Iterable[Interval],
    folds: list[dict[str, InterLap]],
    test_fold: int | None,
    val_fold: int | None,
) -> tuple[list[Interval], list[Interval], list[Interval]]:
    """
    Partitions intervals into train, validation, and test lists based on folds.

    If test_fold and val_fold overlap (or are the same), intervals in the overlapping region
    will be included in BOTH test_intervals and val_intervals.
    train_intervals will contain intervals that are in neither test_fold nor val_fold.

    Args:
        intervals: Iterable of intervals to partition.
        folds: List of fold definitions.
        test_fold: Index of the test fold.
        val_fold: Index of the validation fold.

    Returns:
        Tuple of (train_intervals, val_intervals, test_intervals).
    """
    test_fold_intervals = folds[test_fold] if test_fold is not None else {}
    val_fold_intervals = folds[val_fold] if val_fold is not None else {}

    train_intervals = []
    val_intervals = []
    test_intervals = []

    def is_in_fold(
        fold_intervals: dict[str, InterLap], chrom: str, start: int, end: int
    ) -> bool:
        if chrom not in fold_intervals:
            return False
        # Check overlap: InterLap stores closed intervals [start, end]
        # Interval uses [start, end) -> [start, end-1]
        return (start, end - 1) in fold_intervals[chrom]

    for interval in intervals:
        in_test = is_in_fold(
            test_fold_intervals, interval.chrom, interval.start, interval.end
        )
        in_val = is_in_fold(
            val_fold_intervals, interval.chrom, interval.start, interval.end
        )

        if in_test:
            test_intervals.append(interval)

        if in_val:
            val_intervals.append(interval)

        if not in_test and not in_val:
            train_intervals.append(interval)

    return train_intervals, val_intervals, test_intervals


#### Sampler Protocol and Base Classes ####


class Sampler(Protocol):
    """
    Protocol for data samplers.

    A sampler defines a set of genomic intervals (samples) and supports:
    - Iteration over intervals.
    - Indexed access.
    - Length reporting.
    - K-fold splitting into train/val/test samplers.
    - Resampling (e.g., for epoch-based randomization).
    """

    def __iter__(self) -> Iterator[Interval]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Interval: ...
    def resample(self, seed: int | None = None) -> None:
        """
        Resamples the intervals for the next epoch/iteration.

        This method is used by dynamic samplers (e.g. MultiSampler, ComplexityMatchedSampler)
        to regenerate or re-shuffle the list of intervals. This is critical for:
        1. Subsampling/Oversampling: Generating a new random subset of data.
        2. Shuffling: Ensuring a different order of examples (if not handled by DataLoader).

        Args:
            seed: Optional random seed for reproducibility. If None, behavior depends on implementation.
        """
        ...

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["Sampler", "Sampler", "Sampler"]: ...

    def get_interval_source(self, idx: int) -> str:
        """Return the class name of the sampler that produced interval ``idx``."""
        ...


class BaseSampler(Sampler):
    """
    Base class for samplers providing common configuration and exclusion logic.

    Do not instantiate directly; extend for specific sampler implementations.
    """

    def __init__(
        self,
        chrom_sizes: dict[str, int] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
    ):
        self.chrom_sizes = chrom_sizes if chrom_sizes is not None else {}
        self.folds = folds if folds is not None else []
        self.exclude_intervals = (
            exclude_intervals if exclude_intervals is not None else {}
        )

    def is_excluded(self, chrom: str, start: int, end: int) -> bool:
        """Checks if a region overlaps with the exclusion intervals."""
        return is_excluded(self.exclude_intervals, chrom, start, end)

    def _update_seed(self, seed: int | None) -> None:
        """
        Updates the seed and random number generator.

        If seed is provided, uses it.
        If seed is None, generates a new seed from the existing RNG state.
        Assumes self.rng is initialized.
        """
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(self.seed)
        else:
            # Advance seed using current RNG state
            self.seed = self.rng.getrandbits(32)
            self.rng = random.Random(self.seed)

    def get_interval_source(self, idx: int) -> str:
        """Return the class name of this sampler.

        Subclasses like :class:`MultiSampler` override this to report which
        sub-sampler produced the interval at ``idx``.
        """
        return type(self).__name__

    def resample(self, seed: int | None = None) -> None:
        """
        No-op by default.
        """
        pass

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple[Sampler, Sampler, Sampler]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Interval]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Interval:
        raise NotImplementedError


class ProxySampler(BaseSampler):
    """
    Base class for samplers that wrap another sampler and provide index-based access.

    Do not instantiate directly; extend for specific proxy sampler implementations.
    """

    def __init__(
        self,
        chrom_sizes: dict[str, int] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
    ):
        super().__init__(
            chrom_sizes=chrom_sizes, folds=folds, exclude_intervals=exclude_intervals
        )
        self._source_sampler: Sampler | None = None
        self._indices: list[int] = []

    def __iter__(self) -> Iterator[Interval]:
        if self._source_sampler is None:
            raise IndexError("Source sampler not initialized")
        for idx in self._indices:
            yield self._source_sampler[idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Interval:
        if self._source_sampler is None:
            raise IndexError("Source sampler not initialized")
        real_idx = self._indices[idx]
        return self._source_sampler[real_idx]


class MultiSampler(BaseSampler):
    """
    Combines multiple samplers into a single stream.
    """

    def __init__(
        self,
        samplers: list[Sampler],
        chrom_sizes: dict[str, int] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        seed: int = 42,
        generate_on_init: bool = True,
    ):
        """
        Args:
            samplers: List of Sampler instances.
            chrom_sizes: Dictionary of chromosome sizes.
            folds: List of fold definitions.
            exclude_intervals: Dictionary of excluded regions.
            seed: Random seed for initialization.
            generate_on_init: Whether to generate samples immediately (default: True).
        """
        if chrom_sizes is None:
            if samplers and hasattr(samplers[0], "chrom_sizes"):
                chrom_sizes = samplers[0].chrom_sizes  # type: ignore
            else:
                chrom_sizes = {}

        if folds is None:
            if samplers and hasattr(samplers[0], "folds"):
                folds = samplers[0].folds  # type: ignore
            else:
                folds = []

        if exclude_intervals is None:
            if samplers and hasattr(samplers[0], "exclude_intervals"):
                exclude_intervals = samplers[0].exclude_intervals  # type: ignore
            else:
                exclude_intervals = {}

        super().__init__(
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        self.samplers = samplers
        self._indices: list[tuple[int, int]] = []  # List of (sampler_idx, interval_idx)
        self.seed: int = seed
        self.rng = random.Random(seed)
        if generate_on_init:
            self.resample(seed=seed)

    def resample(self, seed: int | None = None) -> None:
        """
        Regenerates the list of indices.
        """
        self._update_seed(seed)

        # Propagate resample to sub-samplers (e.g. ComplexityMatchedSampler needs to pick new candidates)
        sub_seeds = generate_sub_seeds(self.seed, len(self.samplers))
        for sampler, sub_seed in zip(self.samplers, sub_seeds, strict=True):
            sampler.resample(sub_seed)

        self._indices = []
        for i, sampler in enumerate(self.samplers):
            n_total = len(sampler)
            for idx in range(n_total):
                self._indices.append((i, idx))

        # Shuffle the mixed indices
        self.rng.shuffle(self._indices)

    def __iter__(self) -> Iterator[Interval]:
        for sampler_idx, interval_idx in self._indices:
            yield self.samplers[sampler_idx][interval_idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Interval:
        sampler_idx, interval_idx = self._indices[idx]
        return self.samplers[sampler_idx][interval_idx]

    def get_interval_source(self, idx: int) -> str:
        """Return the class name of the sub-sampler that produced interval ``idx``.

        For a :class:`PeakSampler`, ``samplers[0]`` is an
        :class:`IntervalSampler` (peaks) and subsequent samplers are background
        generators (e.g. :class:`ComplexityMatchedSampler`).  The returned
        string identifies which sub-sampler produced the interval, enabling
        callers to apply source-specific logic.

        The source is preserved correctly after :meth:`split_folds` because
        that method returns new :class:`MultiSampler` instances with the same
        sub-sampler ordering.

        Returns:
            Class name of the originating sub-sampler
            (e.g. ``"IntervalSampler"``, ``"ComplexityMatchedSampler"``).
        """
        sampler_idx, _ = self._indices[idx]
        return type(self.samplers[sampler_idx]).__name__

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

        s1, s2, s3 = generate_sub_seeds(self.seed, 3)

        return (
            MultiSampler(
                train_samplers,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                seed=s1,
            ),
            MultiSampler(
                val_samplers,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                seed=s2,
            ),
            MultiSampler(
                test_samplers,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                seed=s3,
            ),
        )


#### Concrete ListSamplers ####


class ListSampler(BaseSampler):
    """
    Base class for samplers that store a concrete list of intervals.
    """

    def __init__(
        self,
        intervals: list[Interval] | None = None,
        chrom_sizes: dict[str, int] | None = None,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
    ):
        super().__init__(
            chrom_sizes=chrom_sizes, folds=folds, exclude_intervals=exclude_intervals
        )
        self._intervals = intervals if intervals is not None else []

    def __iter__(self) -> Iterator[Interval]:
        yield from self._intervals

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def _subset(self, indices: list[int]) -> "ListSampler":
        return ListSampler(
            intervals=[self._intervals[i] for i in indices],
            chrom_sizes=self.chrom_sizes,
            folds=self.folds,
            exclude_intervals=self.exclude_intervals,
        )

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple[Sampler, Sampler, Sampler]:
        """
        Split the sampler into train, validation, and test sets using K-fold strategy.
        Uses pre-computed fold intervals.
        """
        train, val, test = partition_intervals_by_fold(
            self._intervals, self.folds, test_fold, val_fold
        )

        return (
            ListSampler(train, self.chrom_sizes, self.folds, self.exclude_intervals),
            ListSampler(val, self.chrom_sizes, self.folds, self.exclude_intervals),
            ListSampler(test, self.chrom_sizes, self.folds, self.exclude_intervals),
        )


class RandomSampler(BaseSampler):
    """
    Samples random intervals from the genome, respecting exclusions.
    """

    MAX_ATTEMPT_MULTIPLIER = 100

    def __init__(
        self,
        chrom_sizes: dict[str, int],
        padded_size: int,
        num_intervals: int,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        regions: dict[str, InterLap] | None = None,
        seed: int = 42,
        generate_on_init: bool = True,
    ):
        super().__init__(
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        self.padded_size = padded_size
        self.num_intervals = num_intervals
        self.regions = regions
        self.seed: int = seed
        self.rng = random.Random(seed)
        self._intervals: list[Interval] = []
        if generate_on_init:
            self.resample(seed)

    def __iter__(self) -> Iterator[Interval]:
        yield from self._intervals

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Interval:
        return self._intervals[idx]

    def resample(self, seed: int | None = None) -> None:
        self._update_seed(seed)

        self._intervals = []
        self._generate_intervals()

    def _chrom_sizes_to_regions(self) -> dict[str, InterLap]:
        regions = {}
        for chrom, size in self.chrom_sizes.items():
            tree = InterLap()
            tree.add((0, size))
            regions[chrom] = tree
        return regions

    def _generate_intervals(self) -> None:
        regions_to_use = self.regions

        # If no regions specified, use full chromosomes
        if regions_to_use is None:
            regions_to_use = self._chrom_sizes_to_regions()

        # Flatten regions into list of valid intervals (chrom, start, end)
        flat_regions = []
        weights = []

        for chrom, tree in regions_to_use.items():
            if chrom not in self.chrom_sizes:
                continue
            for interval in tree:
                start, end = interval
                length = end - start
                if length >= self.padded_size:
                    flat_regions.append((chrom, start, end))
                    weights.append(length)

        if not flat_regions:
            if self.num_intervals > 0:
                logger.warning("No allowed regions large enough for sampling.")
            return

        count = 0
        max_attempts = self.num_intervals * self.MAX_ATTEMPT_MULTIPLIER
        attempts = 0

        while count < self.num_intervals and attempts < max_attempts:
            attempts += 1
            # Pick a region weighted by size
            region_idx = self.rng.choices(
                range(len(flat_regions)), weights=weights, k=1
            )[0]
            chrom, r_start, r_end = flat_regions[region_idx]

            # Sample within region
            max_start = r_end - self.padded_size

            # Since we filtered flat_regions by size, max_start >= r_start should hold.
            start = self.rng.randint(r_start, max_start)
            end = start + self.padded_size

            if not self.is_excluded(chrom, start, end):
                self._intervals.append(Interval(chrom, start, end, "+"))
                count += 1

        if count < self.num_intervals:
            logger.warning(
                f"RandomSampler could only generate {count}/{self.num_intervals} "
                f"intervals after {attempts} attempts."
            )

    def _subset(self, indices: list[int]) -> "ListSampler":
        # Kept for compatibility if needed, but split_folds won't use it
        return ListSampler(
            intervals=[self._intervals[i] for i in indices],
            chrom_sizes=self.chrom_sizes,
            folds=self.folds,
            exclude_intervals=self.exclude_intervals,
        )

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["RandomSampler", "RandomSampler", "RandomSampler"]:
        if not self.folds:
            raise ValueError("Cannot split folds without fold definitions.")

        train_intervals, val_intervals, test_intervals = partition_intervals_by_fold(
            self._intervals, self.folds, test_fold, val_fold
        )

        train_count = len(train_intervals)
        val_count = len(val_intervals)
        test_count = len(test_intervals)

        # Define regions for each split
        test_regions = self.folds[test_fold] if test_fold is not None else {}
        val_regions = self.folds[val_fold] if val_fold is not None else {}

        train_regions = defaultdict(InterLap)
        for i, fold in enumerate(self.folds):
            if i == test_fold or i == val_fold:
                continue
            for chrom, tree in fold.items():
                if chrom not in train_regions:
                    train_regions[chrom] = InterLap()
                for interval in tree:
                    train_regions[chrom].add(interval)

        # Generate a master seed for the split.
        # Using self.seed ensures idempotency.
        # Since self.seed changes on resample, splits will also change when parent advances.
        s1, s2, s3 = generate_sub_seeds(self.seed, 3)

        def make_sampler(count, regions, seed):
            return RandomSampler(
                chrom_sizes=self.chrom_sizes,
                padded_size=self.padded_size,
                num_intervals=count,
                folds=self.folds,
                exclude_intervals=self.exclude_intervals,
                regions=regions,
                seed=seed,
            )

        return (
            make_sampler(train_count, train_regions, s1),
            make_sampler(val_count, val_regions, s2),
            make_sampler(test_count, test_regions, s3),
        )


class IntervalSampler(ListSampler):
    """
    Samples from a list of genomic intervals provided in a file.

    Supports BED and narrowPeak formats.
    """

    def __init__(
        self,
        file_path: Path,
        chrom_sizes: dict[str, int],
        padded_size: int,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
    ):
        """
        Args:
            file_path: Path to the interval file (BED/narrowPeak).
            chrom_sizes: Dictionary of chromosome sizes.
            padded_size: Desired length of the intervals.
            folds: Fold definitions for cross-validation.
            exclude_intervals: Dictionary of regions to exclude.
        """
        super().__init__(
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        self.file_path = Path(file_path)
        self.padded_size = padded_size
        self._load()
        self._filter_excludes()

    def _load(self) -> None:
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
            with open(self.file_path) as f:
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

    def _load_bed(self) -> None:
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

    def _load_narrowPeak(self) -> None:
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

    def _filter_excludes(self) -> None:
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


class SlidingWindowSampler(ListSampler):
    """
    Generates samples by sliding a window across the genome.

    Useful for genome-wide prediction tasks.
    """

    def __init__(
        self,
        chrom_sizes: dict[str, int],
        padded_size: int,
        stride: int,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
    ):
        """
        Args:
            chrom_sizes: Dictionary of chromosome sizes.
            padded_size: Size of the window.
            stride: Step size.
            folds: Fold definitions.
            exclude_intervals: Dictionary of regions to exclude.
        """
        super().__init__(
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        self.padded_size = padded_size
        self.stride = stride
        self._generate_intervals()

    def _generate_intervals(self) -> None:
        # Sort chromosomes to ensure deterministic order
        for chrom in sorted(self.chrom_sizes.keys()):
            size = self.chrom_sizes[chrom]
            for start in range(0, size - self.padded_size + 1, self.stride):
                end = start + self.padded_size
                if not self.is_excluded(chrom, start, end):
                    self._intervals.append(Interval(chrom, start, end, "+"))


#### Concrete ProxySamplers ####


class ScaledSampler(ProxySampler):
    """
    Wraps a sampler to resize it (subsample or oversample).
    """

    def __init__(
        self,
        sampler: Sampler,
        num_samples: int,
        seed: int = 42,
        generate_on_init: bool = True,
    ):
        """
        Args:
            sampler: The underlying sampler to wrap.
            num_samples: The desired number of samples per epoch.
            seed: Random seed.
            generate_on_init: Whether to generate samples immediately (default: True).
        """
        super().__init__(
            chrom_sizes=getattr(sampler, "chrom_sizes", {}),
            exclude_intervals=getattr(sampler, "exclude_intervals", {}),
            folds=getattr(sampler, "folds", []),
        )
        self.sampler = sampler
        self.num_samples = int(num_samples)
        self._indices: list[int] = []
        self.seed: int = seed
        self.rng = random.Random(seed)
        if generate_on_init:
            self.resample(seed=seed)

    def resample(self, seed: int | None = None) -> None:
        self._update_seed(seed)

        # Propagate derived seed to child to decouple RNG streams
        child_seed = self.rng.getrandbits(32)
        self.sampler.resample(child_seed)

        n_total = len(self.sampler)

        if n_total == 0:
            self._indices = []
            return

        if self.num_samples > n_total:
            # Oversample with replacement
            self._indices = self.rng.choices(range(n_total), k=self.num_samples)
        else:
            # Subsample without replacement (if possible)
            self._indices = self.rng.sample(range(n_total), k=self.num_samples)

    def __iter__(self) -> Iterator[Interval]:
        for idx in self._indices:
            yield self.sampler[idx]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Interval:
        real_idx = self._indices[idx]
        return self.sampler[real_idx]

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["ScaledSampler", "ScaledSampler", "ScaledSampler"]:
        train, val, test = self.sampler.split_folds(test_fold, val_fold)

        total_len = len(self.sampler)
        if total_len == 0:
            ratio = 1.0
        else:
            ratio = self.num_samples / total_len

        train_size = int(len(train) * ratio)
        val_size = int(len(val) * ratio)
        test_size = int(len(test) * ratio)

        s1, s2, s3 = generate_sub_seeds(self.seed, 3)

        return (
            ScaledSampler(train, train_size, s1),
            ScaledSampler(val, val_size, s2),
            ScaledSampler(test, test_size, s3),
        )


class ComplexityMatchedSampler(ProxySampler):
    """
    Selects candidates from a candidate_sampler that match the distributional properties
    of a target_sampler.

    Matches on a specified list of complexity metrics: 'gc', 'dust', 'cpg'.

    This sampler bins the target intervals by their metrics and then selects
    intervals from the candidate sampler to match the count in each bin, scaled
    by `candidate_ratio`.

    IMPORTANT: Any sampler type that uses ComplexityMatchedSampler must be
    registered in CerberusDataModule._resolve_cache_dir() and
    CerberusDataModule.prepare_data() so that its metrics_cache is persisted
    to disk across runs. Without this, metrics are recomputed from scratch
    every time (expensive FASTA reads).
    """

    def __init__(
        self,
        target_sampler: Sampler,
        candidate_sampler: Sampler,
        fasta_path: Path | str,
        chrom_sizes: dict[str, int],
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        bins: int = 20,
        candidate_ratio: float = 1.0,
        metrics: list[str] | None = None,
        seed: int = 42,
        generate_on_init: bool = True,
        metrics_cache: dict[str, np.ndarray] | None = None,
        center_size: int | None = None,
    ):
        """
        Args:
            target_sampler: Sampler providing target distribution.
            candidate_sampler: Sampler providing candidate pool.
            fasta_path: Path to genome FASTA.
            chrom_sizes: Chromosome sizes.
            folds: Fold definitions.
            exclude_intervals: Excluded regions.
            bins: Number of bins for complexity matching.
            candidate_ratio: Ratio of candidates to targets.
            metrics: List of complexity metrics to match.
            seed: Random seed.
            generate_on_init: Whether to generate samples immediately.
            metrics_cache: Shared cache for interval metrics.
            center_size: If set, crop intervals to center N bp before computing
                         complexity metrics. Decouples model input size from
                         complexity matching scale.
        """
        super().__init__(
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        self.target_sampler = target_sampler
        self.candidate_sampler = candidate_sampler
        self.fasta_path = Path(fasta_path)
        self.bins = bins
        self.candidate_ratio = candidate_ratio
        self.seed: int = seed
        self.rng = random.Random(seed)

        if metrics is None:
            metrics = ["gc", "dust", "cpg"]
        self.metrics = metrics
        self.center_size = center_size
        self.metrics_cache = metrics_cache if metrics_cache is not None else {}

        # Lazy initialization
        self.target_metrics: np.ndarray = np.empty(
            (0, len(self.metrics)), dtype=np.float32
        )
        self.target_hist: dict[tuple[int, ...], int] = {}
        self.candidate_bins: dict[tuple[int, ...], list[int]] = {}
        self._initialized = False

        self._indices: list[int] = []  # Indices into candidate_sampler
        if generate_on_init:
            self.resample()

    def _cache_key(self, iv: Interval) -> str:
        """Build a cache key that includes center_size when set."""
        if self.center_size is not None:
            return f"{iv}:cs={self.center_size}"
        return str(iv)

    def _get_metrics(self, sampler: Sampler, name: str) -> np.ndarray:
        intervals = list(sampler)
        missing = []

        for iv in intervals:
            key = self._cache_key(iv)
            if key not in self.metrics_cache:
                missing.append(iv)

        if missing:
            logger.info(
                f"ComplexityMatchedSampler: Computing {len(missing)} missing metrics for {name} intervals..."
            )
            new_metrics = compute_intervals_complexity(
                missing, self.fasta_path, self.metrics, center_size=self.center_size
            )
            for iv, row in zip(missing, new_metrics, strict=True):
                self.metrics_cache[self._cache_key(iv)] = row

        if not intervals:
            return np.empty((0, len(self.metrics)), dtype=np.float32)

        return np.array(
            [self.metrics_cache[self._cache_key(iv)] for iv in intervals],
            dtype=np.float32,
        )

    def _initialize(self) -> None:
        logger.info("ComplexityMatchedSampler: Initializing complexity matching...")

        # Compute target metrics
        logger.info(
            f"ComplexityMatchedSampler: Retrieving complexity metrics for {len(self.target_sampler)} target intervals..."
        )
        self.target_metrics = self._get_metrics(self.target_sampler, "target")
        self.target_hist = compute_hist(self.target_metrics, self.bins)
        logger.info("ComplexityMatchedSampler: Target metrics ready and binned.")

        # Generate candidate metrics
        # Provide a new seed based on current RNG state
        cand_seed = self.rng.getrandbits(32)
        self.candidate_sampler.resample(cand_seed)

        logger.info(
            f"ComplexityMatchedSampler: Retrieving complexity metrics for {len(self.candidate_sampler)} candidate intervals..."
        )
        self.candidate_metrics = self._get_metrics(self.candidate_sampler, "candidate")
        logger.info("ComplexityMatchedSampler: Candidate metrics ready.")

        # Bin candidates
        logger.info("ComplexityMatchedSampler: Binning candidate intervals...")
        self.candidate_bins = defaultdict(list)

        for i, row in enumerate(self.candidate_metrics):
            interval = self.candidate_sampler[i]
            if self.is_excluded(interval.chrom, interval.start, interval.end):
                continue

            bin_idx = get_bin_index(row, self.bins)
            if bin_idx is not None:
                self.candidate_bins[bin_idx].append(i)

        self._initialized = True
        logger.info("ComplexityMatchedSampler: Initialization complete.")

    def resample(self, seed: int | None = None) -> None:
        self._update_seed(seed)

        # Ensure pool is initialized (lazy initialization if generate_on_init was False)
        if not self._initialized:
            self._initialize()

        # NOTE: We do NOT resample candidate_sampler here.
        # We reuse the pool generated in _initialize.
        self._indices = _select_from_bins(
            self.target_hist, self.candidate_bins, self.candidate_ratio, self.rng
        )

        # Ensure exact count by filling gaps if necessary
        # (e.g. if some bins had no candidates), also the
        # also: int(count * self.candidate_ratio) rounds down so we may be short.
        target_count = len(self.target_sampler)
        expected_count = int(target_count * self.candidate_ratio)
        current_count = len(self._indices)

        if current_count < expected_count:
            missing = expected_count - current_count
            n_candidates = len(self.candidate_sampler)
            if n_candidates > 0:
                # Fallback: sample randomly from all candidates
                fallback_indices = self.rng.choices(range(n_candidates), k=missing)
                self._indices.extend(fallback_indices)

        # We must shuffle because the loop over target_hist produces grouped indices
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
    ) -> tuple[
        "ComplexityMatchedSampler",
        "ComplexityMatchedSampler",
        "ComplexityMatchedSampler",
    ]:
        target_splits = self.target_sampler.split_folds(test_fold, val_fold)

        # If initialized, partition the existing candidate pool to preserve cache hits.
        # If we rely on RandomSampler.split_folds(), it regenerates new random intervals (cache miss).
        if self._initialized:
            # We iterate the candidate sampler and bucket intervals into train/val/test
            train_intervals, val_intervals, test_intervals = (
                partition_intervals_by_fold(
                    self.candidate_sampler, self.folds, test_fold, val_fold
                )
            )

            candidate_splits = (
                ListSampler(
                    train_intervals,
                    self.chrom_sizes,
                    self.folds,
                    self.exclude_intervals,
                ),
                ListSampler(
                    val_intervals, self.chrom_sizes, self.folds, self.exclude_intervals
                ),
                ListSampler(
                    test_intervals, self.chrom_sizes, self.folds, self.exclude_intervals
                ),
            )
        else:
            # Fallback if not initialized (though cache will miss on first run)
            candidate_splits = self.candidate_sampler.split_folds(test_fold, val_fold)

        s1, s2, s3 = generate_sub_seeds(self.seed, 3)

        return (
            ComplexityMatchedSampler(
                target_splits[0],
                candidate_splits[0],
                self.fasta_path,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                self.bins,
                self.candidate_ratio,
                self.metrics,
                s1,
                generate_on_init=True,
                metrics_cache=self.metrics_cache,
            ),
            ComplexityMatchedSampler(
                target_splits[1],
                candidate_splits[1],
                self.fasta_path,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                self.bins,
                self.candidate_ratio,
                self.metrics,
                s2,
                generate_on_init=True,
                metrics_cache=self.metrics_cache,
            ),
            ComplexityMatchedSampler(
                target_splits[2],
                candidate_splits[2],
                self.fasta_path,
                self.chrom_sizes,
                self.folds,
                self.exclude_intervals,
                self.bins,
                self.candidate_ratio,
                self.metrics,
                s3,
                generate_on_init=True,
                metrics_cache=self.metrics_cache,
            ),
        )


#### Concrete MultiSamplers ####


class PeakSampler(MultiSampler):
    """
    A specialized MultiSampler that combines a set of positive intervals (peaks)
    with a complexity-matched negative set.

    This class simplifies the creation of a balanced training set by:
    1. Loading the peaks once.
    2. Automatically excluding peaks from the background candidates.
    3. Generating a complexity-matched negative set with a specified ratio.
    4. Mixing the positives and negatives with given ratio.
    """

    def __init__(
        self,
        intervals_path: Path | str,
        fasta_path: Path | str,
        chrom_sizes: dict[str, int],
        padded_size: int,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        background_ratio: float = 1.0,
        min_candidates: int = 10000,
        candidate_oversample_factor: float = 5.0,
        seed: int = 42,
        prepare_cache: dict[str, np.ndarray] | None = None,
        complexity_center_size: int | None = None,
    ):
        """
        Args:
            intervals_path: Path to peaks.
            fasta_path: Path to genome FASTA.
            chrom_sizes: Chromosome sizes.
            padded_size: Interval size.
            folds: Fold definitions.
            exclude_intervals: Excluded regions.
            background_ratio: Ratio of background intervals to peaks.
                              e.g. 1.0 = 1:1, 2.0 = 2 backgrounds per peak.
            min_candidates: Minimum number of candidate intervals to generate.
            candidate_oversample_factor: Factor to oversample candidates relative to peaks.
            seed: Random seed.
            prepare_cache: Pre-computed data from prepare_data() (e.g. complexity metrics cache).
            complexity_center_size: If set, crop intervals to center N bp before
                                   computing complexity metrics for matching.
        """
        self.intervals_path = Path(intervals_path)
        self.background_ratio = background_ratio

        logger.info(
            f"PeakSampler: Loading positive intervals from {self.intervals_path}..."
        )
        # 1. Positives (Peaks)
        self.positives = IntervalSampler(
            file_path=self.intervals_path,
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        logger.info(f"PeakSampler: Loaded {len(self.positives)} positive intervals.")

        samplers: list[Sampler] = [self.positives]

        if background_ratio > 0:
            logger.info(
                f"PeakSampler: Generating background candidates (ratio={background_ratio})..."
            )

            if exclude_intervals is None:
                exclude_intervals = {}

            # 2. Exclusions for Negatives (Original Excludes + Peaks)
            # Deep copy the exclusions to avoid modifying the global state
            neg_excludes = copy.deepcopy(exclude_intervals)
            for interval in self.positives:
                if interval.chrom not in neg_excludes:
                    # initialize a new empty interval-tree for a chrom if needed
                    neg_excludes[interval.chrom] = InterLap()
                neg_excludes[interval.chrom].add((interval.start, interval.end))

            # 3. Candidates (Random background, excluding peaks)
            # Auto-calculate candidate pool size.
            # We need enough candidates to find matches. 10x is usually safe.
            # But ensure a minimum floor (e.g. 10,000) if peaks are few.
            n_peaks = len(self.positives)
            n_candidates = max(
                min_candidates,
                int(n_peaks * background_ratio * candidate_oversample_factor),
            )
            self.candidates = RandomSampler(
                chrom_sizes=chrom_sizes,
                padded_size=padded_size,
                num_intervals=n_candidates,
                folds=folds,
                exclude_intervals=neg_excludes,
                seed=seed,
                generate_on_init=False,
            )

            # 4. Negatives (Complexity Matched to Positives)
            self.negatives = ComplexityMatchedSampler(
                target_sampler=self.positives,
                candidate_sampler=self.candidates,
                fasta_path=fasta_path,
                chrom_sizes=chrom_sizes,
                folds=folds,
                exclude_intervals=neg_excludes,  # Use the augmented excludes
                candidate_ratio=background_ratio,
                seed=seed,
                generate_on_init=False,
                metrics=["gc", "dust", "cpg"],
                metrics_cache=prepare_cache,
                center_size=complexity_center_size,
            )
            samplers.append(self.negatives)
        else:
            self.candidates = None
            self.negatives = None

        # 5. Initialize MultiSampler
        super().__init__(
            samplers=samplers,
            chrom_sizes=chrom_sizes,
            exclude_intervals=exclude_intervals,  # Base exclusions for validity
            seed=seed,
        )


class NegativePeakSampler(MultiSampler):
    """Background-only sampler: complexity-matched non-peak intervals.

    Like PeakSampler, loads peaks and generates complexity-matched background
    intervals that exclude peak regions. Unlike PeakSampler, the peaks
    themselves are NOT included in the training set — only the background
    intervals are used. All intervals have peak_status=0.

    Used for training bias-only models (e.g., Tn5 sequence bias in ChromBPNet /
    Dalmatian) on non-peak regions where enzymatic bias dominates.

    Args:
        intervals_path: Path to peaks (used for exclusion and complexity reference).
        fasta_path: Path to genome FASTA.
        chrom_sizes: Chromosome sizes.
        padded_size: Interval size.
        folds: Fold definitions.
        exclude_intervals: Excluded regions.
        background_ratio: Number of background intervals per peak.
            Controls epoch size: num_intervals = num_peaks * background_ratio.
        min_candidates: Minimum number of candidate intervals to generate.
        candidate_oversample_factor: Factor to oversample candidates relative to peaks.
        seed: Random seed.
        prepare_cache: Pre-computed data from prepare_data() (e.g. complexity metrics cache).
    """

    def __init__(
        self,
        intervals_path: Path | str,
        fasta_path: Path | str,
        chrom_sizes: dict[str, int],
        padded_size: int,
        folds: list[dict[str, InterLap]] | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        background_ratio: float = 1.0,
        min_candidates: int = 10000,
        candidate_oversample_factor: float = 5.0,
        seed: int = 42,
        prepare_cache: dict[str, np.ndarray] | None = None,
        complexity_center_size: int | None = None,
    ):
        self.intervals_path = Path(intervals_path)
        self.background_ratio = background_ratio

        logger.info(
            f"NegativePeakSampler: Loading peak intervals from {self.intervals_path} (for exclusion/complexity reference)..."
        )
        # 1. Load peaks (for exclusion and complexity reference only — not included in training)
        self.positives = IntervalSampler(
            file_path=self.intervals_path,
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )
        logger.info(
            f"NegativePeakSampler: Loaded {len(self.positives)} peak intervals (reference only)."
        )

        if exclude_intervals is None:
            exclude_intervals = {}

        # 2. Exclusions for negatives (original excludes + peaks)
        neg_excludes = copy.deepcopy(exclude_intervals)
        for interval in self.positives:
            if interval.chrom not in neg_excludes:
                neg_excludes[interval.chrom] = InterLap()
            neg_excludes[interval.chrom].add((interval.start, interval.end))

        # 3. Candidates (random background, excluding peaks)
        n_peaks = len(self.positives)
        n_candidates = max(
            min_candidates,
            int(n_peaks * background_ratio * candidate_oversample_factor),
        )
        self.candidates = RandomSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            num_intervals=n_candidates,
            folds=folds,
            exclude_intervals=neg_excludes,
            seed=seed,
            generate_on_init=False,
        )

        # 4. Negatives (complexity matched to peaks)
        self.negatives = ComplexityMatchedSampler(
            target_sampler=self.positives,
            candidate_sampler=self.candidates,
            fasta_path=fasta_path,
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=neg_excludes,
            candidate_ratio=background_ratio,
            seed=seed,
            generate_on_init=False,
            metrics=["gc", "dust", "cpg"],
            metrics_cache=prepare_cache,
            center_size=complexity_center_size,
        )

        # 5. Initialize MultiSampler with only negatives
        super().__init__(
            samplers=[self.negatives],
            chrom_sizes=chrom_sizes,
            exclude_intervals=exclude_intervals,
            seed=seed,
        )


def create_sampler(
    config: SamplerConfig,
    chrom_sizes: dict[str, int],
    folds: list[dict[str, InterLap]],
    exclude_intervals: dict[str, InterLap],
    fasta_path: Path | str | None = None,
    seed: int = 42,
    prepare_cache: dict[str, np.ndarray] | None = None,
) -> Sampler:
    """
    Factory function to create a Sampler from a config.

    Args:
        config: Sampler configuration dictionary.
        chrom_sizes: Chromosome sizes dictionary.
        folds: List of fold definitions.
        exclude_intervals: Dictionary of excluded regions.
        fasta_path: Path to the genome FASTA file (required for ComplexityMatchedSampler/PeakSampler).
        seed: Random seed.
        prepare_cache: Pre-computed data from prepare_data() (e.g. complexity metrics cache).

    Returns:
        Sampler: An instantiated sampler object.

    Raises:
        ValueError: If sampler_type is unsupported.
    """
    sampler_type = config.sampler_type
    sampler_args = config.sampler_args
    padded_size = config.padded_size

    if sampler_type == "interval":
        return IntervalSampler(
            file_path=sampler_args["intervals_path"],
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )

    elif sampler_type == "sliding_window":
        return SlidingWindowSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            stride=sampler_args["stride"],
            folds=folds,
            exclude_intervals=exclude_intervals,
        )

    elif sampler_type == "random":
        return RandomSampler(
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            num_intervals=sampler_args["num_intervals"],
            folds=folds,
            exclude_intervals=exclude_intervals,
            seed=seed,
        )

    elif sampler_type == "complexity_matched":
        if fasta_path is None:
            raise ValueError(
                "ComplexityMatchedSampler requires 'fasta_path' to be provided."
            )

        target_conf = sampler_args["target_sampler"]
        candidate_conf = sampler_args["candidate_sampler"]

        # Recursive creation: sub-samplers are SamplerConfig models
        target_sampler = create_sampler(
            target_conf,
            chrom_sizes,
            folds,
            exclude_intervals,
            fasta_path,
            seed=generate_sub_seeds(seed, 2)[0],
        )
        candidate_sampler = create_sampler(
            candidate_conf,
            chrom_sizes,
            folds,
            exclude_intervals,
            fasta_path,
            seed=generate_sub_seeds(seed, 2)[1],
        )

        return ComplexityMatchedSampler(
            target_sampler=target_sampler,
            candidate_sampler=candidate_sampler,
            fasta_path=fasta_path,
            chrom_sizes=chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
            bins=sampler_args["bins"],
            candidate_ratio=sampler_args["candidate_ratio"],
            seed=seed,
            metrics=sampler_args["metrics"],
            metrics_cache=prepare_cache,
        )

    elif sampler_type == "peak":
        if fasta_path is None:
            raise ValueError("PeakSampler requires 'fasta_path' to be provided.")

        return PeakSampler(
            intervals_path=sampler_args["intervals_path"],
            fasta_path=fasta_path,
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
            background_ratio=sampler_args["background_ratio"],
            seed=seed,
            prepare_cache=prepare_cache,
            complexity_center_size=sampler_args.get("complexity_center_size"),
        )

    elif sampler_type == "negative_peak":
        if fasta_path is None:
            raise ValueError(
                "NegativePeakSampler requires 'fasta_path' to be provided."
            )

        return NegativePeakSampler(
            intervals_path=sampler_args["intervals_path"],
            fasta_path=fasta_path,
            chrom_sizes=chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
            background_ratio=sampler_args["background_ratio"],
            seed=seed,
            prepare_cache=prepare_cache,
            complexity_center_size=sampler_args.get("complexity_center_size"),
        )

    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")
