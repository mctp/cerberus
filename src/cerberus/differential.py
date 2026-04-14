"""Differential chromatin accessibility label generation and loading.

Provides utilities for computing log2 CPM fold change targets from raw ChIP-seq
or ATAC-seq counts, and for loading/writing the resulting peak-level differential
targets used by :class:`~cerberus.loss.DifferentialCountLoss`.

Typical usage
-------------
::

    import numpy as np
    from cerberus.differential import (
        compute_log2fc_cpm,
        DifferentialRecord,
        write_differential_targets,
        DifferentialTargetIndex,
    )

    # 1. Compute log2FC from raw peak counts (one value per peak)
    log2fc = compute_log2fc_cpm(counts_a, counts_b, pseudocount=1.0)

    # 2. Pack into records and write TSV
    records = [
        DifferentialRecord(chrom=c, start=s, end=e, log2fc=float(fc))
        for c, s, e, fc in zip(chroms, starts, ends, log2fc)
    ]
    write_differential_targets("targets.tsv", records)

    # 3. Build index and load into training
    index = DifferentialTargetIndex.from_tsv("targets.tsv")
    # Pass log2fc via the DifferentialCountLoss log2fc kwarg in your training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybigtools

from cerberus.interval import Interval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log2 CPM fold change computation
# ---------------------------------------------------------------------------


def compute_log2fc_cpm(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    pseudocount: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Compute log2 CPM fold change between two conditions.

    Applies an additive pseudocount before taking the log ratio, which
    acts as shrinkage: low-coverage peaks with uncertain fold changes are
    pulled toward zero, analogous to the effect of apeglm shrinkage but
    without requiring replicates or a statistical model.

    Args:
        counts_a: Raw read counts for condition A (reference), shape ``(N,)``.
            May be any numeric array type.
        counts_b: Raw read counts for condition B (query), shape ``(N,)``.
        pseudocount: Additive pseudocount applied to both CPM tracks before
            log2.  Larger values shrink low-coverage peaks more aggressively
            toward zero.  Default ``1.0`` (in CPM units when
            ``normalize=True``; in raw count units when ``normalize=False``).
        normalize: If ``True`` (default), normalize each condition to counts
            per million (CPM) before applying the pseudocount.  Set
            ``False`` if the input counts are already depth-normalised (e.g.
            RPM from a bigwig extractor).

    Returns:
        ``np.ndarray`` of shape ``(N,)`` containing the log2 fold change
        (condition B / condition A) for each peak.  Positive values indicate
        higher signal in B; negative values indicate higher signal in A;
        values near zero indicate no change.

    Examples::

        >>> import numpy as np
        >>> counts_a = np.array([0, 10, 100, 0])
        >>> counts_b = np.array([0, 20, 100, 5])
        >>> compute_log2fc_cpm(counts_a, counts_b, normalize=False)
        array([0.   ,  0.585...,  0.   ,  2.321...])
        # both-zero peak → 0, not ±inf; low-count peak (0,5) < (0,5) raw

    Raises:
        ValueError: If ``counts_a`` and ``counts_b`` have different shapes.
        ValueError: If ``pseudocount`` is not positive.
    """
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"counts_a and counts_b must have the same shape, "
            f"got {a.shape} and {b.shape}"
        )
    if pseudocount <= 0:
        raise ValueError(f"pseudocount must be positive, got {pseudocount}")

    if normalize:
        total_a = a.sum()
        total_b = b.sum()
        if total_a == 0 or total_b == 0:
            raise ValueError(
                "Total counts are zero for at least one condition. "
                "Cannot normalize to CPM. "
                "Check that the correct BAM/count files were used, "
                "or set normalize=False if counts are pre-normalised."
            )
        a = a / total_a * 1e6
        b = b / total_b * 1e6

    return np.log2((b + pseudocount) / (a + pseudocount))


# ---------------------------------------------------------------------------
# Bigwig-based count extraction
# ---------------------------------------------------------------------------


def compute_bigwig_counts(
    bigwig_path: str | Path,
    intervals: list[Interval],
) -> np.ndarray:
    """Sum bigwig bin values over each interval.

    For each interval, extracts the per-base signal from *bigwig_path* and
    sums it (NaN positions count as zero).  This is identical to how the
    count head computes its training target from the signal tensor::

        # count head (loss.py):
        target_counts = targets.sum(dim=2)   # (B, C)

    Chromosomes or regions absent from the bigwig return a sum of 0.

    Args:
        bigwig_path: Path to a bigwig file.  Typically depth-normalised
            (e.g. RPM) so that sums are comparable across samples without
            further library-size correction.
        intervals: List of :class:`~cerberus.interval.Interval` objects
            defining the peak regions.

    Returns:
        ``np.ndarray`` of shape ``(N,)`` containing the sum of bigwig values
        over each interval, in the units of the bigwig file.
    """
    bigwig_path = Path(bigwig_path)
    bw = pybigtools.open(str(bigwig_path))  # type: ignore[attr-defined]
    counts = np.zeros(len(intervals), dtype=np.float64)
    for i, interval in enumerate(intervals):
        try:
            vals = bw.values(interval.chrom, interval.start, interval.end)
            vals = np.asarray(vals, dtype=np.float64)
            counts[i] = float(np.nansum(vals))
        except RuntimeError:
            logger.debug(
                "Chrom %s not found in %s, returning 0 for %s:%d-%d",
                interval.chrom,
                bigwig_path.name,
                interval.chrom,
                interval.start,
                interval.end,
            )
            counts[i] = 0.0
    logger.info(
        "Extracted bigwig counts from %s for %d intervals",
        bigwig_path.name,
        len(intervals),
    )
    return counts


def compute_log2fc_from_bigwigs(
    bigwig_a: str | Path,
    bigwig_b: str | Path,
    intervals: list[Interval],
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Compute log2 fold change directly from two depth-normalised bigwig files.

    Replaces the BAM-based count extraction step: instead of running
    ``multiBamSummary`` / ``bedtools coverage`` and normalising to CPM, this
    function sums the already depth-normalised bigwig signal over each
    interval and computes the log2 ratio.  This is equivalent to the count
    head's own target computation:

    .. code-block:: text

        count head (per condition, per batch):
            target_counts = bigwig_signal.sum(dim=2)       # (B, C)
            target_log_counts = log(target_counts + pc)

        this function (across conditions, per peak):
            counts_a[i] = sum(bigwig_a over interval i)    # (N,)
            counts_b[i] = sum(bigwig_b over interval i)
            log2fc[i]   = log2((counts_b[i] + pc) /
                               (counts_a[i] + pc))

    Because the bigwigs are expected to be already depth-normalised (e.g.
    RPM), no additional library-size correction is applied
    (``normalize=False`` in :func:`compute_log2fc_cpm`).  The pseudocount
    is in the same units as the bigwig signal.

    Args:
        bigwig_a: Path to the depth-normalised bigwig for condition A
            (reference).
        bigwig_b: Path to the depth-normalised bigwig for condition B
            (query).
        intervals: Peak regions — must be the same set for both conditions.
        pseudocount: Additive pseudocount in bigwig signal units.  Larger
            values pull low-signal peaks toward zero (shrinkage).  Default
            ``1.0``.

    Returns:
        ``np.ndarray`` of shape ``(N,)`` with log2 fold change per interval
        (condition B / condition A).  Positive values indicate higher signal
        in B; negative in A; near-zero means no change.

    Example::

        from cerberus.interval import Interval
        from cerberus.differential import (
            compute_log2fc_from_bigwigs,
            DifferentialRecord,
            write_differential_targets,
        )

        intervals = [Interval("chr1", 1000, 2000), Interval("chr2", 500, 1500)]
        log2fc = compute_log2fc_from_bigwigs(
            "condition_a.rpm.bw",
            "condition_b.rpm.bw",
            intervals,
            pseudocount=1.0,
        )

        records = [
            DifferentialRecord(chrom=iv.chrom, start=iv.start, end=iv.end,
                               log2fc=float(fc))
            for iv, fc in zip(intervals, log2fc)
        ]
        write_differential_targets("targets.tsv", records)
    """
    counts_a = compute_bigwig_counts(bigwig_a, intervals)
    counts_b = compute_bigwig_counts(bigwig_b, intervals)
    return compute_log2fc_cpm(counts_a, counts_b, pseudocount=pseudocount, normalize=False)


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifferentialRecord:
    """Single-peak differential accessibility label.

    Coordinates follow the BED convention: 0-based, half-open ``[start, end)``,
    matching :class:`~cerberus.interval.Interval`.

    Attributes:
        chrom: Chromosome name (e.g. ``'chr1'``).
        start: Start position (0-based inclusive).
        end: End position (0-based exclusive).
        log2fc: Log2 fold change (condition B / condition A).  Typically
            produced by :func:`compute_log2fc_cpm`.
        base_mean: Optional mean normalised coverage across both conditions.
            Not used by the training pipeline; retained for downstream
            filtering or QC.
    """

    chrom: str
    start: int
    end: int
    log2fc: float
    base_mean: float | None = None

    @property
    def interval(self) -> Interval:
        """Return the interval corresponding to this peak.

        Returns an :class:`~cerberus.interval.Interval` with the same
        coordinates.
        """
        return Interval(self.chrom, self.start, self.end)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_differential_targets(
    path: str | Path,
    log2fc_col: str = "log2fc",
    base_mean_col: str = "base_mean",
) -> list[DifferentialRecord]:
    """Load differential targets from a tab-separated file.

    The file must have a tab-separated header row.  Required columns:
    ``chrom``, ``start`` (0-based), ``end``, and the log2FC column
    (default: ``"log2fc"``).  An optional ``base_mean`` column is retained
    for reference.

    The file produced by :func:`write_differential_targets` is accepted
    directly.

    Args:
        path: Path to the TSV file.
        log2fc_col: Column name for log2FC values. Default ``"log2fc"``.
        base_mean_col: Column name for optional mean coverage.
            Default ``"base_mean"``.

    Returns:
        List of :class:`DifferentialRecord` in file order.

    Raises:
        ValueError: If required columns are absent from the header.
    """
    path = Path(path)
    records: list[DifferentialRecord] = []

    with open(path) as fh:
        raw_header = next(fh).rstrip("\n")
        if raw_header.startswith("#"):
            raw_header = raw_header[1:]
        cols = {
            name.strip().lower(): i
            for i, name in enumerate(raw_header.split("\t"))
        }

        required = {"chrom", "start", "end", log2fc_col.lower()}
        missing = required - cols.keys()
        if missing:
            raise ValueError(
                f"Missing required column(s) in {path}: "
                f"{', '.join(sorted(missing))}. "
                f"Available columns: {list(cols)}"
            )

        ic = cols["chrom"]
        is_ = cols["start"]
        ie = cols["end"]
        ilfc = cols[log2fc_col.lower()]
        ibm = cols.get(base_mean_col.lower())

        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")

            base_mean_val: float | None = None
            if ibm is not None and ibm < len(parts) and parts[ibm] not in ("NA", ""):
                base_mean_val = float(parts[ibm])

            records.append(
                DifferentialRecord(
                    chrom=parts[ic],
                    start=int(parts[is_]),
                    end=int(parts[ie]),
                    log2fc=float(parts[ilfc]),
                    base_mean=base_mean_val,
                )
            )

    logger.info("Loaded %d differential records from %s", len(records), path)
    return records


def write_differential_targets(
    path: str | Path,
    records: list[DifferentialRecord],
) -> None:
    """Write a list of :class:`DifferentialRecord` to a TSV file.

    The output is accepted directly by :func:`load_differential_targets`.

    Args:
        path: Destination file path.
        records: Records to write.
    """
    path = Path(path)
    with open(path, "w") as fh:
        fh.write("chrom\tstart\tend\tlog2fc\tbase_mean\n")
        for r in records:
            bm = f"{r.base_mean:.4f}" if r.base_mean is not None else "NA"
            fh.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.log2fc:.6g}\t{bm}\n")
    logger.info("Wrote %d differential records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class DifferentialTargetIndex:
    """Fast ``(chrom, start, end)`` → log2FC lookup for batch assembly.

    Keyed by exact genomic coordinates.  The intervals stored in the index
    must match the sampler intervals in :class:`~cerberus.dataset.CerberusDataset`
    exactly (same chromosome, same 0-based BED start/end).

    Args:
        records: List of :class:`DifferentialRecord` objects to index.
        default: Value returned for intervals not found in the index.
            Defaults to ``0.0`` — treats unknown peaks as non-differential.

    Example::

        index = DifferentialTargetIndex(records, default=0.0)
        from cerberus.interval import Interval
        log2fc = index.get(Interval("chr1", 1000, 2000))
    """

    def __init__(
        self,
        records: list[DifferentialRecord],
        default: float = 0.0,
    ) -> None:
        self._index: dict[tuple[str, int, int], float] = {
            (r.chrom, r.start, r.end): r.log2fc for r in records
        }
        self.default = default
        logger.info(
            "Built DifferentialTargetIndex with %d entries (default=%.2f)",
            len(self._index),
            default,
        )

    @classmethod
    def from_tsv(
        cls,
        path: str | Path,
        default: float = 0.0,
        **kwargs: object,
    ) -> "DifferentialTargetIndex":
        """Construct directly from a TSV file.

        Args:
            path: Path to the targets TSV (see :func:`load_differential_targets`).
            default: Fallback value for intervals absent from the index.
            **kwargs: Forwarded to :func:`load_differential_targets`.

        Returns:
            A populated :class:`DifferentialTargetIndex`.
        """
        records = load_differential_targets(path, **kwargs)  # type: ignore[arg-type]
        return cls(records, default=default)

    def get(self, interval: object) -> float:
        """Return the log2FC for *interval*, or ``self.default`` if not found.

        Args:
            interval: A :class:`~cerberus.interval.Interval` to look up.

        Returns:
            Shrunken log2FC float, or ``self.default``.
        """
        if not isinstance(interval, Interval):
            raise TypeError(f"Expected Interval, got {type(interval).__name__}")
        return self._index.get(
            (interval.chrom, interval.start, interval.end), self.default
        )

    def __len__(self) -> int:
        """Number of intervals in the index."""
        return len(self._index)

    def __contains__(self, interval: object) -> bool:
        """Return True if *interval* is in the index."""
        if not isinstance(interval, Interval):
            return False
        return (interval.chrom, interval.start, interval.end) in self._index
