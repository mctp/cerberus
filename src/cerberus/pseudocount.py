"""Scale-aware pseudocount helpers for count-head training.

Two helpers, one per phase, reflecting the distinct role the pseudocount plays
in each training objective. See ``docs/internal/count_log_spaces.md`` and
``docs/internal/asap_pseudocount_considerations.md`` for background.

Phase 1 / single-task absolute models
-------------------------------------
``resolve_reads_equivalent_pseudocount`` converts a reads-equivalent
specification into a pseudocount in the same scaled units as the training
targets. Here the pseudocount's job is to prevent ``log(0)`` for silent regions
and to embed the zero-reads cluster into the rest of the count distribution — a
value on the order of "one read's contribution" is usually right, and the
correct value depends on read length, bin size, and (for CPM bigWigs) library
depth.

Phase 2 differential models
---------------------------
``resolve_quantile_pseudocount`` picks the pseudocount from the actual
distribution of training-region total counts (a low quantile, typically 0.10).
Here the pseudocount is an empirical-Bayes shrinkage prior on the log-fold
change: it pulls ``log((c_b + pc)/(c_a + pc))`` toward zero whenever both
totals are comparable to the noise floor, so weak peaks do not dominate the
MSE. This is data-driven and automatically adapts to raw vs. CPM inputs and to
library-depth differences without the user reasoning about read length or bin
size.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import numpy as np

logger = logging.getLogger(__name__)

InputScale = Literal["raw", "cpm"]


class _SupportsCountQuantileSamples(Protocol):
    """Structural type: anything exposing :meth:`compute_count_quantile_samples`.

    :class:`cerberus.datamodule.CerberusDataModule` satisfies this naturally;
    test stubs and alternative data sources can too without inheriting.
    """

    def compute_count_quantile_samples(
        self, n_samples: int = ..., per_channel: bool = ...
    ) -> np.ndarray: ...


def resolve_reads_equivalent_pseudocount(
    reads_equiv: float,
    read_length: int,
    bin_size: int,
    target_scale: float,
    input_scale: InputScale = "raw",
    total_reads: float | None = None,
) -> float:
    """Compute a scaled pseudocount from a reads-equivalent specification.

    The length-summed target for one region is approximately::

        counts = n_reads * (read_length / bin_size) * normalization_factor * target_scale

    where ``normalization_factor`` is ``1`` for raw-coverage bigWigs and
    ``1e6 / total_reads`` for CPM-normalized bigWigs. This helper returns the
    scaled value corresponding to ``reads_equiv`` reads' worth of coverage, so
    that passing the result as ``ModelConfig.count_pseudocount`` is equivalent
    to saying "shrink log-counts near the noise floor where a region has
    roughly ``reads_equiv`` reads or fewer".

    Args:
        reads_equiv: Number of reads the pseudocount should correspond to.
            ``1.0`` is the typical Phase 1 default (just avoid ``log(0)`` /
            keep the zero cluster embedded in the distribution).
        read_length: Sequencing read or fragment length in bp. ChIP-seq reads
            are ~150 bp, ATAC-seq fragments ~100 bp.
        bin_size: Output bin size, matches ``DataConfig.output_bin_size``.
            BPNet uses 1 bp bins, ASAP uses 4 bp bins.
        target_scale: Multiplicative scale applied to raw targets; matches
            ``DataConfig.target_scale``.
        input_scale: ``"raw"`` for raw-coverage bigWigs, ``"cpm"`` for
            CPM-normalized bigWigs.
        total_reads: Total mapped read count used for CPM normalization.
            Required when ``input_scale="cpm"``; ignored for raw.

    Returns:
        Scaled pseudocount suitable for ``ModelConfig.count_pseudocount``.

    Raises:
        ValueError: If inputs are non-positive, or if ``input_scale="cpm"``
            but ``total_reads`` is not supplied.
    """
    if reads_equiv <= 0:
        raise ValueError(f"reads_equiv must be positive, got {reads_equiv}")
    if read_length <= 0:
        raise ValueError(f"read_length must be positive, got {read_length}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    if target_scale <= 0:
        raise ValueError(f"target_scale must be positive, got {target_scale}")

    per_read_contribution = read_length / bin_size

    if input_scale == "raw":
        raw_pseudocount = reads_equiv * per_read_contribution
    elif input_scale == "cpm":
        if total_reads is None or total_reads <= 0:
            raise ValueError(
                "input_scale='cpm' requires total_reads > 0 "
                f"(library depth), got {total_reads}"
            )
        raw_pseudocount = reads_equiv * per_read_contribution * (1e6 / total_reads)
    else:
        raise ValueError(
            f"input_scale must be 'raw' or 'cpm', got {input_scale!r}"
        )

    scaled_pseudocount = raw_pseudocount * target_scale
    logger.info(
        "Resolved reads-equivalent pseudocount: %.4f scaled "
        "(reads_equiv=%.3f × read_length=%d / bin_size=%d × input_scale=%s × "
        "target_scale=%g)",
        scaled_pseudocount,
        reads_equiv,
        read_length,
        bin_size,
        input_scale,
        target_scale,
    )
    return scaled_pseudocount


def resolve_quantile_pseudocount(
    datamodule: _SupportsCountQuantileSamples,
    quantile: float = 0.10,
    n_samples: int = 2000,
    per_channel: bool = True,
) -> float:
    """Compute a quantile of training-region total counts as a shrinkage prior.

    Intended for Phase 2 ``DifferentialCountLoss`` training. Samples training
    intervals via the datamodule, computes each region's length-summed count
    (per-channel when ``per_channel=True``, cross-channel sum otherwise), and
    returns the specified quantile of the resulting pool — already in the same
    scaled units as the training targets.

    Feeding the return value into ``ModelConfig.count_pseudocount`` makes
    ``log((c_b + pc) / (c_a + pc))`` collapse toward zero for any region whose
    per-condition total is at or below the chosen quantile (i.e., in the
    bottom ``quantile`` fraction of training peaks), while leaving the ratio
    essentially unchanged for regions with counts well above the quantile.

    This is automatically scale-correct across raw vs. CPM bigWigs and
    across libraries of different depth.

    Args:
        datamodule: Setup ``CerberusDataModule`` whose training fold is the
            population the shrinkage prior should be calibrated against.
            Must have had ``setup(...)`` called.
        quantile: Quantile to compute from pooled per-region totals. Default
            ``0.10`` matches the "shrink the bottom 10% of peaks" heuristic.
        n_samples: Number of training intervals to sample.
        per_channel: If ``True`` (default), pool length-summed counts
            separately for each target channel (recommended for differential
            models — each condition's low-count regions should be shrunk
            independently). If ``False``, sum across channels first so one
            value per region enters the quantile.

    Returns:
        Quantile of pooled scaled counts, ready for
        ``ModelConfig.count_pseudocount``.

    Raises:
        ValueError: If ``quantile`` is outside ``(0, 1)``.
        RuntimeError: If the datamodule has not been setup.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    counts = datamodule.compute_count_quantile_samples(
        n_samples=n_samples,
        per_channel=per_channel,
    )
    if len(counts) == 0:
        raise RuntimeError(
            "No training regions sampled; cannot compute quantile pseudocount."
        )

    pseudocount = float(np.quantile(counts, quantile))
    logger.info(
        "Resolved quantile pseudocount: %.4f scaled "
        "(quantile=%.3f of %d per-%s length-summed counts, "
        "min=%.4f median=%.4f max=%.4f)",
        pseudocount,
        quantile,
        len(counts),
        "channel-region" if per_channel else "region",
        float(np.min(counts)),
        float(np.median(counts)),
        float(np.max(counts)),
    )
    return pseudocount
