"""Scale-aware pseudocount helpers for count-head training.

Two helpers, named for the distinct role the pseudocount plays in the
count loss it feeds.  See ``docs/internal/count_log_spaces.md`` and
``docs/internal/asap_pseudocount_considerations.md`` for background.

Absolute-count log losses (``log(count + pc)``)
-----------------------------------------------
:func:`resolve_read_coverage_pseudocount` converts a user-facing
read-coverage specification (e.g. "the pseudocount equivalent to one
read of coverage at this bin size and target scale") into the
corresponding scaled value for ``ModelConfig.count_pseudocount``.  Here
the pseudocount prevents ``log(0)`` for silent regions and anchors the
zero-reads cluster into the rest of the count distribution; the natural
unit is "one read's worth of signal."

Log-fold-change losses (``log((c_b + pc) / (c_a + pc))``)
---------------------------------------------------------
:func:`resolve_noise_floor_pseudocount` derives the pseudocount from
training data: it samples per-channel total counts and returns the
chosen quantile (default 10th percentile) per channel, then takes the
**maximum** across channels.  In a log-fold-change loss the pseudocount
acts as an empirical-Bayes shrinkage prior that pulls the log-ratio
toward zero whenever both per-channel totals are at or below the noise
floor.  Picking ``max`` across per-channel quantiles keeps the
higher-coverage channel's noise floor shrunk too; ``min`` would leave
its quiet regions producing large unshrunk log-ratios.

Both helpers are mathematically pseudocounts (additive offsets to
counts in the same scaled units as the training targets), but the
second is closer in spirit to edgeR's ``prior.count`` / DESeq2's
``lfcShrink``.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import numpy as np

logger = logging.getLogger(__name__)

InputScale = Literal["raw", "cpm"]


class _SupportsCountQuantileSamples(Protocol):
    """Structural type for the one method :func:`resolve_noise_floor_pseudocount` consumes.

    :class:`cerberus.datamodule.CerberusDataModule` is the production
    implementor; lightweight test stubs satisfy this naturally without
    inheriting from the full datamodule.
    """

    def compute_count_quantile_samples(
        self,
        n_samples: int = ...,
        per_channel: bool = ...,
        seed: int | None = ...,
    ) -> np.ndarray: ...


def resolve_read_coverage_pseudocount(
    reads_equiv: float,
    read_length: int,
    bin_size: int,
    target_scale: float,
    input_scale: InputScale = "raw",
    total_reads: float | None = None,
) -> float:
    """Compute a scaled pseudocount from a read-coverage specification.

    The length-summed target for one region is approximately::

        counts = n_reads × (read_length / bin_size) × norm_factor × target_scale

    where ``norm_factor`` is ``1`` for raw-coverage bigWigs and
    ``1e6 / total_reads`` for CPM-normalised bigWigs.  This helper returns
    the scaled value corresponding to ``reads_equiv`` reads' worth of
    coverage, so passing the result as ``ModelConfig.count_pseudocount`` is
    equivalent to saying "shrink log-counts near the noise floor where a
    region has roughly ``reads_equiv`` reads or fewer."

    Args:
        reads_equiv: Number of reads the pseudocount should correspond to.
            ``1.0`` is the typical default (avoid ``log(0)``, keep the
            zero cluster embedded in the distribution).
        read_length: Sequencing read or fragment length in bp.  ChIP-seq
            reads ~150 bp, ATAC-seq fragments ~100 bp.
        bin_size: Output bin size; matches ``DataConfig.output_bin_size``.
            BPNet uses 1 bp bins, ASAP uses 4 bp bins.
        target_scale: Multiplicative scale applied to raw targets; matches
            ``DataConfig.target_scale``.
        input_scale: ``"raw"`` for raw-coverage bigWigs, ``"cpm"`` for
            CPM-normalised bigWigs.
        total_reads: Total mapped read count used for CPM normalisation.
            Required when ``input_scale="cpm"``; ignored for raw.

    Returns:
        Scaled pseudocount suitable for ``ModelConfig.count_pseudocount``.

    Raises:
        ValueError: If inputs are non-positive, ``input_scale`` is
            unknown, or ``input_scale="cpm"`` but ``total_reads`` is not
            supplied.
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
        "Resolved read-coverage pseudocount: %.4f scaled "
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


def resolve_noise_floor_pseudocount(
    datamodule: _SupportsCountQuantileSamples,
    quantile: float = 0.10,
    n_samples: int = 2000,
    seed: int | None = None,
) -> float:
    """Derive a noise-floor pseudocount from training-data quantiles.

    Intended as the shrinkage prior for log-fold-change losses such as
    ``DifferentialCountLoss``.  Samples training intervals via the
    datamodule, computes each region's *per-channel* length-summed
    counts, takes ``quantile`` along each channel, and returns the
    **maximum** of those per-channel quantiles — a single scalar in the
    same scaled units as the training targets.

    Per-channel maximum is the correct combination for log-fold-change
    losses: with channels A and B at different depths (``q_A < q_B``),
    using ``q_B`` keeps both noise floors at log-ratios within
    ``±log(2)``, whereas ``q_A`` leaves the deeper channel's quiet
    regions producing large unshrunk log-ratios.

    Feeding the return value into ``ModelConfig.count_pseudocount``
    makes ``log((c_b + pc) / (c_a + pc))`` collapse toward zero for any
    region whose per-channel total sits at or below the chosen
    quantile, while leaving the ratio essentially unchanged for regions
    with counts well above it.

    The result is automatically scale-correct across raw vs. CPM
    bigWigs and across libraries of different depths — no user
    reasoning about read length or normalisation factor is required.

    Args:
        datamodule: A setup :class:`CerberusDataModule` whose training
            fold is the population the prior should be calibrated against.
            ``setup(...)`` must have been called.
        quantile: Quantile of training-region per-channel totals to use.
            Default ``0.10`` matches the "shrink the bottom 10% of peaks"
            heuristic.
        n_samples: Number of training intervals to sample.
        seed: RNG seed forwarded to the datamodule's sampler.  ``None``
            uses OS entropy; pass a fixed integer for reproducible
            pseudocount calibration across runs.

    Returns:
        Scaled pseudocount ready for ``ModelConfig.count_pseudocount``.

    Raises:
        ValueError: If ``quantile`` is outside ``(0, 1)``.
        RuntimeError: If the datamodule has not been setup, or the
            sampler returned an empty pool.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    counts = datamodule.compute_count_quantile_samples(
        n_samples=n_samples,
        per_channel=True,
        seed=seed,
    )
    if counts.size == 0:
        raise RuntimeError(
            "No training regions sampled; cannot compute noise-floor pseudocount."
        )

    # 2D shape (N, C): take the quantile along axis=0 (per channel), then
    # max across channels.  See module docstring for the per-channel-max
    # rationale.
    per_channel_q = np.quantile(counts, quantile, axis=0)
    pseudocount = float(np.max(per_channel_q))

    logger.info(
        "Resolved noise-floor pseudocount: %.4f scaled "
        "(quantile=%.3f of %d region(s) × %d channel(s); "
        "per-channel quantiles=%s, max selected)",
        pseudocount,
        quantile,
        counts.shape[0],
        counts.shape[1],
        np.array2string(per_channel_q, precision=4),
    )
    return pseudocount
