"""Batched variant effect scoring using a trained model or ensemble.

Composes :func:`~cerberus.variants.variant_to_ref_alt` and
:func:`~cerberus.variants.compute_variant_effects` with batched model
inference (via :class:`~cerberus.model_ensemble.ModelEnsemble` or a plain
``nn.Module``) into a streaming variant-scoring pipeline.

Functions here follow the same pattern as :mod:`cerberus.predict_misc` —
free functions that accept a model/ensemble and compose existing primitives
into a convenient workflow.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pyfaidx
import torch
from torch import nn

from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ModelOutput, get_log_count_params
from cerberus.variants import Variant, compute_variant_effects, variant_to_ref_alt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantResult:
    """Container for a single variant's effect scores.

    Attributes:
        variant: The scored :class:`~cerberus.variants.Variant`.
        effects: Dict mapping metric names to per-channel tensors
            (shape ``(C,)``).
        interval: The reference-coordinate input window used for
            scoring (for provenance / fold-routing information).
    """

    variant: Variant
    effects: dict[str, torch.Tensor]
    interval: Interval


def score_variants(
    model: nn.Module | ModelEnsemble,
    variants: Iterable[Variant],
    fasta: pyfaidx.Fasta,
    input_len: int,
    *,
    log_counts_include_pseudocount: bool = False,
    pseudocount: float = 1.0,
    device: torch.device | None = None,
    batch_size: int = 64,
    use_folds: list[str] | None = None,
) -> Iterator[VariantResult]:
    """Score variant effects by comparing model predictions on ref vs alt sequences.

    For each variant, constructs one-hot ref and alt sequences via
    :func:`~cerberus.variants.variant_to_ref_alt`, runs both through the
    model, and computes effect metrics via
    :func:`~cerberus.variants.compute_variant_effects`.

    Variants that fail sequence construction (e.g. window extends beyond
    chromosome boundaries, ref allele mismatch) are logged as warnings
    and skipped — they do not interrupt the scoring run.

    .. note:: Sequence-only models

       This function constructs input tensors directly from the reference
       FASTA (4 one-hot channels).  Models trained with additional input
       signal tracks (``data_config.inputs`` non-empty) are not supported;
       pass the model through a dataset pipeline instead.

    Args:
        model: A trained model or :class:`ModelEnsemble`.  For an ensemble,
            fold routing uses the interval returned by ``variant_to_ref_alt``
            (centered on the variant in reference coordinates).
        variants: Iterable of :class:`Variant` objects to score.
        fasta: An open ``pyfaidx.Fasta`` for the reference genome.
        input_len: Model input length in bp (e.g. 2114).
        log_counts_include_pseudocount: Whether the model's ``log_counts``
            include a pseudocount offset.  Obtain from
            :func:`~cerberus.output.get_log_count_params`.
        pseudocount: The pseudocount value (default 1.0).
        device: Device for inference.  If ``None``, inferred from the model
            parameters.
        batch_size: Number of variants per forward-pass batch.
        use_folds: Fold roles to include when *model* is a
            :class:`ModelEnsemble` (e.g. ``["test", "val"]``).  Passed
            through to :meth:`ModelEnsemble.forward`.  Ignored for plain
            models.

    Yields:
        :class:`VariantResult` for each successfully scored variant, in
        input order.  Variants that fail construction are skipped (with
        a warning logged).
    """
    if device is None:
        device = next(model.parameters()).device

    is_ensemble = isinstance(model, ModelEnsemble)

    if not is_ensemble and use_folds is not None:
        logger.warning(
            "use_folds=%r was passed but model is a plain nn.Module, not a "
            "ModelEnsemble. Fold routing will not be applied.",
            use_folds,
        )

    effect_kwargs = {
        "log_counts_include_pseudocount": log_counts_include_pseudocount,
        "pseudocount": pseudocount,
    }

    n_scored = 0
    n_skipped = 0
    n_batches = 0

    for batch_tuple in itertools.batched(variants, batch_size):
        batch_variants = list(batch_tuple)

        # -- Construct ref/alt tensors, collecting successes --
        refs: list[torch.Tensor] = []
        alts: list[torch.Tensor] = []
        intervals: list[Interval] = []
        kept_variants: list[Variant] = []

        for v in batch_variants:
            try:
                ref_t, alt_t, iv = variant_to_ref_alt(v, fasta, input_len)
            except ValueError as exc:
                logger.warning("Skipping variant %s: %s", v, exc)
                n_skipped += 1
                continue
            refs.append(ref_t)
            alts.append(alt_t)
            intervals.append(iv)
            kept_variants.append(v)

        if not refs:
            continue

        ref_batch = torch.stack(refs).to(device)
        alt_batch = torch.stack(alts).to(device)

        # -- Forward passes (ref and alt through same folds) --
        with torch.no_grad():
            if is_ensemble:
                assert isinstance(model, ModelEnsemble)
                ref_output: ModelOutput = model.forward(
                    ref_batch, intervals=intervals, use_folds=use_folds
                )
                alt_output: ModelOutput = model.forward(
                    alt_batch, intervals=intervals, use_folds=use_folds
                )
            else:
                ref_output = model(ref_batch)
                alt_output = model(alt_batch)

        # -- Compute effects for the batch --
        effects = compute_variant_effects(ref_output, alt_output, **effect_kwargs)

        # -- Unbatch and yield per-variant results --
        for i, v in enumerate(kept_variants):
            per_variant: dict[str, torch.Tensor] = {}
            for key, tensor in effects.items():
                per_variant[key] = tensor[i].detach().cpu()
            yield VariantResult(variant=v, effects=per_variant, interval=intervals[i])

        n_scored += len(kept_variants)
        n_batches += 1
        if n_batches % 10 == 0:
            logger.info(
                "Progress: %d scored, %d skipped (%d batches)",
                n_scored, n_skipped, n_batches,
            )

    logger.info(
        "Variant scoring complete: %d scored, %d skipped", n_scored, n_skipped
    )


def score_variants_from_ensemble(
    ensemble: ModelEnsemble,
    variants: Iterable[Variant],
    fasta: pyfaidx.Fasta | None = None,
    *,
    batch_size: int = 64,
    use_folds: list[str] | None = None,
) -> Iterator[VariantResult]:
    """Convenience wrapper that extracts parameters from a :class:`ModelEnsemble`.

    Reads ``input_len``, ``fasta_path``, and pseudocount parameters from
    ``ensemble.cerberus_config``, opens the FASTA if not provided, and
    delegates to :func:`score_variants`.

    Args:
        ensemble: A loaded :class:`ModelEnsemble`.
        variants: Iterable of :class:`Variant` objects to score.
        fasta: An open ``pyfaidx.Fasta``.  If ``None``, opened from
            ``ensemble.cerberus_config.genome_config.fasta_path``.
        batch_size: Number of variants per forward-pass batch.
        use_folds: Fold roles to include (e.g. ``["test", "val"]``).

    Yields:
        :class:`VariantResult` for each successfully scored variant.

    Raises:
        ValueError: If ``data_config.use_sequence`` is ``False``.
        ValueError: If the model has input signal tracks
            (``data_config.inputs`` is non-empty), which requires a
            dataset pipeline not supported by this function.
    """
    config = ensemble.cerberus_config
    data_config = config.data_config

    if not data_config.use_sequence:
        raise ValueError(
            "data_config.use_sequence is False; variant scoring requires "
            "sequence-based models."
        )

    if data_config.inputs:
        raise ValueError(
            f"Model has input signal tracks ({list(data_config.inputs.keys())}). "
            "Variant scoring currently supports sequence-only models. "
            "Multi-channel model support requires apply_variant_to_tensor() "
            "(not yet implemented)."
        )

    log_counts_include_pseudocount, pseudocount = get_log_count_params(
        config.model_config_
    )

    own_fasta = False
    if fasta is None:
        fasta = pyfaidx.Fasta(str(config.genome_config.fasta_path))
        own_fasta = True

    try:
        yield from score_variants(
            model=ensemble,
            variants=variants,
            fasta=fasta,
            input_len=data_config.input_len,
            log_counts_include_pseudocount=log_counts_include_pseudocount,
            pseudocount=pseudocount,
            device=ensemble.device,
            batch_size=batch_size,
            use_folds=use_folds,
        )
    finally:
        if own_fasta:
            fasta.close()
