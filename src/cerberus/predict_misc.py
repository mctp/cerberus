"""High-level inference utilities that compose ModelEnsemble, CerberusDataset,
and output transforms into common prediction workflows.

Functions here are *free functions* — they take a config or an ensemble as
an argument rather than living as methods on ``ModelEnsemble``.  This keeps
``ModelEnsemble`` focused on fold-routing and batched inference while providing
convenient entry points for scripts and notebooks.
"""

import itertools
import logging
from pathlib import Path

import torch

from cerberus.config import CerberusConfig
from cerberus.dataset import CerberusDataset
from cerberus.exclude import get_exclude_intervals
from cerberus.genome import create_genome_folds
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    compute_obs_total_log_counts,
    compute_total_log_counts,
    get_log_count_params,
)
from cerberus.samplers import IntervalSampler, MultiSampler, create_sampler

logger = logging.getLogger(__name__)


def create_eval_dataset(
    config: CerberusConfig, in_memory: bool = False
) -> CerberusDataset:
    """Create a :class:`CerberusDataset` configured for inference.

    Returns a dataset with no sampler and ``is_train=False`` (deterministic
    transforms, no jitter/RC augmentation).

    Args:
        config: Full cerberus config (as stored on ``ensemble.cerberus_config``).
        in_memory: If True, load all signal data into RAM.

    Returns:
        A ready-to-use CerberusDataset for inference.
    """
    return CerberusDataset(
        genome_config=config.genome_config,
        data_config=config.data_config,
        sampler_config=None,
        in_memory=in_memory,
        is_train=False,
    )


def load_bed_intervals(config: CerberusConfig, bed_path: str | Path) -> list[Interval]:
    """Load a BED file as centered intervals of the model's input length.

    Each interval is centered on the BED region midpoint (or peak summit for
    narrowPeak) and padded to ``input_len``.  Intervals overlapping exclude
    regions are filtered out.

    Args:
        config: Full cerberus config.
        bed_path: Path to a BED or narrowPeak file.

    Returns:
        List of :class:`Interval` objects.
    """
    genome_config = config.genome_config
    data_config = config.data_config
    sampler = IntervalSampler(
        file_path=Path(bed_path),
        chrom_sizes=genome_config.chrom_sizes,
        padded_size=data_config.input_len,
        folds=None,
        exclude_intervals=get_exclude_intervals(genome_config.exclude_intervals),
    )
    return list(sampler)


def get_eval_intervals(
    config: CerberusConfig,
    split: str = "test",
    include_background: bool = True,
    seed: int = 1234,
) -> tuple[list[Interval], list[Interval]] | list[Interval]:
    """Reconstruct evaluation intervals from the training config.

    Rebuilds the sampler and genome folds from ``config``, splits by fold, and
    separates peak from background intervals using ``get_interval_source``.

    Args:
        config: Full cerberus config (must include ``sampler_config``).
        split: Which fold split to return — ``"test"``, ``"val"``, or ``"train"``.
        include_background: If True (default), return ``(peak_intervals,
            bg_intervals)``.  If False, return only peak intervals.
        seed: Random seed for background sampling.

    Returns:
        ``(peak_intervals, bg_intervals)`` when *include_background* is True,
        otherwise a flat list of peak intervals.
    """
    genome_config = config.genome_config
    sampler_config = config.sampler_config
    data_config = config.data_config

    folds = create_genome_folds(
        genome_config.chrom_sizes,
        genome_config.fold_type,
        genome_config.fold_args,
    )
    exclude_intervals = get_exclude_intervals(genome_config.exclude_intervals)

    fold_args = genome_config.fold_args
    test_fold_idx = fold_args["test_fold"]
    val_fold_idx = fold_args["val_fold"]

    full_sampler_config = sampler_config.model_copy(
        update={"padded_size": data_config.input_len},
    )

    combined_sampler = create_sampler(
        full_sampler_config,
        genome_config.chrom_sizes,
        folds=folds,
        exclude_intervals=exclude_intervals,
        fasta_path=genome_config.fasta_path,
        seed=seed,
    )

    if not isinstance(combined_sampler, MultiSampler):
        raise RuntimeError("Expected sampler to be a MultiSampler.")

    train_s, val_s, test_s = combined_sampler.split_folds(test_fold_idx, val_fold_idx)
    split_sampler = {"test": test_s, "val": val_s, "train": train_s}[split]

    if not isinstance(split_sampler, MultiSampler):
        raise RuntimeError("Expected split sampler to be a MultiSampler.")

    n = len(split_sampler)
    intervals = [split_sampler[i] for i in range(n)]
    sources = [split_sampler.get_interval_source(i) for i in range(n)]

    peak_intervals = [
        iv for iv, s in zip(intervals, sources, strict=True) if s == "IntervalSampler"
    ]

    if not include_background:
        return peak_intervals

    bg_intervals = [
        iv for iv, s in zip(intervals, sources, strict=True) if s != "IntervalSampler"
    ]
    return peak_intervals, bg_intervals


def predict_log_counts(
    ensemble: ModelEnsemble,
    dataset: CerberusDataset,
    intervals: list[Interval],
    batch_size: int = 64,
) -> list[float]:
    """Predict total log-counts for a list of genomic intervals.

    Handles pseudocount detection automatically from the ensemble's model
    config via ``get_log_count_params``.

    Args:
        ensemble: Loaded model ensemble.
        dataset: Initialized dataset (from :func:`create_eval_dataset`).
        intervals: Genomic intervals to predict (must match model input_len).
        batch_size: Batch size for inference.

    Returns:
        List of predicted total log-count values (one per interval).
    """
    model_config = ensemble.cerberus_config.model_config_
    log_counts_include_pseudocount, count_pseudocount = get_log_count_params(
        model_config
    )

    results: list[float] = []
    for batch_output, _batch_intervals in ensemble.predict_intervals_batched(
        intervals,
        dataset,
        batch_size=batch_size,
    ):
        pred_log_total = compute_total_log_counts(
            batch_output,
            log_counts_include_pseudocount=log_counts_include_pseudocount,
            pseudocount=count_pseudocount,
        )
        results.extend(pred_log_total.cpu().tolist())
    return results


def observed_log_counts(
    dataset: CerberusDataset,
    intervals: list[Interval],
    config: CerberusConfig,
    batch_size: int = 64,
) -> list[float]:
    """Extract observed total log-counts from the target signal for each interval.

    Mirror of :func:`predict_log_counts` — together they enable comparing a
    model's predicted total log-counts against the ground-truth observed
    counts on the same set of intervals (e.g. for evaluation scatter plots).

    Each input interval is center-cropped to ``output_len`` and the target
    signal is extracted from ``dataset.target_signal_extractor`` over that
    window.  The result is summed and log-transformed in the same space as
    the model — with or without pseudocount, inferred from
    ``model_config.loss_cls`` via :func:`get_log_count_params`.

    Args:
        dataset: Initialized dataset (typically from
            :func:`create_eval_dataset`) whose ``target_signal_extractor``
            provides observed counts.
        intervals: Genomic intervals.  Each is center-cropped to
            ``config.data_config.output_len`` before signal extraction, so
            passing input-length intervals (the same list given to
            :func:`predict_log_counts`) is the typical pattern.
        config: Full cerberus config — supplies ``output_len``,
            ``target_scale``, and pseudocount parameters.
        batch_size: Number of intervals stacked per ``compute_obs_total_log_counts``
            call.  Has no effect on the result; tunes peak memory.

    Returns:
        List of observed total log-count values (one per interval).

    Raises:
        RuntimeError: If ``dataset.target_signal_extractor`` is None.
    """
    if dataset.target_signal_extractor is None:
        raise RuntimeError(
            "observed_log_counts requires dataset.target_signal_extractor; "
            "ensure data_config.targets is configured."
        )

    output_len = config.data_config.output_len
    target_scale = config.data_config.target_scale
    log_counts_include_pseudocount, count_pseudocount = get_log_count_params(
        config.model_config_
    )

    results: list[float] = []
    for batch in itertools.batched(intervals, batch_size):
        signals = [
            dataset.target_signal_extractor.extract(iv.center(output_len))
            for iv in batch
        ]
        raw = torch.stack(signals)  # (B, C, output_len)
        obs = compute_obs_total_log_counts(
            raw,
            target_scale=target_scale,
            log_counts_include_pseudocount=log_counts_include_pseudocount,
            pseudocount=count_pseudocount,
        )
        results.extend(obs.cpu().tolist())
    return results
