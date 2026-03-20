"""High-level inference utilities that compose ModelEnsemble, CerberusDataset,
and output transforms into common prediction workflows.

Functions here are *free functions* — they take a config dict or an ensemble as
an argument rather than living as methods on ``ModelEnsemble``.  This keeps
``ModelEnsemble`` focused on fold-routing and batched inference while providing
convenient entry points for scripts and notebooks.
"""
import logging
from pathlib import Path

from cerberus.config import CerberusConfig, get_log_count_params
from cerberus.dataset import CerberusDataset
from cerberus.exclude import get_exclude_intervals
from cerberus.genome import create_genome_folds
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import compute_total_log_counts
from cerberus.samplers import IntervalSampler, MultiSampler, create_sampler

logger = logging.getLogger(__name__)


def create_eval_dataset(config: CerberusConfig, in_memory: bool = False) -> CerberusDataset:
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
        genome_config=config["genome_config"],
        data_config=config["data_config"],
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
    genome_config = config["genome_config"]
    data_config = config["data_config"]
    sampler = IntervalSampler(
        file_path=Path(bed_path),
        chrom_sizes=genome_config["chrom_sizes"],
        padded_size=data_config["input_len"],
        folds=None,
        exclude_intervals=get_exclude_intervals(genome_config["exclude_intervals"]),
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
    genome_config = config["genome_config"]
    sampler_config = config["sampler_config"]
    data_config = config["data_config"]

    folds = create_genome_folds(
        genome_config["chrom_sizes"],
        genome_config["fold_type"],
        genome_config["fold_args"],
    )
    exclude_intervals = get_exclude_intervals(genome_config["exclude_intervals"])

    fold_args = genome_config["fold_args"]
    test_fold_idx = fold_args["test_fold"]
    val_fold_idx = fold_args["val_fold"]

    sampler_args = {**sampler_config["sampler_args"]}
    full_sampler_config = {
        **sampler_config,
        "sampler_args": sampler_args,
        "padded_size": data_config["input_len"],
    }

    combined_sampler = create_sampler(
        full_sampler_config,
        genome_config["chrom_sizes"],
        folds=folds,
        exclude_intervals=exclude_intervals,
        fasta_path=genome_config["fasta_path"],
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

    peak_intervals = [iv for iv, s in zip(intervals, sources) if s == "IntervalSampler"]

    if not include_background:
        return peak_intervals

    bg_intervals = [iv for iv, s in zip(intervals, sources) if s != "IntervalSampler"]
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
    model_config = ensemble.cerberus_config["model_config"]
    log_counts_include_pseudocount, count_pseudocount = get_log_count_params(model_config)

    results: list[float] = []
    for batch_output, _batch_intervals in ensemble.predict_intervals_batched(
        intervals, dataset, batch_size=batch_size,
    ):
        pred_log_total = compute_total_log_counts(
            batch_output,
            log_counts_include_pseudocount=log_counts_include_pseudocount,
            pseudocount=count_pseudocount,
        )
        results.extend(pred_log_total.cpu().tolist())
    return results
