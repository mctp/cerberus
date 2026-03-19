import torch
import numpy as np
import logging
from collections.abc import Iterable
import pybigtools
from pathlib import Path

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    ModelOutput, ProfileCountOutput, ProfileLogRates,
    unbatch_modeloutput, aggregate_tensor_track_values,
)
from cerberus.samplers import SlidingWindowSampler

logger = logging.getLogger(__name__)


def predict_to_bigwig(
    output_path: str | Path,
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    stride: int | None = None,
    use_folds: list[str] | None = None,
    aggregation: str = "model",
    batch_size: int = 64,
    regions: list[Interval] | None = None,
) -> None:
    """
    Generates predictions and writes to a BigWig file.

    If *regions* is None (default), processes the entire genome chromosome-by-
    chromosome using a SlidingWindowSampler (respecting blacklists).  If *regions*
    is provided, only predicts within those intervals.

    Args:
        output_path: Path to the output BigWig file.
        dataset: Initialized CerberusDataset (provides data configuration and extractors).
        model_ensemble: Initialized ModelEnsemble (provides models).
        stride: Stride for sliding window predictions. If None, defaults to output_len // 2.
        use_folds: Folds to use.
        aggregation: Aggregation mode.
        batch_size: Batch size for inference.
        regions: Optional list of Intervals to restrict prediction to.
            When provided, only these regions are predicted (no blacklist filtering).
    """
    genome_config = dataset.genome_config
    data_config = dataset.data_config

    if use_folds is None:
        use_folds = ["test", "val"]

    if stride is None:
        stride = data_config["output_len"] // 2

    # Prepare generator for pybigtools
    def stream_generator() -> Iterable[tuple[str, int, int, float]]:
        chrom_sizes = genome_config["chrom_sizes"]
        input_len = data_config["input_len"]
        output_len = data_config["output_len"]
        offset = (input_len - output_len) // 2

        if regions is not None:
            # Predict only within specified regions
            # Sort by chrom then start for BigWig write order
            sorted_regions = sorted(regions, key=lambda r: (r.chrom, r.start))
            for region in sorted_regions:
                if region.chrom not in chrom_sizes:
                    logger.warning(f"Skipping region on unknown chrom: {region.chrom}")
                    continue
                # Tile the region with input-length windows
                windows: list[Interval] = []
                pos = region.start - offset
                while pos + input_len <= region.end + offset + input_len:
                    win_start = max(0, pos)
                    win_end = win_start + input_len
                    if win_end > chrom_sizes[region.chrom]:
                        break
                    windows.append(Interval(region.chrom, win_start, win_end, "+"))
                    pos += stride
                    if win_end >= region.end + offset:
                        break
                if windows:
                    yield from _process_island(
                        windows, dataset, model_ensemble,
                        use_folds, aggregation, batch_size,
                    )
        else:
            # Genome-wide prediction
            allowed_chroms = genome_config["allowed_chroms"]
            exclude_intervals = dataset.exclude_intervals

            for chrom in allowed_chroms:
                if chrom not in chrom_sizes:
                    continue

                sampler = SlidingWindowSampler(
                    chrom_sizes={chrom: chrom_sizes[chrom]},
                    padded_size=input_len,
                    stride=stride,  # type: ignore (checked above)
                    exclude_intervals=exclude_intervals,
                    folds=[],
                )

                current_island: list[Interval] = []
                prev_input_start = -999999

                for window in sampler:
                    if current_island and (window.start >= prev_input_start + output_len):
                        yield from _process_island(
                            current_island,
                            dataset,
                            model_ensemble,
                            use_folds,
                            aggregation,
                            batch_size,
                        )
                        current_island = []

                    current_island.append(window)
                    prev_input_start = window.start

                if current_island:
                    yield from _process_island(
                        current_island,
                        dataset,
                        model_ensemble,
                        use_folds,
                        aggregation,
                        batch_size,
                    )

    logger.info(f"Writing BigWig to {output_path}...")
    bw = pybigtools.open(str(output_path), "w")  # type: ignore
    bw.write(genome_config["chrom_sizes"], stream_generator())


def _reconstruct_linear_signal(output: ModelOutput) -> torch.Tensor:
    """
    Converts a per-window model output to linear signal (counts per bin).

    Reconstruction depends on the output type:
      - ProfileCountOutput (BPNet/Dalmatian): softmax(logits) * exp(log_counts)
      - ProfileLogRates: exp(log_rates)
      - ProfileLogits (fallback): raw logits (no absolute scale)

    Args:
        output: A single (unbatched) model output with shape (C, L) tensors.

    Returns:
        Tensor of shape (C, L) with linear signal values.
    """
    if isinstance(output, ProfileCountOutput):
        logits = output.logits.float()       # (C, L)
        log_counts = output.log_counts.float()  # (C,)
        # Numerically stable softmax over the length axis
        logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)  # (C, L)
        total_counts = torch.exp(log_counts).unsqueeze(-1)  # (C, 1)
        return probs * total_counts
    elif isinstance(output, ProfileLogRates):
        return torch.exp(output.log_rates.float())
    elif hasattr(output, "logits"):
        logger.warning(
            "Model output has logits but no log_counts or log_rates. "
            "Exporting raw logits — values will not match input BigWig scale."
        )
        return output.logits  # type: ignore[union-attr]
    else:
        raise ValueError(
            f"Cannot extract profile track from output type {type(output).__name__}"
        )


def _process_island(
    island_intervals: list[Interval],
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    use_folds: list[str],
    aggregation: str,
    batch_size: int,
) -> Iterable[tuple[str, int, int, float]]:
    """
    Runs prediction on a contiguous island of intervals and yields
    per-bp linear counts matching the scale of the input BigWig.

    Linear signal is reconstructed per-window (softmax + count scaling)
    before spatial aggregation, because the softmax normalization is only
    valid within the model's output_len, not across the merged island.

    The result is divided by target_scale and output_bin_size to undo the
    training target transforms and produce average per-bp signal.
    """
    target_scale = dataset.data_config["target_scale"]
    output_bin_size = dataset.data_config["output_bin_size"]
    output_len = dataset.data_config["output_len"]

    # Collect per-window linear signals and their output intervals
    linear_signals: list[torch.Tensor] = []
    output_intervals: list[Interval] = []

    for batched_output, batch_intervals in model_ensemble.predict_intervals_batched(
        island_intervals,
        dataset,
        use_folds=use_folds,
        aggregation="model",
        batch_size=batch_size,
    ):
        # Unbatch into per-window outputs
        unbatched = unbatch_modeloutput(batched_output, len(batch_intervals))
        output_cls = type(batched_output)

        for interval, out_dict in zip(batch_intervals, unbatched):
            # Reconstruct as the original ModelOutput type for _reconstruct_linear_signal
            single_output = output_cls(**out_dict)
            signal = _reconstruct_linear_signal(single_output)  # (C, L)
            linear_signals.append(signal)
            output_intervals.append(interval.center(output_len))

    if not linear_signals:
        return

    # Spatially merge overlapping linear signals
    chrom = output_intervals[0].chrom
    strand = output_intervals[0].strand
    min_start = min(i.start for i in output_intervals)
    max_end = max(i.end for i in output_intervals)
    merged_interval = Interval(chrom, min_start, max_end, strand)

    track_data = aggregate_tensor_track_values(
        linear_signals, output_intervals, merged_interval, output_len, output_bin_size
    )
    # Apply inverse target transforms
    track_data = track_data / target_scale / output_bin_size

    # Select first channel for BigWig output
    while track_data.ndim > 2:
        track_data = track_data[0]  # squeeze leading dims
    n_channels = track_data.shape[0]
    if n_channels > 1:
        logger.warning(
            "Model output has %d channels but BigWig is single-track; "
            "only channel 0 will be exported. Other channels are discarded.",
            n_channels,
        )
    values = track_data[0]  # Shape: (Bins,)

    start = merged_interval.start
    logger.info("island: %s:%d-%d  len=%d", chrom, start, merged_interval.end, len(values))

    for i, val in enumerate(values):
        bin_start = start + i * output_bin_size
        bin_end = bin_start + output_bin_size

        yield (chrom, bin_start, bin_end, float(val))
