import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pybigtools
import torch

from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    ModelOutput,
    ProfileCountOutput,
    ProfileLogRates,
    get_log_count_params,
    unbatch_modeloutput,
)
from cerberus.samplers import SlidingWindowSampler

logger = logging.getLogger(__name__)


def predict_to_bigwig(
    output_path: str | Path,
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    stride: int | None = None,
    use_folds: list[str] | None = None,
    batch_size: int = 64,
    regions: list[Interval] | None = None,
) -> None:
    """Generates predictions and writes to a BigWig file.

    If *regions* is None (default), processes the entire genome chromosome-by-
    chromosome using a SlidingWindowSampler (respecting blacklists).  If *regions*
    is provided, only predicts within those intervals.

    Per-window signal is reconstructed via softmax + count scaling before spatial
    merging, so model aggregation is always "model" (per-window outputs required).

    The pseudocount for inverting log_counts (MSE-trained models) is determined
    automatically from the model configuration via ``get_log_count_params``.

    Args:
        output_path: Path to the output BigWig file.
        dataset: Initialized CerberusDataset (provides data configuration and extractors).
        model_ensemble: Initialized ModelEnsemble (provides models).
        stride: Stride for sliding window predictions. If None, defaults to output_len // 2.
        use_folds: Folds to use.
        batch_size: Batch size for inference.
        regions: Optional list of Intervals to restrict prediction to.
            When provided, only these regions are predicted (no blacklist filtering).
    """
    genome_config = dataset.genome_config
    data_config = dataset.data_config

    _, count_pseudocount = get_log_count_params(
        model_ensemble.cerberus_config.model_config_
    )
    logger.info("count_pseudocount=%.4g (from model config)", count_pseudocount)

    if use_folds is None:
        use_folds = ["test", "val"]

    if stride is None:
        stride = data_config.output_len // 2

    # Prepare generator for pybigtools
    def stream_generator() -> Iterable[tuple[str, int, int, float]]:
        chrom_sizes = genome_config.chrom_sizes
        input_len = data_config.input_len
        output_len = data_config.output_len
        offset = (input_len - output_len) // 2

        if regions is not None:
            # Predict only within specified regions
            # Sort by chrom then start for BigWig write order
            sorted_regions = sorted(regions, key=lambda r: (r.chrom, r.start))
            for region in sorted_regions:
                if region.chrom not in chrom_sizes:
                    logger.warning(f"Skipping region on unknown chrom: {region.chrom}")
                    continue
                # Tile the region with input-length windows.
                # Clamp start so windows never begin before position 0.
                windows: list[Interval] = []
                ideal_start = region.start - offset
                pos = max(0, ideal_start)
                if pos != ideal_start:
                    logger.warning(
                        "Region %s starts within %d bp of chromosome start; "
                        "predictions will begin at position %d instead of %d",
                        region,
                        offset,
                        pos + offset,
                        region.start,
                    )
                while pos + input_len <= region.end + offset + input_len:
                    win_end = pos + input_len
                    if win_end > chrom_sizes[region.chrom]:
                        break
                    windows.append(Interval(region.chrom, pos, win_end, "+"))
                    pos += stride
                    if win_end >= region.end + offset:
                        break
                if windows:
                    yield from _process_island(
                        windows,
                        dataset,
                        model_ensemble,
                        use_folds,
                        batch_size,
                        count_pseudocount,
                    )
        else:
            # Genome-wide prediction
            allowed_chroms = genome_config.allowed_chroms
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
                    if current_island and (
                        window.start >= prev_input_start + output_len
                    ):
                        yield from _process_island(
                            current_island,
                            dataset,
                            model_ensemble,
                            use_folds,
                            batch_size,
                            count_pseudocount,
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
                        batch_size,
                        count_pseudocount,
                    )

    logger.info(f"Writing BigWig to {output_path}...")
    bw = pybigtools.open(str(output_path), "w")  # type: ignore
    try:
        bw.write(genome_config.chrom_sizes, stream_generator())
    finally:
        bw.close()


def _reconstruct_linear_signal(
    output: ModelOutput, count_pseudocount: float = 0.0
) -> torch.Tensor:
    """
    Converts a per-window model output to linear signal (counts per bin).

    Reconstruction depends on the output type:
      - ProfileCountOutput (BPNet/Dalmatian): softmax(logits) * (exp(log_counts) - pseudocount)
      - ProfileLogRates: exp(log_rates)
      - ProfileLogits (fallback): raw logits (no absolute scale)

    Args:
        output: A single (unbatched) model output with shape (C, L) tensors.
        count_pseudocount: Pseudocount to subtract from exp(log_counts) for
            models trained with MSE loss in log(count + pseudocount) space.
            0.0 means no inversion (Poisson/NB losses).

    Returns:
        Tensor of shape (C, L) with linear signal values.
    """
    if isinstance(output, ProfileCountOutput):
        logits = output.logits.float()  # (C, L)
        log_counts = output.log_counts.float()  # (C,)
        # Numerically stable softmax over the length axis
        logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)  # (C, L)
        total_counts = (
            (torch.exp(log_counts) - count_pseudocount).clamp_min(0.0).unsqueeze(-1)
        )  # (C, 1)
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
    batch_size: int,
    count_pseudocount: float,
) -> Iterable[tuple[str, int, int, float]]:
    """Runs prediction on a contiguous island of intervals and yields
    per-bp linear counts matching the scale of the input BigWig.

    Linear signal is reconstructed per-window (softmax + count scaling)
    and accumulated into a streaming accumulator, so memory usage is bounded
    by the accumulator array rather than the number of windows.

    The result is divided by target_scale and output_bin_size to undo the
    training target transforms and produce average per-bp signal.
    """
    if not island_intervals:
        return

    target_scale = dataset.data_config.target_scale
    output_bin_size = dataset.data_config.output_bin_size
    output_len = dataset.data_config.output_len
    input_len = dataset.data_config.input_len
    offset = (input_len - output_len) // 2

    # Compute merged output interval upfront from input intervals
    chrom = island_intervals[0].chrom
    min_start = min(iv.start + offset for iv in island_intervals)
    max_end = max(iv.start + offset + output_len for iv in island_intervals)
    merged_interval = Interval(chrom, min_start, max_end, island_intervals[0].strand)

    span_bp = max_end - min_start
    n_bins = span_bp // output_bin_size
    interval_bins = output_len // output_bin_size

    # Streaming accumulator — allocated on first signal (need n_channels)
    accumulator: np.ndarray | None = None
    counts = np.zeros((1, n_bins), dtype=np.float32)

    for batched_output, batch_intervals in model_ensemble.predict_intervals_batched(
        island_intervals,
        dataset,
        use_folds=use_folds,
        aggregation="model",
        batch_size=batch_size,
    ):
        unbatched = unbatch_modeloutput(batched_output, len(batch_intervals))
        output_cls = type(batched_output)

        for interval, out_dict in zip(batch_intervals, unbatched, strict=True):
            single_output = output_cls(**out_dict)
            signal = _reconstruct_linear_signal(single_output, count_pseudocount)
            val = signal.cpu().numpy()  # (C, L)

            out_start = interval.start + offset
            rel_start_bin = (out_start - min_start) // output_bin_size
            end_bin = rel_start_bin + interval_bins

            if accumulator is None:
                n_channels = val.shape[0]
                accumulator = np.zeros((n_channels, n_bins), dtype=np.float32)

            accumulator[:, rel_start_bin:end_bin] += val
            counts[:, rel_start_bin:end_bin] += 1.0

    if accumulator is None:
        return

    # Average overlapping predictions
    counts = np.maximum(counts, 1.0)
    track_data = accumulator / counts

    # Undo training transforms (Scale then Bin) to recover per-bp signal.
    # Training: raw * target_scale → binned. Inverse: / target_scale / bin_size.
    track_data = track_data / target_scale / output_bin_size

    # Select first channel for BigWig output
    n_channels = track_data.shape[0]
    if n_channels > 1:
        logger.warning(
            "Model output has %d channels but BigWig is single-track; "
            "only channel 0 will be exported. Other channels are discarded.",
            n_channels,
        )
    values = track_data[0]  # (n_bins,)

    start = merged_interval.start
    logger.info(
        "island: %s:%d-%d  len=%d", chrom, start, merged_interval.end, len(values)
    )

    for i, val in enumerate(values):
        bin_start = start + i * output_bin_size
        bin_end = bin_start + output_bin_size
        yield (chrom, bin_start, bin_end, float(val))
