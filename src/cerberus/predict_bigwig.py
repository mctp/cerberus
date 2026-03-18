import torch
import logging
from collections.abc import Iterable
import pybigtools
from pathlib import Path

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.samplers import SlidingWindowSampler
from cerberus.output import (
    ModelOutput,
    ProfileCountOutput,
    ProfileLogits,
    ProfileLogRates,
    aggregate_tensor_track_values,
)

logger = logging.getLogger(__name__)


def _reconstruct_profile_counts(
    logits: torch.Tensor, log_counts: torch.Tensor
) -> torch.Tensor:
    """Reconstruct profile counts from BPNet-style outputs.

    Formula matches chrombpnet-pytorch export semantics:
    `softmax(logits) * exp(log_counts)`.

    Supports batched `(B, C, L)` logits or unbatched `(C, L)` logits.
    """
    squeeze_batch = False
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        squeeze_batch = True
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits to have shape (B, C, L) or (C, L), got {tuple(logits.shape)}"
        )

    if log_counts.ndim == 1:
        log_counts = log_counts.unsqueeze(0)
    if log_counts.ndim != 2:
        raise ValueError(
            f"Expected log_counts to have shape (B, C) or (B, 1), got {tuple(log_counts.shape)}"
        )
    if logits.shape[0] != log_counts.shape[0]:
        raise ValueError(
            "Batch dimension mismatch between logits and log_counts: "
            f"{logits.shape[0]} vs {log_counts.shape[0]}"
        )
    if log_counts.shape[1] not in (1, logits.shape[1]):
        raise ValueError(
            "Channel dimension mismatch between logits and log_counts: "
            f"{logits.shape[1]} vs {log_counts.shape[1]}"
        )

    probs = torch.softmax(logits.float(), dim=-1)
    total_counts = torch.exp(log_counts.float()).unsqueeze(-1)
    pred_counts = probs * total_counts
    return pred_counts.squeeze(0) if squeeze_batch else pred_counts


def _select_track_tensor(output: ModelOutput) -> torch.Tensor:
    """Selects the profile track to export from a model output object."""
    if isinstance(output, ProfileCountOutput):
        expected_count_ndim = output.logits.ndim - 1
        if output.log_counts.ndim == expected_count_ndim:
            return _reconstruct_profile_counts(output.logits, output.log_counts)
        logger.warning(
            "ProfileCountOutput log_counts has shape %s incompatible with per-window reconstruction "
            "(expected ndim=%d). Falling back to logits export.",
            tuple(output.log_counts.shape),
            expected_count_ndim,
        )
        return output.logits.float()
    if isinstance(output, ProfileLogRates):
        return output.log_rates.float()
    if isinstance(output, ProfileLogits):
        return output.logits.float()
    raise ValueError(f"Unsupported model output type for BigWig export: {type(output).__name__}")


def predict_to_bigwig(
    output_path: str | Path,
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    stride: int | None = None,
    use_folds: list[str] | None = None,
    aggregation: str = "model",
    batch_size: int = 64,
) -> None:
    """
    Generates predictions for all chromosomes and writes to a BigWig file.

    This function processes the genome chromosome-by-chromosome using a SlidingWindowSampler
    to generate valid input windows (respecting blacklists) and predict_intervals to
    aggregate predictions into contiguous islands. The results are streamed to a BigWig file.

    Args:
        output_path: Path to the output BigWig file.
        dataset: Initialized CerberusDataset (provides data configuration and extractors).
        model_ensemble: Initialized ModelEnsemble (provides models).
        stride: Stride for sliding window predictions. If None, defaults to output_len // 2.
        use_folds: Folds to use.
        aggregation: Aggregation mode.
        device: Device to run inference on.
        batch_size: Batch size for inference.
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
        allowed_chroms = genome_config["allowed_chroms"]
        input_len = data_config["input_len"]
        output_len = data_config["output_len"]
        exclude_intervals = dataset.exclude_intervals

        for chrom in allowed_chroms:
            if chrom not in chrom_sizes:
                continue

            # Create specific sampler for this chromosome to generate windows
            # We use input_len as padded_size because predict_intervals expects inputs of that length
            sampler = SlidingWindowSampler(
                chrom_sizes={chrom: chrom_sizes[chrom]},
                padded_size=input_len,
                stride=stride,  # type: ignore (checked above)
                exclude_intervals=exclude_intervals,
                folds=[],  # No folds needed for prediction generation
            )

            current_island: list[Interval] = []
            prev_input_start = -999999

            for window in sampler:
                # Check for gaps:
                # - Model output start: `window.start + offset`
                # - Previous window output end: `prev_input_start + offset + output_len`
                # - Gap (no overlap) condition: `Output_Start >= Previous_Output_End`
                # - `window.start + offset >= prev_input_start + offset + output_len`
                # - `window.start >= prev_input_start + output_len`
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
    # Using pybigtools to write the stream
    bw = pybigtools.open(str(output_path), "w")  # type: ignore
    bw.write(genome_config["chrom_sizes"], stream_generator())


def _process_island(
    island_intervals: list[Interval],
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    use_folds: list[str],
    aggregation: str,
    batch_size: int,
) -> Iterable[tuple[str, int, int, float]]:
    """
    Runs prediction on a contiguous island of intervals and yields values.
    """
    output_len = dataset.data_config["output_len"]
    output_bin_size = dataset.data_config["output_bin_size"]

    tracks: list[torch.Tensor] = []
    output_intervals: list[Interval] = []

    for batch_output, batch_intervals in model_ensemble.predict_intervals_batched(
        island_intervals,
        dataset,
        use_folds=use_folds,
        aggregation=aggregation,
        batch_size=batch_size,
    ):
        track_tensor = _select_track_tensor(batch_output).detach()

        if aggregation == "model":
            if track_tensor.ndim != 3:
                raise ValueError(
                    f"Expected batched track tensor with shape (B, C, L), got {tuple(track_tensor.shape)}"
                )
            if track_tensor.shape[0] != len(batch_intervals):
                raise ValueError(
                    "Batch size mismatch between predictions and intervals: "
                    f"{track_tensor.shape[0]} vs {len(batch_intervals)}"
                )
            for idx, interval in enumerate(batch_intervals):
                tracks.append(track_tensor[idx])
                output_intervals.append(interval.center(output_len))
        elif aggregation == "interval+model":
            if track_tensor.ndim != 2:
                raise ValueError(
                    f"Expected merged track tensor with shape (C, L), got {tuple(track_tensor.shape)}"
                )
            if batch_output.out_interval is None:
                raise ValueError("ModelOutput.out_interval is None, but expected merged interval.")
            tracks.append(track_tensor)
            output_intervals.append(batch_output.out_interval)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation} (supported: 'model', 'interval+model')")

    if not tracks:
        return

    merged_interval = Interval(
        chrom=output_intervals[0].chrom,
        start=min(iv.start for iv in output_intervals),
        end=max(iv.end for iv in output_intervals),
        strand=output_intervals[0].strand,
    )
    track_data = aggregate_tensor_track_values(
        outputs=tracks,
        intervals=output_intervals,
        merged_interval=merged_interval,
        output_len=output_len,
        output_bin_size=output_bin_size,
    )

    # TODO(#32): Add a `channel` parameter to predict_to_bigwig / _process_island
    # so callers can choose which channel to export. BigWig is single-track, so
    # multi-channel export would require one file per channel or an aggregation
    # strategy (sum/mean).  For now we always take the first channel.
    n_channels = int(track_data.shape[0])
    if n_channels > 1:
        logger.warning(
            "Model output has %d channels but BigWig is single-track; "
            "only channel 0 will be exported. Other channels are discarded.",
            n_channels,
        )
    values = track_data[0]  # Shape: (Bins,)

    chrom = merged_interval.chrom
    start = merged_interval.start

    for i, val in enumerate(values):
        bin_start = start + i * output_bin_size
        bin_end = bin_start + output_bin_size
        
        yield (chrom, bin_start, bin_end, float(val))
