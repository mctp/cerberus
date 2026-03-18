import logging
from collections.abc import Iterable
from pathlib import Path

import pybigtools
import torch

from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    ModelOutput,
    ProfileCountOutput,
    ProfileLogits,
    ProfileLogRates,
    aggregate_tensor_track_values,
)
from cerberus.samplers import IntervalSampler, SlidingWindowSampler

logger = logging.getLogger(__name__)

_VALID_REGION_GENERATION_MODES = {"sliding_window", "peaks_nonpeaks"}
_VALID_OVERLAP_RESOLUTION_MODES = {"average", "midpoint_between_summits"}


def _validate_mode(name: str, value: str, valid_values: set[str]) -> None:
    if value not in valid_values:
        valid = ", ".join(sorted(valid_values))
        raise ValueError(f"Invalid {name}: {value!r}. Expected one of: {valid}.")


def _split_sorted_intervals_into_islands(
    intervals: Iterable[Interval],
    output_len: int,
) -> Iterable[list[Interval]]:
    current_island: list[Interval] = []
    prev_input_start = -999999

    for interval in intervals:
        if current_island and (interval.start >= prev_input_start + output_len):
            yield current_island
            current_island = []

        current_island.append(interval)
        prev_input_start = interval.start

    if current_island:
        yield current_island


def _load_peak_nonpeak_intervals_for_chrom(
    peaks_path: str | Path,
    nonpeaks_path: str | Path | None,
    chrom: str,
    chrom_sizes: dict[str, int],
    padded_size: int,
    exclude_intervals: dict | None,
) -> list[Interval]:
    if chrom not in chrom_sizes:
        raise ValueError(f"Chromosome {chrom!r} not found in chrom_sizes.")

    chrom_only_sizes = {chrom: int(chrom_sizes[chrom])}
    exclude = exclude_intervals if exclude_intervals is not None else {}

    def _load(path: str | Path) -> list[Interval]:
        sampler = IntervalSampler(
            file_path=Path(path),
            chrom_sizes=chrom_only_sizes,
            padded_size=padded_size,
            folds=[],
            exclude_intervals=exclude,
        )
        return list(sampler)

    intervals = _load(peaks_path)
    if nonpeaks_path is not None:
        intervals.extend(_load(nonpeaks_path))

    # Match chrombpnet ordering behavior used for overlap resolution:
    # sort by (chrom, start, summit/midpoint, end). Chrom is fixed here.
    intervals.sort(key=lambda iv: (iv.start, (iv.start + iv.end) // 2, iv.end))
    return intervals


def _stitch_tracks_midpoint_between_summits(
    tracks: list[torch.Tensor],
    intervals: list[Interval],
    summit_positions: list[int],
    output_bin_size: int,
) -> Iterable[tuple[str, int, int, float]]:
    if not tracks:
        return
    if not (len(tracks) == len(intervals) == len(summit_positions)):
        raise ValueError(
            "Length mismatch for midpoint stitching: "
            f"{len(tracks)} tracks, {len(intervals)} intervals, {len(summit_positions)} summits."
        )

    has_multi_channel = any(t.ndim >= 2 and int(t.shape[0]) > 1 for t in tracks)
    if has_multi_channel:
        logger.warning(
            "Model output has multiple channels but BigWig is single-track; "
            "only channel 0 will be exported. Other channels are discarded."
        )

    # Convert to (chrom, start, end, summit, values[channel0]) rows and sort,
    # mirroring chrombpnet write_bigwig ordering.
    rows: list[tuple[str, int, int, int, torch.Tensor]] = []
    for track, interval, summit in zip(tracks, intervals, summit_positions):
        if track.ndim == 2:
            values = track[0].detach().cpu()
        elif track.ndim == 1:
            values = track.detach().cpu()
        else:
            raise ValueError(
                "Midpoint stitching expects per-interval track tensors with shape (C, L) or (L,), "
                f"got {tuple(track.shape)}"
            )
        rows.append((interval.chrom, interval.start, interval.end, int(summit), values))

    rows.sort(key=lambda r: (r[0], r[1], r[3], r[2]))

    cur_chr = ""
    cur_end = 0
    b = int(output_bin_size)
    if b <= 0:
        raise ValueError(f"Invalid output_bin_size: {output_bin_size}")

    for i, (chrom, start, end, summit, values) in enumerate(rows):
        if chrom != cur_chr:
            cur_chr = chrom
            cur_end = 0

        if cur_end < start:
            cur_end = start
        if cur_end >= end:
            continue

        next_end = end
        if i + 1 < len(rows):
            n_chrom, n_start, _, n_summit, _ = rows[i + 1]
            if n_chrom == chrom and n_start < end:
                next_end = (summit + n_summit) // 2

        next_end = min(next_end, end)
        if next_end <= cur_end:
            continue

        # Use ceil for starts so bins begin at or after cur_end/next_end boundaries.
        left_bin = max(0, (cur_end - start + b - 1) // b)
        right_bin = min(int(values.shape[0]), (next_end - start + b - 1) // b)
        if right_bin <= left_bin:
            continue

        for bi in range(left_bin, right_bin):
            bin_start = start + bi * b
            if bin_start >= next_end:
                break
            bin_end = min(bin_start + b, next_end)
            if bin_end <= bin_start:
                continue
            yield (chrom, bin_start, bin_end, float(values[bi]))

        cur_end = next_end


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
    region_generation: str = "sliding_window",
    overlap_resolution: str = "average",
    selected_chrom: str | None = None,
    peaks_path: str | Path | None = None,
    nonpeaks_path: str | Path | None = None,
) -> None:
    """
    Generates predictions and writes to a BigWig file.

    This function supports two region generation strategies:

    1. `sliding_window` (default): process allowed chromosomes with a SlidingWindowSampler.
    2. `peaks_nonpeaks`: load chrombpnet-style regions (peaks plus optional non-peaks)
       for one selected chromosome.

    Overlap resolution between predicted regions can be either:
    - `average` (default): average overlapping bins.
    - `midpoint_between_summits`: split overlaps at midpoint between adjacent region summits
      (chrombpnet-style stitching).

    Args:
        output_path: Path to the output BigWig file.
        dataset: Initialized CerberusDataset (provides data configuration and extractors).
        model_ensemble: Initialized ModelEnsemble (provides models).
        stride: Stride for sliding window predictions. If None, defaults to output_len // 2.
        use_folds: Folds to use.
        aggregation: Aggregation mode.
        batch_size: Batch size for inference.
        region_generation: `sliding_window` or `peaks_nonpeaks`.
        overlap_resolution: `average` or `midpoint_between_summits`.
        selected_chrom: Optional single chromosome to process. Required for
            `peaks_nonpeaks` mode unless dataset allowed_chroms has length 1.
        peaks_path: Peaks BED/narrowPeak path for `peaks_nonpeaks` mode.
        nonpeaks_path: Optional non-peaks BED/narrowPeak path for `peaks_nonpeaks` mode.
    """
    genome_config = dataset.genome_config
    data_config = dataset.data_config

    _validate_mode("region_generation", region_generation, _VALID_REGION_GENERATION_MODES)
    _validate_mode("overlap_resolution", overlap_resolution, _VALID_OVERLAP_RESOLUTION_MODES)

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

        if region_generation == "sliding_window":
            chroms_to_process = [selected_chrom] if selected_chrom is not None else list(allowed_chroms)

            for chrom in chroms_to_process:
                if chrom is None or chrom not in chrom_sizes:
                    continue

                # Create specific sampler for this chromosome to generate windows.
                # We use input_len as padded_size because predict_intervals expects inputs of that length.
                sampler = SlidingWindowSampler(
                    chrom_sizes={chrom: chrom_sizes[chrom]},
                    padded_size=input_len,
                    stride=stride,  # type: ignore (checked above)
                    exclude_intervals=exclude_intervals,
                    folds=[],  # No folds needed for prediction generation
                )

                for island in _split_sorted_intervals_into_islands(sampler, output_len):
                    yield from _process_island(
                        island,
                        dataset,
                        model_ensemble,
                        use_folds,
                        aggregation,
                        batch_size,
                        overlap_resolution=overlap_resolution,
                    )
            return

        # region_generation == "peaks_nonpeaks"
        if peaks_path is None:
            raise ValueError(
                "region_generation='peaks_nonpeaks' requires peaks_path to be provided."
            )

        if selected_chrom is None:
            if len(allowed_chroms) == 1:
                selected = allowed_chroms[0]
            else:
                raise ValueError(
                    "region_generation='peaks_nonpeaks' requires selected_chrom when "
                    "dataset allowed_chroms contains multiple chromosomes."
                )
        else:
            selected = selected_chrom

        if selected not in allowed_chroms:
            raise ValueError(
                f"selected_chrom {selected!r} is not present in dataset allowed_chroms."
            )

        intervals = _load_peak_nonpeak_intervals_for_chrom(
            peaks_path=peaks_path,
            nonpeaks_path=nonpeaks_path,
            chrom=selected,
            chrom_sizes=chrom_sizes,
            padded_size=input_len,
            exclude_intervals=exclude_intervals,
        )
        if not intervals:
            logger.warning(
                "No intervals found for chrombpnet-style region generation: chrom=%s, peaks=%s, nonpeaks=%s",
                selected,
                str(peaks_path),
                str(nonpeaks_path) if nonpeaks_path is not None else "None",
            )
            return

        for island in _split_sorted_intervals_into_islands(intervals, output_len):
            yield from _process_island(
                island,
                dataset,
                model_ensemble,
                use_folds,
                aggregation,
                batch_size,
                overlap_resolution=overlap_resolution,
            )

    logger.info(
        "Writing BigWig to %s (region_generation=%s, overlap_resolution=%s)...",
        output_path,
        region_generation,
        overlap_resolution,
    )
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
    overlap_resolution: str = "average",
) -> Iterable[tuple[str, int, int, float]]:
    """
    Runs prediction on a contiguous island of intervals and yields values.
    """
    _validate_mode("overlap_resolution", overlap_resolution, _VALID_OVERLAP_RESOLUTION_MODES)

    output_len = dataset.data_config["output_len"]
    output_bin_size = dataset.data_config["output_bin_size"]

    tracks: list[torch.Tensor] = []
    output_intervals: list[Interval] = []
    summit_positions: list[int] = []

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
                summit_positions.append((interval.start + interval.end) // 2)
        elif aggregation == "interval+model":
            if track_tensor.ndim != 2:
                raise ValueError(
                    f"Expected merged track tensor with shape (C, L), got {tuple(track_tensor.shape)}"
                )
            if batch_output.out_interval is None:
                raise ValueError("ModelOutput.out_interval is None, but expected merged interval.")
            tracks.append(track_tensor)
            output_intervals.append(batch_output.out_interval)
            summit_positions.append((batch_output.out_interval.start + batch_output.out_interval.end) // 2)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation} (supported: 'model', 'interval+model')")

    if not tracks:
        return

    if overlap_resolution == "midpoint_between_summits":
        yield from _stitch_tracks_midpoint_between_summits(
            tracks=tracks,
            intervals=output_intervals,
            summit_positions=summit_positions,
            output_bin_size=output_bin_size,
        )
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
    # strategy (sum/mean). For now we always take the first channel.
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
