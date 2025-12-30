import re
import math
from pathlib import Path
from typing import Iterator

import torch
import numpy as np
import pybigtools
from torch import nn

from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    PredictConfig,
)
from cerberus.model_manager import ModelManager
from cerberus.sequence import SequenceExtractor, BaseSequenceExtractor
from cerberus.interval import Interval, parse_intervals, merge_intervals
from cerberus.dataset import CerberusDataset


def _predict_to(
    checkpoint_path: str | Path,
    genome_config: GenomeConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    predict_config: PredictConfig,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Iterator[tuple[str, int, int, float]]:
    """
    Core prediction generator that yields (chrom, start, end, value) tuples.
    Can be used by different output formats (BigWig, etc.).
    """
    # Extract from predict_config
    intervals_list = predict_config["intervals"]
    interval_paths = predict_config["intervals_paths"]
    stride = predict_config["stride"]
    use_folds = predict_config["use_folds"]
    aggregation = predict_config["aggregation"]

    torch_device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    # 1. Prepare Intervals
    parsed_intervals = parse_intervals(intervals_list, interval_paths, genome_config)
    merged_intervals = merge_intervals(parsed_intervals)

    # 2. Prepare Model Manager
    model_manager = ModelManager(
        checkpoint_path, model_config, data_config, train_config, genome_config, torch_device
    )

    # 3. Prepare Sequence Extractor
    fasta_path = genome_config["fasta_path"]
    extractor = SequenceExtractor(fasta_path)

    input_len = data_config["input_len"]
    output_len = data_config["output_len"]
    model_bin_size = data_config["output_bin_size"]

    # 4. Iterate Intervals
    chrom_sizes = genome_config["chrom_sizes"]
    for interval in merged_intervals:
        print(f"Processing {interval.chrom}:{interval.start}-{interval.end}...")
        yield from _predict_single_interval(
            interval,
            model_manager,
            extractor,
            chrom_sizes,
            input_len,
            output_len,
            model_bin_size,
            stride,
            use_folds,
            aggregation,
            batch_size,
            torch_device,
        )


def predict_dataset(
    dataset: CerberusDataset,
    model_manager: ModelManager,
    use_folds: list[str] = ["test"],
    aggregation: str = "mean",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    stride: int | None = None,
) -> Iterator[tuple[str, int, int, float]]:
    """
    Predicts values for intervals defined in the dataset.
    """
    torch_device = torch.device(device)

    # Extract configs
    chrom_sizes = dataset.genome_config["chrom_sizes"]
    input_len = dataset.data_config["input_len"]
    output_len = dataset.data_config["output_len"]
    model_bin_size = dataset.data_config["output_bin_size"]

    extractor = dataset.sequence_extractor
    if extractor is None:
        raise ValueError("Dataset must have a sequence extractor for prediction.")

    # Determine stride
    if stride is not None:
        actual_stride = stride
    elif dataset.sampler_config["sampler_type"] == "sliding_window":
        actual_stride = dataset.sampler_config["sampler_args"]["stride"]
    else:
        actual_stride = output_len

    # Iterate sampler
    for interval in dataset.sampler:
        yield from _predict_single_interval(
            interval,
            model_manager,
            extractor,
            chrom_sizes,
            input_len,
            output_len,
            model_bin_size,
            int(actual_stride),
            use_folds,
            aggregation,
            batch_size,
            torch_device,
        )


def _predict_single_interval(
    interval: Interval,
    model_manager: ModelManager,
    extractor: BaseSequenceExtractor,
    chrom_sizes: dict[str, int],
    input_len: int,
    output_len: int,
    model_bin_size: int,
    stride: int,
    use_folds: list[str],
    aggregation: str,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[str, int, int, float]]:
    chrom = interval.chrom
    start = interval.start
    end = interval.end

    offset = (input_len - output_len) // 2

    # Align buffer to model_bin_size
    buffer_start, buffer_end, n_bins = _get_buffer_params(start, end, model_bin_size)

    accum = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.float32)

    # Get models
    models = model_manager.get_models(chrom, start, end, use_folds=use_folds)
    if not models:
        print(f"No models found for {chrom}:{start}-{end}. Skipping.")
        return

    # Sliding Window Loop
    curr_in_start = buffer_start - offset

    # Batched processing
    batch_inputs = []
    batch_coords = []  # (out_start_bin_idx) relative to buffer

    while curr_in_start + offset < buffer_end:
        out_start = curr_in_start + offset
        out_end = out_start + output_len

        # Check overlap with buffer [buffer_start, buffer_end)
        if out_end > buffer_start and out_start < buffer_end:
            # Prepare input
            chrom_len = chrom_sizes[chrom]
            seq_tensor = _prepare_sequence_input(
                chrom, curr_in_start, input_len, chrom_len, extractor
            )

            batch_inputs.append(seq_tensor)

            rel_start_bp = out_start - buffer_start
            rel_start_bin = rel_start_bp // model_bin_size

            batch_coords.append(rel_start_bin)

            if len(batch_inputs) >= batch_size:
                _process_batch(
                    models,
                    batch_inputs,
                    batch_coords,
                    accum,
                    counts,
                    device,
                    model_bin_size,
                    n_bins,
                    output_len,
                    aggregation,
                )
                batch_inputs = []
                batch_coords = []

        curr_in_start += stride

    # Process remaining batch
    if batch_inputs:
        _process_batch(
            models,
            batch_inputs,
            batch_coords,
            accum,
            counts,
            device,
            model_bin_size,
            n_bins,
            output_len,
            aggregation,
        )

    # Normalize and Yield
    yield from _yield_predictions(
        chrom, start, end, accum, counts, buffer_start, model_bin_size, n_bins
    )


def _get_buffer_params(
    start: int, end: int, model_bin_size: int
) -> tuple[int, int, int]:
    """Calculates buffer alignment parameters."""
    buffer_start = (start // model_bin_size) * model_bin_size
    buffer_end = ((end + model_bin_size - 1) // model_bin_size) * model_bin_size
    n_bins = (buffer_end - buffer_start) // model_bin_size
    return buffer_start, buffer_end, n_bins


def _prepare_sequence_input(
    chrom: str,
    curr_in_start: int,
    input_len: int,
    chrom_len: int,
    extractor: BaseSequenceExtractor,
) -> torch.Tensor:
    """Extracts and pads sequence input."""
    seq_start = max(0, curr_in_start)
    seq_end = min(chrom_len, curr_in_start + input_len)

    pad_left = max(0, -curr_in_start)
    pad_right = max(0, (curr_in_start + input_len) - chrom_len)

    if seq_start < seq_end:
        seq_tensor = extractor.extract(Interval(chrom, seq_start, seq_end))
        if pad_left > 0 or pad_right > 0:
            seq_tensor = torch.nn.functional.pad(seq_tensor, (pad_left, pad_right))
    else:
        seq_tensor = torch.zeros((4, input_len), dtype=torch.float32)

    return seq_tensor


def _yield_predictions(
    chrom: str,
    start: int,
    end: int,
    accum: np.ndarray,
    counts: np.ndarray,
    buffer_start: int,
    model_bin_size: int,
    n_bins: int,
) -> Iterator[tuple[str, int, int, float]]:
    """Normalizes accumulators and yields clipped predictions."""
    counts = np.maximum(counts, 1.0)
    final_vals = accum / counts

    for i in range(n_bins):
        bin_start = buffer_start + i * model_bin_size
        bin_end = bin_start + model_bin_size

        # Clip to requested [start, end)
        w_start = max(start, bin_start)
        w_end = min(end, bin_end)

        if w_start < w_end:
            val = float(final_vals[i])
            yield (chrom, w_start, w_end, val)


def predict_to_bigwig(
    output_path: str,
    checkpoint_path: str | Path,
    genome_config: GenomeConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    predict_config: PredictConfig,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generates predictions and writes them to a BigWig file.

    Args:
        output_path: Path to output BigWig file.
        checkpoint_path: Path to model checkpoint file or directory containing fold subdirectories.
        genome_config: Genome configuration.
        data_config: Data configuration.
        model_config: Model configuration.
        train_config: Training configuration.
        predict_config: Prediction configuration.
        batch_size: Batch size for inference.
        device: Device to run inference on.
    """
    generator = _predict_to(
        checkpoint_path=checkpoint_path,
        genome_config=genome_config,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        predict_config=predict_config,
        batch_size=batch_size,
        device=device,
    )

    print(f"Opening BigWig writer at {output_path}...")
    bw = pybigtools.open(output_path, "w") # type: ignore
    bw.write(genome_config["chrom_sizes"], generator)


def _process_batch(
    models: list[nn.Module],
    inputs: list[torch.Tensor],
    coords: list[int],
    accum: np.ndarray,
    counts: np.ndarray,
    device: torch.device,
    model_bin_size: int,
    buffer_n_bins: int,
    output_len: int,
    aggregation: str,
):
    """
    Runs inference on a batch of sliding windows and updates accumulation buffers.

    This function handles two levels of processing:
    1. Batching: Processes multiple sliding windows (inputs) in parallel.
    2. Aggregation: Runs multiple models (e.g., from different folds) on the same batch
       and aggregates their predictions (mean/median) to get a robust estimate.

    The aggregated predictions are then added to the `accum` buffer at positions specified by `coords`.
    """
    # 1. Batching: Stack inputs to create a batch tensor (B, C, L)
    x = torch.stack(inputs).to(device)

    all_preds = []
    with torch.no_grad():
        # Run inference for each model on the entire batch
        for model in models:
            pred = model(x)
            if pred.dim() == 3:
                # (B, C, L), take channel 0
                pred = pred[:, 0, :]
            all_preds.append(pred)

    if not all_preds:
        return

    # 2. Aggregation: Stack predictions from all models (M, B, L) and aggregate across models (dim 0)
    stacked = torch.stack(all_preds) 
    
    if aggregation == "mean":
        avg_pred = torch.mean(stacked, dim=0).cpu().numpy()
    elif aggregation == "median":
        # torch.median returns (values, indices)
        avg_pred = torch.median(stacked, dim=0).values.cpu().numpy()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    bins_per_window = output_len // model_bin_size

    for i, start_bin in enumerate(coords):
        # coord gives the starting bin index in the accumulation buffer
        # We need to map the prediction window to the accumulation buffer
        
        # Source indices (prediction window)
        p_start = 0
        p_end = bins_per_window

        # Destination indices (accumulation buffer)
        b_start = start_bin
        b_end = start_bin + bins_per_window

        # Handle left clipping (if window starts before buffer start)
        if b_start < 0:
            p_start = -b_start
            b_start = 0

        # Handle right clipping (if window ends after buffer end)
        if b_end > buffer_n_bins:
            diff = b_end - buffer_n_bins
            p_end -= diff
            b_end = buffer_n_bins

        # Add prediction to accumulator if there is a valid overlap
        if p_start < p_end and b_start < b_end:
            accum[b_start:b_end] += avg_pred[i, p_start:p_end]
            counts[b_start:b_end] += 1.0
