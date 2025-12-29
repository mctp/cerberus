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
from cerberus.sequence import SequenceExtractor
from cerberus.core import Interval


def parse_intervals(
    intervals: list[str], interval_paths: list[Path], genome_config: GenomeConfig
) -> list[tuple[str, int, int]]:
    """
    Parses intervals from a list of strings and paths to BED files.

    Args:
        intervals: List of strings (e.g. ["chr1", "chr2:1000-2000"]).
        interval_paths: List of paths to BED files.
        genome_config: Genome configuration containing chromosome sizes.

    Returns:
        List of (chrom, start, end) tuples.
    """
    parsed = []
    chrom_sizes = genome_config["chrom_sizes"]

    # If both lists are empty, default to whole genome
    if not intervals and not interval_paths:
        for chrom in genome_config["allowed_chroms"]:
            if chrom in chrom_sizes:
                parsed.append((chrom, 0, chrom_sizes[chrom]))
        return parsed

    # Process intervals from strings
    for item in intervals:
        if ":" in item:
            try:
                chrom, coords = item.split(":")
                start, end = map(int, coords.split("-"))
                parsed.append((chrom, start, end))
            except ValueError:
                raise ValueError(
                    f"Invalid interval format: {item}. Expected 'chr:start-end'."
                )
        else:
            chrom = item
            if chrom not in chrom_sizes:
                raise ValueError(f"Chromosome {chrom} not found in genome config.")
            parsed.append((chrom, 0, chrom_sizes[chrom]))

    # Process intervals from files
    for p in interval_paths:
        with open(p) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    parsed.append((parts[0], int(parts[1]), int(parts[2])))

    return parsed


def merge_intervals(
    intervals: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """
    Sorts and merges overlapping or adjacent intervals.
    """
    if not intervals:
        return []

    # Sort by chrom, then start
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

    merged = []
    current_chrom, current_start, current_end = sorted_intervals[0]

    for i in range(1, len(sorted_intervals)):
        chrom, start, end = sorted_intervals[i]

        if chrom == current_chrom and start <= current_end:
            # Overlap or adjacent, merge
            current_end = max(current_end, end)
        else:
            merged.append((current_chrom, current_start, current_end))
            current_chrom, current_start, current_end = chrom, start, end

    merged.append((current_chrom, current_start, current_end))
    return merged


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

    if stride is None:
        stride = output_len

    offset = (input_len - output_len) // 2

    # 4. Iterate Intervals
    for chrom, start, end in merged_intervals:
        print(f"Processing {chrom}:{start}-{end}...")

        # Align buffer to model_bin_size
        buffer_start = (start // model_bin_size) * model_bin_size
        buffer_end = (
            (end + model_bin_size - 1) // model_bin_size
        ) * model_bin_size
        n_bins = (buffer_end - buffer_start) // model_bin_size

        accum = np.zeros(n_bins, dtype=np.float32)
        counts = np.zeros(n_bins, dtype=np.float32)

        # Get models
        models = model_manager.get_models(chrom, start, end, use_folds=use_folds)
        if not models:
            print(f"No models found for {chrom}:{start}-{end}. Skipping.")
            continue

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
                chrom_len = genome_config["chrom_sizes"][chrom]
                seq_start = max(0, curr_in_start)
                seq_end = min(chrom_len, curr_in_start + input_len)

                pad_left = max(0, -curr_in_start)
                pad_right = max(0, (curr_in_start + input_len) - chrom_len)

                if seq_start < seq_end:
                    seq_tensor = extractor.extract(
                        Interval(chrom, seq_start, seq_end)
                    )
                    if pad_left > 0 or pad_right > 0:
                        seq_tensor = torch.nn.functional.pad(
                            seq_tensor, (pad_left, pad_right)
                        )
                else:
                    seq_tensor = torch.zeros(
                        (4, input_len), dtype=torch.float32
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
                        torch_device,
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
                torch_device,
                model_bin_size,
                n_bins,
                output_len,
                aggregation,
            )

        # Normalize and Yield
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
    Runs inference and updates accum/counts buffers.
    """
    x = torch.stack(inputs).to(device)

    all_preds = []
    with torch.no_grad():
        for model in models:
            pred = model(x)
            if pred.dim() == 3:
                # (B, C, L), take channel 0
                pred = pred[:, 0, :]
            all_preds.append(pred)

    if not all_preds:
        return

    # Aggregate
    stacked = torch.stack(all_preds) # (M, B, L)
    
    if aggregation == "mean":
        avg_pred = torch.mean(stacked, dim=0).cpu().numpy()
    elif aggregation == "median":
        # torch.median returns (values, indices)
        avg_pred = torch.median(stacked, dim=0).values.cpu().numpy()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    bins_per_window = output_len // model_bin_size

    for i, start_bin in enumerate(coords):
        p_start = 0
        p_end = bins_per_window

        b_start = start_bin
        b_end = start_bin + bins_per_window

        if b_start < 0:
            p_start = -b_start
            b_start = 0

        if b_end > buffer_n_bins:
            diff = b_end - buffer_n_bins
            p_end -= diff
            b_end = buffer_n_bins

        if p_start < p_end and b_start < b_end:
            accum[b_start:b_end] += avg_pred[i, p_start:p_end]
            counts[b_start:b_end] += 1.0
