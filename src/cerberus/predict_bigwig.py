import torch
from collections.abc import Iterable
import pybigtools
from pathlib import Path
import dataclasses

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.config import PredictConfig
from cerberus.samplers import SlidingWindowSampler


def predict_to_bigwig(
    output_path: str | Path,
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    predict_config: PredictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        predict_config: Prediction configuration.
        device: Device to run inference on.
        batch_size: Batch size for inference.
    """
    genome_config = dataset.genome_config
    data_config = dataset.data_config
    
    # Prepare generator for pybigtools
    def stream_generator() -> Iterable[tuple[str, int, int, float]]:
        chrom_sizes = genome_config["chrom_sizes"]
        allowed_chroms = genome_config["allowed_chroms"]
        input_len = data_config["input_len"]
        output_len = data_config["output_len"]
        stride = predict_config["stride"]
        exclude_intervals = dataset.exclude_intervals

        for chrom in allowed_chroms:
            if chrom not in chrom_sizes:
                continue

            # Create specific sampler for this chromosome to generate windows
            # We use input_len as padded_size because predict_intervals expects inputs of that length
            sampler = SlidingWindowSampler(
                chrom_sizes={chrom: chrom_sizes[chrom]},
                padded_size=input_len,
                stride=stride,
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
                        predict_config,
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
                    predict_config,
                    batch_size,
                )

    print(f"Writing BigWig to {output_path}...")
    # Using pybigtools to write the stream
    bw = pybigtools.open(str(output_path), "w") # type: ignore
    bw.write(genome_config["chrom_sizes"], stream_generator())


def _process_island(
    island_intervals: list[Interval],
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    predict_config: PredictConfig,
    batch_size: int,
) -> Iterable[tuple[str, int, int, float]]:
    """
    Runs prediction on a contiguous island of intervals and yields values.
    """
    # predict_intervals aggregates the island into one result
    output = model_ensemble.predict_intervals(
        island_intervals, dataset, predict_config, batch_size
    )
    aggregated_output = dataclasses.asdict(output)
    merged_interval = output.out_interval
    if merged_interval is None:
        raise ValueError("ModelOutput.out_interval is None, but expected a merged interval.")

    # aggregated_output is a dict[str, np.ndarray]
    # We attempt to retrieve the primary profile track
    if "log_rates" in aggregated_output:
        track_data = aggregated_output["log_rates"]
    elif "logits" in aggregated_output:
        track_data = aggregated_output["logits"]
    else:
        raise ValueError(
            f"Could not identify a profile track in output keys: {list(aggregated_output.keys())}"
        )
    
    values = track_data[0]  # Shape: (Bins,)

    chrom = merged_interval.chrom
    start = merged_interval.start
    output_bin_size = dataset.data_config["output_bin_size"]

    for i, val in enumerate(values):
        bin_start = start + i * output_bin_size
        bin_end = bin_start + output_bin_size
        
        yield (chrom, bin_start, bin_end, float(val))
