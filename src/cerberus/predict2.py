import torch
import numpy as np
from typing import Iterator
from torch import nn

from cerberus.config import PredictConfig
from cerberus.dataset import CerberusDataset
from cerberus.model_manager import ModelManager
from cerberus.interval import Interval

def _compute_accumulators(
    interval: Interval,
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Computes accumulated predictions and counts for an interval.

    batch_size: The number of input sequences (windows) passed to the model in a single forward pass.
    """
    chrom_sizes = dataset.genome_config["chrom_sizes"]
    input_len = dataset.data_config["input_len"]
    output_len = dataset.data_config["output_len"]
    output_bin_size = dataset.data_config["output_bin_size"]
    
    stride = predict_config["stride"]
    use_folds = predict_config["use_folds"]
    aggregation = predict_config["aggregation"]

    if stride % output_bin_size != 0:
        raise ValueError(
            f"Stride ({stride}) must be a multiple of output_bin_size ({output_bin_size}) "
            "to maintain bin alignment."
        )

    if output_len % output_bin_size != 0:
        raise ValueError(
            f"Output length ({output_len}) must be a multiple of output_bin_size ({output_bin_size})."
        )
        
    torch_device = torch.device(device)
    
    chrom = interval.chrom
    start = interval.start
    end = interval.end
    
    if len(interval) % output_bin_size != 0:
        raise ValueError(
            f"Interval length ({len(interval)}) must be a multiple of output_bin_size ({output_bin_size})."
        )
    chrom_len = chrom_sizes[chrom]
    
    # Calculate offset to center the output window within the input window
    # consistently with TargetCrop transform.
    offset = (input_len - output_len) // 2
    
    # Number of output bins in the interval
    n_bins = (end - start) // output_bin_size

    # accumulated_predictions: Stores the sum of model predictions for each bin in the interval.
    # Since we might have overlapping windows (due to stride) or multiple folds,
    # we accumulate values here and average them later.
    accumulated_predictions = np.zeros(n_bins, dtype=np.float32)
    
    # prediction_counts: Tracks how many predictions contributed to each bin.
    # This is used for averaging the accumulated predictions.
    prediction_counts = np.zeros(n_bins, dtype=np.float32)

    models = model_manager.get_models(chrom, start, end, use_folds=use_folds)
    if not models:
        return accumulated_predictions, prediction_counts, output_bin_size

    input_window_start = start - offset

    # buffered_inputs: Collects model inputs to be processed in a batch.
    buffered_inputs = []
    
    # buffered_start_bins: Collects corresponding bin indices for the inputs.
    buffered_start_bins = []

    while input_window_start + offset < end:
        out_start = input_window_start + offset
        
        if (input_window_start >= 0 and input_window_start + input_len <= chrom_len):
            
            # Create an input interval expanded by offset to ensure output coverage
            input_interval = Interval(chrom, input_window_start, input_window_start + input_len)
            data = dataset.get_interval(input_interval)
            inputs = data["inputs"]
            
            buffered_inputs.append(inputs)

            # Calculate the start position of the current output window relative to the start of the full interval.
            rel_start_bp = out_start - start
            
            # Convert the relative start position to the corresponding bin index in the accumulator arrays.
            rel_start_bin = rel_start_bp // output_bin_size

            buffered_start_bins.append(rel_start_bin)

            if len(buffered_inputs) >= batch_size:
                # _process_batch modifies accumulated_predictions and prediction_counts in-place
                _process_batch(
                    models,
                    buffered_inputs,
                    buffered_start_bins,
                    accumulated_predictions,
                    prediction_counts,
                    torch_device,
                    output_bin_size,
                    n_bins,
                    output_len,
                    aggregation,
                )
                buffered_inputs = []
                buffered_start_bins = []

        input_window_start += stride

    # Process any remaining buffered inputs
    if buffered_inputs:
        # _process_batch modifies accumulated_predictions and prediction_counts in-place
        _process_batch(
            models,
            buffered_inputs,
            buffered_start_bins,
            accumulated_predictions,
            prediction_counts,
            torch_device,
            output_bin_size,
            n_bins,
            output_len,
            aggregation,
        )
        
    return accumulated_predictions, prediction_counts, output_bin_size

def predict_interval(
    interval: Interval,
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Iterator[tuple[str, int, int, float]]:
    """
    Generates predictions for a single interval, yielding binned values.
    
    This function yields a tuple (chrom, start, end, value) for each bin in the interval.
    The value represents the prediction for that specific bin.
    """
    accumulated_predictions, prediction_counts, output_bin_size = _compute_accumulators(
        interval, dataset, model_manager, predict_config, batch_size, device
    )
    
    chrom = interval.chrom
    start = interval.start
    n_bins = len(accumulated_predictions)

    for i in range(n_bins):
        if prediction_counts[i] == 0:
            continue

        bin_start = start + i * output_bin_size
        bin_end = bin_start + output_bin_size

        val = float(accumulated_predictions[i] / prediction_counts[i])
        yield (chrom, bin_start, bin_end, val)

def predict_single_interval_matrix(
    interval: Interval,
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resolution: str = "binned",
) -> np.ndarray:
    """
    Generates predictions for a single interval and returns a numpy array.
    
    Args:
        resolution: "binned" (returns array of shape [n_bins]) or 
                    "bp" (returns array of shape [length]).
    """
    accumulated_predictions, prediction_counts, output_bin_size = _compute_accumulators(
        interval, dataset, model_manager, predict_config, batch_size, device
    )
    
    # Avoid division by zero
    prediction_counts = np.maximum(prediction_counts, 1.0)
    vals = accumulated_predictions / prediction_counts
    
    if resolution == "binned":
        return vals
    elif resolution == "bp":
        # Upsample by repeating values
        return np.repeat(vals, output_bin_size)
    else:
        raise ValueError(f"Unknown resolution: {resolution}. Supported: 'binned', 'bp'.")

def _process_batch(
    models: list[nn.Module],
    inputs: list[torch.Tensor],
    start_bins: list[int],
    accum: np.ndarray,
    counts: np.ndarray,
    device: torch.device,
    model_bin_size: int,
    buffer_n_bins: int,
    output_len: int,
    aggregation: str,
):
    x = torch.stack(inputs).to(device)

    all_preds = []
    with torch.no_grad():
        for model in models:
            pred = model(x)
            if pred.dim() == 3:
                pred = pred[:, 0, :]
            all_preds.append(pred)

    if not all_preds:
        return

    stacked = torch.stack(all_preds) 
    
    if aggregation == "mean":
        avg_pred = torch.mean(stacked, dim=0).cpu().numpy()
    elif aggregation == "median":
        avg_pred = torch.median(stacked, dim=0).values.cpu().numpy()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    bins_per_window = output_len // model_bin_size

    for i, start_bin in enumerate(start_bins):
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
