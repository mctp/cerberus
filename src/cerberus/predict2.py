import torch
import numpy as np
from typing import Iterator, Iterable, List, Tuple, Deque, Optional, Union
from collections import deque
from torch import nn

from cerberus.config import PredictConfig
from cerberus.dataset import CerberusDataset
from cerberus.model_manager import ModelManager
from cerberus.interval import Interval

class PredictionAccumulator:
    """
    Manages the accumulation of predictions for a single interval.
    """
    def __init__(
        self,
        interval: Interval,
        output_bin_size: int,
        output_len: int,
        n_bins: int
    ):
        self.interval = interval
        self.output_bin_size = output_bin_size
        self.output_len = output_len
        self.n_bins = n_bins
        
        # storage
        self.accumulated_predictions = np.zeros(n_bins, dtype=np.float32)
        self.prediction_counts = np.zeros(n_bins, dtype=np.float32)
        
        self.is_submitted = False
        
        # Pre-calculate bins per window
        self.bins_per_window = output_len // output_bin_size

    def update(self, pred: np.ndarray, start_bin: int):
        """
        Updates the accumulator with a prediction vector starting at start_bin.
        pred shape: (bins_per_window,)
        """
        p_start = 0
        p_end = self.bins_per_window
        
        b_start = start_bin
        b_end = start_bin + self.bins_per_window
        
        if p_start < p_end and b_start < b_end:
            self.accumulated_predictions[b_start:b_end] += pred[p_start:p_end]
            self.prediction_counts[b_start:b_end] += 1.0

    def yield_results(self) -> Iterator[Tuple[str, int, int, float]]:
        chrom = self.interval.chrom
        start = self.interval.start
        
        for i in range(self.n_bins):
            if self.prediction_counts[i] == 0:
                continue

            bin_start = start + i * self.output_bin_size
            bin_end = bin_start + self.output_bin_size

            val = float(self.accumulated_predictions[i] / self.prediction_counts[i])
            yield (chrom, bin_start, bin_end, val)
            
    def get_results_matrix(self, resolution: str = "binned") -> np.ndarray:
        # Avoid division by zero
        counts = np.maximum(self.prediction_counts, 1.0)
        vals = self.accumulated_predictions / counts
        
        if resolution == "binned":
            return vals
        elif resolution == "bp":
            return np.repeat(vals, self.output_bin_size)
        else:
            raise ValueError(f"Unknown resolution: {resolution}. Supported: 'binned', 'bp'.")


def predict_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_accumulators: bool = False,
) -> Iterator[Union[Tuple[str, int, int, float], PredictionAccumulator]]:
    """
    Generates predictions for multiple intervals, using batching across intervals.
    
    Args:
        intervals: Iterable of intervals to predict.
        dataset: Dataset object.
        model_manager: Model manager.
        predict_config: Prediction configuration.
        batch_size: Batch size (number of input sequences per forward pass).
        device: Device to use.
        return_accumulators: If True, yields PredictionAccumulator objects instead of tuples.
        
    Yields:
        (chrom, start, end, value) tuples for each bin in the intervals.
        The 'value' represents the aggregated model prediction for that specific genomic bin 
        (e.g., averaged across overlapping sliding windows and/or cross-validation folds).
        
        If return_accumulators is True, yields PredictionAccumulator objects.
    """
    
    # Configuration
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

    if output_len % stride != 0:
        raise ValueError(
            f"Output length ({output_len}) must be a multiple of stride ({stride})."
        )
        
    torch_device = torch.device(device)
    offset = (input_len - output_len) // 2

    # Buffers
    buffered_inputs: List[torch.Tensor] = []
    # Metadata: (accumulator, rel_start_bin)
    buffered_metadata: List[Tuple[PredictionAccumulator, int]] = []
    
    current_models: Optional[List[nn.Module]] = None
    
    # Queue of active accumulators
    active_accumulators: Deque[PredictionAccumulator] = deque()

    for interval in intervals:
        chrom = interval.chrom
        start = interval.start
        end = interval.end
        
        if len(interval) % output_bin_size != 0:
            raise ValueError(
                f"Interval length ({len(interval)}) must be a multiple of output_bin_size ({output_bin_size})."
            )
        
        chrom_len = chrom_sizes[chrom]
        n_bins = (end - start) // output_bin_size
        
        # 1. Setup accumulator
        acc = PredictionAccumulator(interval, output_bin_size, output_len, n_bins)
        active_accumulators.append(acc)
        
        # 2. Get models for this interval
        models = model_manager.get_models(chrom, start, end, use_folds=use_folds)
        if not models:
            acc.is_submitted = True # No models, mark done
            continue
            
        # 3. Check consistency with buffered batch
        # We assume modules are cached, so we can check list equality or identity
        # Assuming get_models returns a NEW list of cached modules.
        # We check if the set of modules is the same.
        if current_models is not None and models != current_models:
            # Models changed, flush buffer
            _flush_batch(
                current_models,
                buffered_inputs,
                buffered_metadata,
                torch_device,
                output_bin_size,
                output_len,
                aggregation
            )
            buffered_inputs = []
            buffered_metadata = []
            current_models = models
        elif current_models is None:
            current_models = models
            
        # 4. Generate windows
        input_window_start = start - offset
        
        while input_window_start + offset < end:
            out_start = input_window_start + offset
            
            if (input_window_start >= 0 and input_window_start + input_len <= chrom_len):
                
                # Create input
                input_interval = Interval(chrom, input_window_start, input_window_start + input_len)
                data = dataset.get_interval(input_interval)
                inputs = data["inputs"]
                
                buffered_inputs.append(inputs)
                
                # Meta
                rel_start_bp = out_start - start
                rel_start_bin = rel_start_bp // output_bin_size
                buffered_metadata.append((acc, rel_start_bin))
                
                # Check batch size
                if len(buffered_inputs) >= batch_size:
                    _flush_batch(
                        current_models,
                        buffered_inputs,
                        buffered_metadata,
                        torch_device,
                        output_bin_size,
                        output_len,
                        aggregation
                    )
                    buffered_inputs = []
                    buffered_metadata = []
            
            input_window_start += stride
            
        acc.is_submitted = True
        
        # 5. Yield completed accumulators
        while active_accumulators:
            acc_check = active_accumulators[0]
            # An accumulator is complete if it's submitted (no more windows coming)
            # AND it's not referenced in the current buffer (all windows processed).
            is_done = acc_check.is_submitted and not any(m[0] is acc_check for m in buffered_metadata)
            
            if is_done:
                acc = active_accumulators.popleft()
                if return_accumulators:
                    yield acc
                else:
                    yield from acc.yield_results()
            else:
                break
            
    # Final flush
    if buffered_inputs and current_models:
        _flush_batch(
            current_models,
            buffered_inputs,
            buffered_metadata,
            torch_device,
            output_bin_size,
            output_len,
            aggregation
        )
        
    # Yield remaining
    while active_accumulators:
        acc = active_accumulators.popleft()
        if return_accumulators:
            yield acc
        else:
            yield from acc.yield_results()


def _flush_batch(
    models: List[nn.Module],
    inputs: List[torch.Tensor],
    metadata: List[Tuple[PredictionAccumulator, int]],
    device: torch.device,
    model_bin_size: int,
    output_len: int,
    aggregation: str,
):
    """
    Processes the batch and updates the accumulators.
    """
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

    # avg_pred shape: (batch_size, output_len_bins)
    
    for i, (acc, start_bin) in enumerate(metadata):
        acc.update(avg_pred[i], start_bin)


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
    Wrapper around the batch machinery.
    """
    gen = predict_intervals(
        [interval], 
        dataset, 
        model_manager, 
        predict_config, 
        batch_size=batch_size, 
        device=device,
        return_accumulators=True
    )
    
    # Retrieve the single accumulator
    try:
        acc = next(gen)
        if isinstance(acc, PredictionAccumulator):
            return acc.get_results_matrix(resolution)
        else:
             # Should not happen given return_accumulators=True
             raise ValueError("Expected PredictionAccumulator")
    except StopIteration:
        # No results (e.g. no models found)
        # Return empty array of correct shape
        output_bin_size = dataset.data_config["output_bin_size"]
        n_bins = (interval.end - interval.start) // output_bin_size
        if resolution == "binned":
            return np.zeros(n_bins)
        elif resolution == "bp":
            return np.zeros(n_bins * output_bin_size)
        else:
             raise ValueError(f"Unknown resolution: {resolution}")
