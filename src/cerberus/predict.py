import torch
import torch.nn as nn
from typing import List, Tuple, Union, Any, cast, Dict, Iterable
import numpy as np
from collections import defaultdict

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_manager import ModelManager
from cerberus.config import PredictConfig


def predict_interval(
    interval: Interval,
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Any, Interval]:
    """
    Wrapper around predict_intervals for single interval.
    """
    return predict_intervals(
        [interval], dataset, model_manager, predict_config, device
    )


def predict_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Any:
    """
    Predicts and aggregates outputs for multiple intervals in a single batch.

    Args:
        interval: The Interval object to predict on. Must be of length dataset.data_config['input_len'].
        dataset: CerberusDataset instance for data retrieval.
        model_manager: ModelManager instance for retrieving models.
        predict_config: PredictConfig instance.
        device: Device to run models on.

    Returns:
        A tuple containing:
        - The aggregated output from the models. Structure matches the model output
          (e.g., Tensor, or Tuple[Tensor, ...]).
        - The merged genomic interval corresponding to the aggregated output.
    """
    interval_list = list(intervals)
    if not interval_list:
        return None

    input_len = dataset.data_config["input_len"]
    output_len = dataset.data_config["output_len"]

    # 1. Prepare Batch Data
    inputs_list = []
    for i, interval in enumerate(interval_list):
        if len(interval) != input_len:
            raise ValueError(
                f"Interval {interval} (index {i}) has length {len(interval)}, "
                f"expected {input_len}."
            )
        data = dataset.get_interval(interval)
        inputs_list.append(data["inputs"])

    # Stack into (Batch, Channels, Length)
    batch_inputs = torch.stack(inputs_list).to(device)

    # 2. Retrieve Models (using first interval, assuming all share models)
    first_interval = interval_list[0]
    models = model_manager.get_models(
        first_interval.chrom,
        first_interval.start,
        first_interval.end,
        use_folds=predict_config["use_folds"],
    )

    if not models:
        raise RuntimeError(
            f"No valid models found for interval {first_interval} "
            f"with use_folds={predict_config['use_folds']}"
        )

    # 3. Run Models on Batch
    batch_outputs = []
    with torch.no_grad():
        for model in models:
            out = model(batch_inputs)
            batch_outputs.append(out)

    # 4. Aggregate Ensemble Outputs (result is still batched)
    # This aggregates predictions across the different models/folds (e.g. taking the mean).
    # It preserves the batch dimension (N intervals) and does NOT combine overlapping intervals yet.
    aggregation = predict_config["aggregation"]
    aggregated_batch = _aggregate_ensemble_outputs(batch_outputs, aggregation)

    # 5. Unbatch and Pair with Output Intervals
    # Splits the batched model output (which may be a Tensor or a recursive structure)
    # into a list of individual interval outputs. Each output is then paired with its 
    # corresponding genomic interval (center-cropped to output_len).
    results = []
    unbatched_outputs = _unbatch_output(aggregated_batch, len(interval_list))
    
    for interval, output in zip(interval_list, unbatched_outputs):
        output_interval = interval.center(output_len)
        results.append((output, output_interval))

    # 6. Aggregate Overlaps
    return _aggregate_overlapping_output_intervals(
        results,
        dataset.data_config["output_bin_size"],
        dataset.data_config["output_len"],
    )


def _aggregate_ensemble_outputs(outputs: List[Any], method: str) -> Any:
    """
    Aggregates a list of model outputs.
    Outputs can be Tensors or recursive structures (tuples/lists) of Tensors.
    All outputs must have the same structure.
    """
    if not outputs:
        return None
        
    first = outputs[0]
    
    if isinstance(first, torch.Tensor):
        # Base case: List of Tensors
        # Stack: (N_Models, Batch, ...)
        stacked = torch.stack(cast(List[torch.Tensor], outputs))
        
        if method == "mean":
            # mean over N_Models dimension (dim 0)
            return torch.mean(stacked, dim=0)
        elif method == "median":
            # median over N_Models dimension (dim 0)
            return torch.median(stacked, dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    elif isinstance(first, (tuple, list)):
        # Recursive case: List of Tuples/Lists
        # We want to aggregate element-wise.
        # outputs = [(t1_a, t2_a), (t1_b, t2_b), ...]
        # result = (aggregate([t1_a, t1_b]), aggregate([t2_a, t2_b]))
        
        n_elements = len(first)
        aggregated_elements = []
        
        for i in range(n_elements):
            # Extract the i-th element from each output
            ith_elements = [out[i] for out in outputs]
            aggregated_elements.append(_aggregate_ensemble_outputs(ith_elements, method))
            
        return type(first)(aggregated_elements)
        
    else:
        raise TypeError(f"Unsupported output type for aggregation: {type(first)}")


def _unbatch_output(batched_output: Any, batch_size: int) -> List[Any]:
    """
    Splits a batched output (Tensor or recursive Tuple) into a list of individual samples.
    """
    if isinstance(batched_output, torch.Tensor):
        # Check batch size consistency
        if batched_output.shape[0] != batch_size:
             raise ValueError(f"Batch size mismatch: expected {batch_size}, got {batched_output.shape[0]}")
        # Unbind along dimension 0
        return list(torch.unbind(batched_output, dim=0))
        
    elif isinstance(batched_output, (tuple, list)):
        # Recursive unbatching
        # batched_output is (BatchTensor1, BatchTensor2) -> [(Tensor1_0, Tensor2_0), ...]
        
        n_components = len(batched_output)
        unbatched_components = []
        for i in range(n_components):
            unbatched_components.append(_unbatch_output(batched_output[i], batch_size))
            
        # Zip components together
        # unbatched_components is [[T1_0, T1_1], [T2_0, T2_1]]
        # We want [(T1_0, T2_0), (T1_1, T2_1)]
        
        return [
            type(batched_output)(sample_comps) 
            for sample_comps in zip(*unbatched_components)
        ]
    else:
        raise TypeError(f"Unsupported output type for unbatching: {type(batched_output)}")


def _aggregate_overlapping_output_intervals(
    results: List[Tuple[Any, Interval]],
    output_bin_size: int,
    output_len: int,
) -> Tuple[Any, Interval]:
    """
    Aggregates overlapping predictions into a single merged output.
    
    Assumptions:
    - Model outputs are Tensors or recursive Tuples/Lists of Tensors.
    - Tensors with spatial dimension matching `output_len` (via `bin_size`) are treated as Profiles (summed/averaged per bin).
    - Other Tensors are treated as Scalars (global values for the interval) and are broadcasted to the interval length.
    - All intervals are from the same chromosome.
    """
    if not results:
        raise ValueError("No results to aggregate")

    # Unzip
    outputs = [r[0] for r in results]
    intervals = [r[1] for r in results]
    
    # Compute merged interval once (same for all components)
    chrom = intervals[0].chrom
    strand = intervals[0].strand
    min_start = min(i.start for i in intervals)
    max_end = max(i.end for i in intervals)
    merged_interval = Interval(chrom, min_start, max_end, strand)

    aggregated_values = _aggregate_values_recursive(
        outputs, intervals, merged_interval, output_bin_size, output_len
    )
    
    return aggregated_values, merged_interval


def _aggregate_values_recursive(
    outputs: List[Any],
    intervals: List[Interval],
    merged_interval: Interval,
    output_bin_size: int,
    output_len: int,
) -> Any:
    """
    Recursively aggregates values into the merged interval.
    """
    first_out = outputs[0]
    
    if isinstance(first_out, torch.Tensor):
        return _aggregate_tensor_track_values(
            cast(List[torch.Tensor], outputs), intervals, merged_interval, output_bin_size, output_len
        )
    elif isinstance(first_out, (tuple, list)):
        n = len(first_out)
        aggregated_components = []
        for i in range(n):
            comp_outputs = [out[i] for out in outputs]
            agg_comp = _aggregate_values_recursive(
                comp_outputs, intervals, merged_interval, output_bin_size, output_len
            )
            aggregated_components.append(agg_comp)
        return type(first_out)(aggregated_components)
    else:
        raise TypeError(f"Unsupported output type for aggregation: {type(first_out)}")


def _aggregate_tensor_track_values(
    outputs: List[torch.Tensor],
    intervals: List[Interval],
    merged_interval: Interval,
    output_bin_size: int,
    output_len: int,
) -> np.ndarray:
    """
    Aggregates a list of tensors into a single merged array.
    """
    # Merged extent
    min_start = merged_interval.start
    max_end = merged_interval.end
    
    span_bp = max_end - min_start
    n_bins = (span_bp + output_bin_size - 1) // output_bin_size
    
    # Determine channels
    # item[0] is Tensor (Batch=1, C, L) or (Batch=1, C)
    # But wait! _unbatch_output removes the Batch dim.
    # So outputs[0] is (C, L) or (C,).
    # In predict_intervals logic:
    # aggregated_batch is (Batch, C, L).
    # unbatched_outputs is [(C, L), ...].
    # So here outputs[0] is (C, L).
    
    sample_out = outputs[0]
    
    n_channels = sample_out.shape[0]
    
    # Accumulators
    accumulator = np.zeros((n_channels, n_bins), dtype=np.float32)
    counts = np.zeros((1, n_bins), dtype=np.float32)
    
    for out_tensor, interval in zip(outputs, intervals):
        # Convert to numpy
        val = out_tensor.cpu().numpy() # (C, L) or (C,)
        
        # Check dimensions
        # Profile: last dim * bin_size == output_len
        is_profile = (val.shape[-1] * output_bin_size == output_len)
        
        # Relative start bin
        rel_start_bp = interval.start - min_start
        rel_start_bin = rel_start_bp // output_bin_size
        
        if is_profile:
            # val is (C, L_bins)
            l_bins = val.shape[-1]
            accumulator[:, rel_start_bin : rel_start_bin + l_bins] += val
            counts[:, rel_start_bin : rel_start_bin + l_bins] += 1.0
        else:
            # Scalar (C,)
            # Broadcast to interval length to create a constant track over the interval
            # Interval length in bins
            int_bins = output_len // output_bin_size
            # val is (C,) -> (C, int_bins)
            val_b = np.expand_dims(val, -1)
            
            accumulator[:, rel_start_bin : rel_start_bin + int_bins] += val_b
            counts[:, rel_start_bin : rel_start_bin + int_bins] += 1.0

    # Average
    counts = np.maximum(counts, 1.0)
    final_values = accumulator / counts
    
    return final_values
