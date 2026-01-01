import torch
from typing import Any
from collections.abc import Iterable
import itertools
import numpy as np
import dataclasses

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_manager import ModelManager
from cerberus.config import PredictConfig


def predict_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> Any:
    """
    Predicts and aggregates outputs for multiple intervals in batches.

    Args:
        intervals: The Interval objects to predict on. Must be of length dataset.data_config['input_len'].
        dataset: CerberusDataset instance for data retrieval.
        model_manager: ModelManager instance for retrieving models.
        predict_config: PredictConfig instance.
        device: Device to run models on.
        batch_size: Number of intervals to process in each batch.

    Returns:
        A tuple containing:
        - The aggregated output from the models. Always a tuple of numpy arrays.
          Each element in the tuple corresponds to a model output head (e.g., profile, total counts).
          All outputs are returned as tracks with dimensions (Channels, Bins), where Bins corresponds
          to the merged interval length divided by dataset.data_config['output_bin_size'].
          Scalar outputs (like total counts) are broadcasted to the full length of the merged interval.
        - The merged genomic interval corresponding to the aggregated output.
    """
    iterator = iter(intervals)
    try:
        first_interval = next(iterator)
    except StopIteration:
        raise ValueError("No intervals provided for prediction.")

    # Retrieve Models (using first interval, assuming all share models)
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

    input_len = dataset.data_config["input_len"]
    output_len = dataset.data_config["output_len"]
    aggregation = predict_config["aggregation"]
    
    results = []
    
    # Chain back the first interval
    full_iterator = itertools.chain([first_interval], iterator)
    
    global_index = 0
    while True:
        # Get next batch
        batch_intervals = list(itertools.islice(full_iterator, batch_size))
        if not batch_intervals:
            break
            
        # 1. Prepare Batch Data
        inputs_list = []
        for interval in batch_intervals:
            if len(interval) != input_len:
                raise ValueError(
                    f"Interval {interval} (index {global_index}) has length {len(interval)}, "
                    f"expected {input_len}."
                )
            data = dataset.get_interval(interval)
            inputs_list.append(data["inputs"])
            global_index += 1

        # Stack into (Batch, Channels, Length)
        batch_inputs = torch.stack(inputs_list).to(device)

        # 3. Run Models on Batch
        batch_outputs = []
        with torch.no_grad():
            for model in models:
                out = model(batch_inputs)
                out = dataclasses.astuple(out)
                batch_outputs.append(out)

        # 4. Aggregate Ensemble Outputs (result is still batched)
        if not batch_outputs:
            raise RuntimeError("No model outputs generated.")

        aggregated_batch = _aggregate_ensemble_outputs(batch_outputs, aggregation)

        # 5. Unbatch and Pair with Output Intervals
        unbatched_outputs = _unbatch_output(aggregated_batch, len(batch_intervals))
        
        for interval, output in zip(batch_intervals, unbatched_outputs):
            output_interval = interval.center(output_len)
            results.append((output, output_interval))

    # 6. Aggregate Overlaps
    return _aggregate_overlapping_output_intervals(
        results,
        dataset.data_config["output_bin_size"],
        dataset.data_config["output_len"],
    )


def _aggregate_ensemble_outputs(
    outputs: list[tuple[torch.Tensor, ...]], method: str
) -> tuple[torch.Tensor, ...]:
    """
    Aggregates a list of model outputs.
    Outputs are expected to be tuples of Tensors.
    """
    if not outputs:
        raise ValueError("No outputs to aggregate.")

    n_elements = len(outputs[0])
    aggregated_elements = []
    
    for i in range(n_elements):
        # Stack: (N_Models, Batch, ...)
        stacked = torch.stack([out[i] for out in outputs])
        
        if method == "mean":
            # mean over N_Models dimension (dim 0)
            aggregated_elements.append(torch.mean(stacked, dim=0))
        elif method == "median":
            # median over N_Models dimension (dim 0)
            aggregated_elements.append(torch.median(stacked, dim=0).values)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    return tuple(aggregated_elements)


def _unbatch_output(batched_output: tuple[torch.Tensor, ...], batch_size: int) -> list[tuple[torch.Tensor, ...]]:
    """
    Splits a batched output (tuple of Tensors) into a list of individual intervals (list of tuples).
    """

    # Unbind each component
    # unbatched_components: list[list[Tensor]] -> [ [T1_0, T1_1, ...], [T2_0, T2_1, ...] ]
    unbatched_components = [list(torch.unbind(tensor, dim=0)) for tensor in batched_output]
    
    # Zip to get per-interval tuples
    # zip(*unbatched_components) -> ( (T1_0, T2_0), (T1_1, T2_1), ... )
    return list(zip(*unbatched_components))


def _aggregate_overlapping_output_intervals(
    results: list[tuple[tuple[Any, ...], Interval]],
    output_bin_size: int,
    output_len: int,
) -> tuple[tuple[np.ndarray, ...], Interval]:
    """
    Aggregates overlapping predictions into a single merged output.
    """
    if not results:
        raise ValueError("No results to aggregate")

    outputs = [r[0] for r in results] # list[tuple[Tensor, ...]]
    intervals = [r[1] for r in results]
    
    n_elements = len(outputs[0])

    aggregated_components = []    
    for i in range(n_elements):
        component_outputs = [out[i] for out in outputs]
        agg_comp, merged_interval = _aggregate_tensor_track_values(
            component_outputs, intervals, output_bin_size, output_len
        )
        aggregated_components.append(agg_comp)

    return tuple(aggregated_components), merged_interval


def _aggregate_tensor_track_values(
    outputs: list[torch.Tensor],
    intervals: list[Interval],
    output_bin_size: int,
    output_len: int,
) -> tuple[np.ndarray, Interval]:
    """
    Aggregates a list of tensors into a single merged array.
    """
    # Compute merged interval, assuming all intervals are on the same chrom and strand
    chrom = intervals[0].chrom
    strand = intervals[0].strand
    min_start = min(i.start for i in intervals)
    max_end = max(i.end for i in intervals)
    merged_interval = Interval(chrom, min_start, max_end, strand)

    if len(merged_interval) % output_bin_size != 0:
         raise ValueError(
             f"Merged interval span ({len(merged_interval)}) is not a multiple of output_bin_size ({output_bin_size})."
         )

    # Merged extent
    min_start = merged_interval.start
    max_end = merged_interval.end
    
    span_bp = max_end - min_start
    # Assumes span_bp is a multiple of output_bin_size (checked above)
    n_bins = span_bp // output_bin_size
    
    # outputs[0] is (C, L) or (C,)
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
    
    return final_values, merged_interval


def predict_output_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_manager: ModelManager,
    predict_config: PredictConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> list[tuple[Any, Interval]]:
    """
    Predicts outputs for a list of target intervals by tiling them with input intervals.

    For each target interval, this function generates the necessary input intervals to cover it,
    runs the prediction (using predict_intervals), and returns the aggregated result.

    Args:
        intervals: Iterable of target intervals.
        dataset: CerberusDataset instance containing data configuration.
        model_manager: ModelManager instance for retrieving models.
        predict_config: PredictConfig instance containing 'stride'.
        device: Device to run models on.
        batch_size: Number of intervals to process in each batch.

    Returns:
        list[tuple[Any, Interval]]: A list of tuples, each containing the aggregated output
        and the merged genomic interval for a target interval.
    """
    input_len = dataset.data_config["input_len"]
    output_len = dataset.data_config["output_len"]
    stride = predict_config["stride"]

    offset = (input_len - output_len) // 2

    results = []

    for target in intervals:
        target_input_intervals = []
        current_start = target.start

        while current_start < target.end:
            # Calculate input interval corresponding to output window at current_start
            input_start = current_start - offset
            input_end = input_start + input_len

            target_input_intervals.append(
                Interval(target.chrom, input_start, input_end, target.strand)
            )

            current_start += stride

        if target_input_intervals:
            prediction = predict_intervals(
                target_input_intervals,
                dataset,
                model_manager,
                predict_config,
                device,
                batch_size,
            )
            results.append(prediction)

    return results
