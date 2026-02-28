from dataclasses import dataclass
import dataclasses
import logging
import torch
import numpy as np
from typing import Any, Sequence
from cerberus.interval import Interval

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class ModelOutput:
    """Base class for model outputs."""
    out_interval: Interval | None = None

    def detach(self):
        """Returns a new instance with all tensors detached from the graph."""
        raise NotImplementedError

@dataclass
class ProfileLogits(ModelOutput):
    """
    Output for models predicting a profile (shape) using unnormalized log-probabilities.
    Interpretation: softmax(logits) = probabilities.
    """
    logits: torch.Tensor # (Batch, Channels, Length)

    def detach(self):
        return ProfileLogits(logits=self.logits.detach(), out_interval=self.out_interval)

@dataclass
class ProfileLogRates(ModelOutput):
    """
    Output for models predicting log-rates (log-intensities).
    Interpretation: exp(log_rates) = counts.
    """
    log_rates: torch.Tensor # (Batch, Channels, Length)

    def detach(self):
        return ProfileLogRates(log_rates=self.log_rates.detach(), out_interval=self.out_interval)

@dataclass
class ProfileCountOutput(ProfileLogits):
    """Output for models predicting profile (logits) and total counts."""
    log_counts: torch.Tensor # (Batch, Channels)

    def detach(self):
        return ProfileCountOutput(
            logits=self.logits.detach(), 
            log_counts=self.log_counts.detach(),
            out_interval=self.out_interval
        )

def unbatch_modeloutput(batched_output: ModelOutput, batch_size: int) -> list[dict[str, Any]]:
    """
    Splits a batched output (ModelOutput) into a list of individual interval dictionaries.
    
    Args:
        batched_output: The batched ModelOutput object to split.
        batch_size: The number of items in the batch.
        
    Returns:
        list[dict[str, Any]]: A list of dictionaries, each representing an unbatched output.
    """
    batched_output_dict = dataclasses.asdict(batched_output)

    unbatched_components = {}
    for key, val in batched_output_dict.items():
        if isinstance(val, torch.Tensor):
            unbatched_components[key] = list(torch.unbind(val, dim=0))
        else:
            # For metadata fields like out_interval, replicate
            unbatched_components[key] = [val] * batch_size
    
    # Reassemble into list of dicts
    result_list = []
    keys = unbatched_components.keys()
    for i in range(batch_size):
        item = {key: unbatched_components[key][i] for key in keys}
        result_list.append(item)
        
    return result_list

def aggregate_tensor_track_values(
    outputs: list[torch.Tensor],
    intervals: list[Interval],
    merged_interval: Interval,
    output_len: int,
    output_bin_size: int,
) -> np.ndarray:
    """
    Aggregates a list of tensors into a single merged array.
    
    This function handles the spatial alignment of multiple (potentially overlapping)
    prediction tracks into a single contiguous track. It computes the average value
    for bins where multiple predictions overlap.
    
    Note on Alignment (Snap-to-Grid):
    When output_bin_size > 1, the function snaps interval starts to the nearest bin
    (flooring behavior). Sub-bin shifts (e.g. from Jitter) are effectively Aliased/Quantized
    to the bin grid defined by merged_interval.start.
    
    Args:
        outputs: List of tensors to aggregate.
        intervals: List of intervals corresponding to the tensors.
        merged_interval: The overall interval covering all inputs.
        output_len: The expected length of profile outputs.
        output_bin_size: The bin size of the outputs.
        
    Returns:
        np.ndarray: The aggregated values as a numpy array.
    """
    # Merged extent
    min_start = merged_interval.start
    max_end = merged_interval.end
    
    span_bp = max_end - min_start
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
        
        # Relative start bin
        rel_start_bp = interval.start - min_start
        rel_start_bin = rel_start_bp // output_bin_size

        # Expected bins for this interval
        interval_len_bp = interval.end - interval.start
        interval_bins = interval_len_bp // output_bin_size
        
        # Check dimensions
        # It is a profile/track if it has spatial dimension and matches the interval length
        is_profile = (val.ndim >= 2 and val.shape[-1] == interval_bins)
        
        if is_profile:
            # val is (C, L_bins)
            end_bin = rel_start_bin + interval_bins
            
            # Bounds check (simple clipping or assume fits)
            # Since intervals are within merged_interval, it should fit unless precision issues
            
            accumulator[:, rel_start_bin : end_bin] += val
            counts[:, rel_start_bin : end_bin] += 1.0
        else:
            # Scalar (C,)
            # Broadcast to interval length to create a constant track over the interval
            # Interval length in bins
            end_bin = rel_start_bin + interval_bins
            
            # val is (C,) -> (C, int_bins)
            if val.ndim == 1:
                val_b = np.expand_dims(val, -1)
            else:
                val_b = val

            accumulator[:, rel_start_bin : end_bin] += val_b
            counts[:, rel_start_bin : end_bin] += 1.0

    # Average
    counts = np.maximum(counts, 1.0)
    final_values = accumulator / counts
    
    return final_values

def aggregate_intervals(
    outputs: list[dict[str, Any]],
    intervals: list[Interval],
    output_len: int,
    output_bin_size: int,
    output_cls: type[ModelOutput] | None = None,
) -> ModelOutput:
    """
    Aggregates overlapping predictions into a single merged output.
    
    This high-level function unifies predictions from multiple genomic intervals (e.g. tiles)
    into a single ModelOutput object covering the union of all input intervals.
    It delegates to `aggregate_tensor_track_values` for spatial merging of profile tracks.
    
    Args:
        outputs: List of unbatched output dictionaries.
        intervals: List of intervals corresponding to the outputs.
        output_len: The length of the output profile in bins/bp (used for profile detection).
        output_bin_size: The size of each bin in base pairs.
        output_cls: The ModelOutput class to use for the result.
        
    Returns:
        ModelOutput: The merged output object covering the union of intervals.
    """
    # Filter for keys that are tensors (ignore metadata like out_interval)
    keys = [k for k in outputs[0].keys() if isinstance(outputs[0][k], torch.Tensor)]

    # Compute merged interval, assuming all intervals are on the same chrom and strand
    chrom = intervals[0].chrom
    strand = intervals[0].strand
    min_start = min(i.start for i in intervals)
    max_end = max(i.end for i in intervals)
    merged_interval = Interval(chrom, min_start, max_end, strand)
    
    aggregated_components = {}
    
    for key in keys:
        component_outputs = [out[key] for out in outputs] # list[Tensor]
        agg_comp = aggregate_tensor_track_values(
            component_outputs, intervals, merged_interval, output_len, output_bin_size
        )
        # Convert numpy to tensor (CPU)
        aggregated_components[key] = torch.from_numpy(agg_comp)

    if output_cls is None:
        raise ValueError("output_cls must be provided to aggregate_intervals")

    logger.debug(f"Aggregated {len(outputs)} intervals into {merged_interval}")
    return output_cls(**aggregated_components, out_interval=merged_interval)

def aggregate_models(
    outputs: Sequence[ModelOutput], method: str
) -> ModelOutput:
    """
    Aggregates a list of model outputs.
    
    Args:
        outputs: List of outputs to aggregate.
        method: Aggregation method ("mean" or "median").
        
    Returns:
        ModelOutput: The aggregated output.
    """
    output_dicts = [dataclasses.asdict(out) for out in outputs]
    # Filter tensor keys
    keys = [k for k in output_dicts[0].keys() if isinstance(output_dicts[0][k], torch.Tensor)]
    
    aggregated_elements = {}
    
    for key in keys:
        stacked = torch.stack([out[key] for out in output_dicts])
        
        if method == "mean":
            # mean over N_Models dimension (dim 0)
            aggregated_elements[key] = torch.mean(stacked, dim=0)
        elif method == "median":
            # median over N_Models dimension (dim 0)
            aggregated_elements[key] = torch.median(stacked, dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    # Reconstruct object
    cls = type(outputs[0])
    # Preserve out_interval if consistent? 
    # Usually for batched aggregation, out_interval is None or same.
    # We take the first one.
    out_int = outputs[0].out_interval

    logger.debug(f"Aggregated {len(outputs)} model outputs using '{method}'")
    return cls(**aggregated_elements, out_interval=out_int)

def compute_total_log_counts(model_output: ModelOutput, log_counts_include_pseudocount: bool = False, pseudocount: float = 1.0) -> torch.Tensor:
    """
    Extracts total log counts from the model output.
    Supports ProfileCountOutput and ProfileLogRates.

    Args:
        model_output: The output from the model.
        log_counts_include_pseudocount: If True, indicates that per-channel log_counts
            are in log(count + pseudocount) space (as trained by MSEMultinomialLoss with
            count_per_channel=True). In this case, multi-channel counts are aggregated
            by inverting the log transform per channel, summing, then reapplying it —
            giving the correct log(total + pseudocount) rather than the incorrect
            log(n_channels + total) that logsumexp would produce.
            If False (default), log_counts are treated as being in log space (Poisson/NB
            losses), where logsumexp correctly gives log(total).
        pseudocount: Additive offset used when log_counts_include_pseudocount=True to
            invert and reapply the log transform. Must match the count_pseudocount used
            during training. Default 1.0 reproduces the original log1p behaviour.

    Returns:
        A tensor of shape (batch_size,) containing the total log counts.
    """
    if isinstance(model_output, ProfileCountOutput):
        log_counts = model_output.log_counts
        if log_counts.ndim == 2 and log_counts.shape[1] > 1:
            if log_counts_include_pseudocount:
                # Undo log(x + pseudocount) per channel, sum channels, reapply.
                total = (torch.exp(log_counts.float()) - pseudocount).clamp_min(0.0).sum(dim=1)
                return torch.log(total + pseudocount)
            else:
                return torch.logsumexp(log_counts.float(), dim=1)
        else:
            return log_counts.flatten()
            
    elif isinstance(model_output, ProfileLogRates):
        log_rates = model_output.log_rates
        if log_rates.shape[1] > 1:
            return torch.logsumexp(log_rates.float().flatten(start_dim=1), dim=-1)
        else:
            return torch.logsumexp(log_rates.float(), dim=(1,2)).flatten()
            
    raise ValueError(f"Model output type {type(model_output)} not supported for total log counts extraction.")
