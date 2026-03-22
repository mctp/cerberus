from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cerberus.interval import Interval
from cerberus.utils import import_class

if TYPE_CHECKING:
    from cerberus.config import ModelConfig

logger = logging.getLogger(__name__)


def get_log_count_params(model_config: ModelConfig) -> tuple[bool, float]:
    """Determine log-count transform parameters from the model configuration.

    Losses with ``uses_count_pseudocount = True`` (MSE-family, Dalmatian)
    train log_counts in ``log(count + pseudocount)`` space, while
    Poisson/NB losses use ``log(count)`` directly.

    Args:
        model_config: Model configuration containing ``loss_cls`` and
            ``count_pseudocount``.

    Returns:
        Tuple of ``(log_counts_include_pseudocount, count_pseudocount)``.
        The pseudocount is in scaled units or ``0.0`` for losses that
        do not use it.
    """
    loss_cls = import_class(model_config.loss_cls)
    if loss_cls.uses_count_pseudocount:
        return True, model_config.count_pseudocount
    return False, 0.0

@dataclass(kw_only=True)
class ModelOutput:
    """Base class for model outputs."""
    out_interval: Interval | None = None

    def detach(self) -> ModelOutput:
        """Returns a new instance with all tensors detached from the graph."""
        raise NotImplementedError

@dataclass
class ProfileLogits(ModelOutput):
    """Output for models predicting a profile shape as unnormalized log-probabilities.

    softmax(logits, dim=-1) gives the per-position probability distribution.
    """
    logits: torch.Tensor # (Batch, Channels, Length)

    def detach(self) -> ProfileLogits:
        return ProfileLogits(logits=self.logits.detach(), out_interval=self.out_interval)

@dataclass
class ProfileLogRates(ModelOutput):
    """Output for models predicting log-rates (log-intensities).

    exp(log_rates) gives expected counts per bin.
    """
    log_rates: torch.Tensor # (Batch, Channels, Length)

    def detach(self) -> ProfileLogRates:
        return ProfileLogRates(log_rates=self.log_rates.detach(), out_interval=self.out_interval)

@dataclass
class ProfileCountOutput(ProfileLogits):
    """Output for models with factored profile shape (logits) and total counts.

    The profile shape and total count are predicted independently:
    predicted_counts = softmax(logits) * exp(log_counts).
    """
    log_counts: torch.Tensor # (Batch, Channels)

    def detach(self) -> ProfileCountOutput:
        return ProfileCountOutput(
            logits=self.logits.detach(),
            log_counts=self.log_counts.detach(),
            out_interval=self.out_interval
        )

@dataclass
class FactorizedProfileCountOutput(ProfileCountOutput):
    """Combined output with decomposed sub-model outputs for the Dalmatian model.

    Inherits combined logits and log_counts from ProfileCountOutput.
    Adds decomposed bias and signal sub-model outputs for use by DalmatianLoss
    during training and interpretation tools.
    """
    bias_logits: torch.Tensor       # (B, C, L) -- bias model profile logits
    bias_log_counts: torch.Tensor   # (B, C)   -- bias model log counts
    signal_logits: torch.Tensor     # (B, C, L) -- signal model profile logits
    signal_log_counts: torch.Tensor # (B, C)   -- signal model log counts

    def detach(self) -> FactorizedProfileCountOutput:
        return FactorizedProfileCountOutput(
            logits=self.logits.detach(),
            log_counts=self.log_counts.detach(),
            bias_logits=self.bias_logits.detach(),
            bias_log_counts=self.bias_log_counts.detach(),
            signal_logits=self.signal_logits.detach(),
            signal_log_counts=self.signal_log_counts.detach(),
            out_interval=self.out_interval,
        )

def unbatch_modeloutput(batched_output: ModelOutput, batch_size: int) -> list[dict[str, Any]]:
    """Splits a batched ModelOutput into per-sample dictionaries.

    Tensor fields are unbound along dim=0. Non-tensor fields (e.g. out_interval)
    are replicated to every sample.

    Args:
        batched_output: Batched ModelOutput with tensors of shape (B, ...).
        batch_size: Number of samples in the batch.

    Returns:
        List of dicts, one per sample, with the same keys as the dataclass fields.
    """
    batched_output_dict = {f.name: getattr(batched_output, f.name) for f in dataclasses.fields(batched_output)}

    unbatched_components = {}
    for key, val in batched_output_dict.items():
        if isinstance(val, torch.Tensor):
            unbatched_components[key] = list(torch.unbind(val, dim=0))
        else:
            unbatched_components[key] = [val] * batch_size

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
    """Spatially merges overlapping prediction tensors by averaging overlapping bins.

    Each tensor is placed into a unified bin grid defined by merged_interval. Where
    multiple predictions overlap, their values are averaged. Tensors are classified
    as profile (spatial) or scalar based on whether their last dimension matches the
    expected bin count for their interval.

    Snap-to-grid: when output_bin_size > 1, interval starts are floored to the
    nearest bin boundary. Sub-bin shifts (e.g. from jitter) are quantized away.

    Args:
        outputs: Per-interval tensors, each (C, L) for profiles or (C,) for scalars.
        intervals: Genomic interval for each tensor (same length as outputs).
        merged_interval: Union interval covering all inputs.
        output_len: Expected profile length in bins (currently unused, reserved).
        output_bin_size: Bin resolution in base pairs.

    Returns:
        For profiles: (C, n_bins) array over the merged interval.
        For scalars: (C,) array — overlap-weighted mean, excluding empty gaps.
    """
    min_start = merged_interval.start
    max_end = merged_interval.end

    span_bp = max_end - min_start
    n_bins = span_bp // output_bin_size

    n_channels = outputs[0].shape[0]

    accumulator = np.zeros((n_channels, n_bins), dtype=np.float32)
    counts = np.zeros((1, n_bins), dtype=np.float32)
    has_scalar = False

    for out_tensor, interval in zip(outputs, intervals, strict=True):
        val = out_tensor.cpu().numpy()  # (C, L) or (C,)

        rel_start_bp = interval.start - min_start
        rel_start_bin = rel_start_bp // output_bin_size
        interval_bins = (interval.end - interval.start) // output_bin_size
        end_bin = rel_start_bin + interval_bins

        is_profile = (val.ndim >= 2 and val.shape[-1] == interval_bins)

        if is_profile:
            accumulator[:, rel_start_bin : end_bin] += val
            counts[:, rel_start_bin : end_bin] += 1.0
        else:
            has_scalar = True
            # Broadcast scalar (C,) → (C, 1) for overlap-aware spatial averaging
            val_b = np.expand_dims(val, -1) if val.ndim == 1 else val
            accumulator[:, rel_start_bin : end_bin] += val_b
            counts[:, rel_start_bin : end_bin] += 1.0

    raw_counts = counts.copy()
    counts = np.maximum(counts, 1.0)
    final_values = accumulator / counts

    if has_scalar:
        # Collapse spatial grid back to (C,). Only average over bins that
        # received contributions to avoid dilution from empty gaps.
        valid_mask = raw_counts[0] > 0
        if valid_mask.any():
            final_values = final_values[:, valid_mask].mean(axis=-1)
        else:
            final_values = np.zeros(n_channels, dtype=np.float32)

    return final_values

def aggregate_intervals(
    outputs: list[dict[str, Any]],
    intervals: list[Interval],
    output_len: int,
    output_bin_size: int,
    output_cls: type[ModelOutput] | None = None,
) -> ModelOutput:
    """Merges predictions from multiple genomic intervals into a single ModelOutput.

    Delegates per-tensor spatial merging to ``aggregate_tensor_track_values``.
    All intervals must be on the same chromosome and strand.

    Args:
        outputs: Per-interval output dicts (from ``unbatch_modeloutput``).
        intervals: Genomic interval for each output (same length as outputs).
        output_len: Profile length in bins (passed through; currently unused).
        output_bin_size: Bin resolution in base pairs.
        output_cls: ModelOutput subclass to construct the result.

    Returns:
        A single ModelOutput covering the union of all input intervals.
    """
    keys = [k for k in outputs[0].keys() if isinstance(outputs[0][k], torch.Tensor)]

    chrom = intervals[0].chrom
    strand = intervals[0].strand
    min_start = min(i.start for i in intervals)
    max_end = max(i.end for i in intervals)
    merged_interval = Interval(chrom, min_start, max_end, strand)

    aggregated_components = {}
    for key in keys:
        component_outputs = [out[key] for out in outputs]
        agg_np = aggregate_tensor_track_values(
            component_outputs, intervals, merged_interval, output_len, output_bin_size
        )
        aggregated_components[key] = torch.from_numpy(agg_np)

    if output_cls is None:
        raise ValueError("output_cls must be provided to aggregate_intervals")

    logger.debug(f"Aggregated {len(outputs)} intervals into {merged_interval}")
    return output_cls(**aggregated_components, out_interval=merged_interval)

def aggregate_models(
    outputs: Sequence[ModelOutput], method: str
) -> ModelOutput:
    """Ensembles outputs from multiple models by reducing each tensor field.

    All outputs must be the same ModelOutput subclass with identically-shaped
    tensors. The out_interval is taken from the first output (all models
    predict on the same intervals).

    Args:
        outputs: One ModelOutput per model, all for the same input batch.
        method: Reduction across models — "mean" or "median".

    Returns:
        A single ModelOutput of the same type with ensembled tensor fields.
    """
    output_dicts = [{f.name: getattr(out, f.name) for f in dataclasses.fields(out)} for out in outputs]
    keys = [k for k in output_dicts[0].keys() if isinstance(output_dicts[0][k], torch.Tensor)]

    aggregated_elements = {}
    for key in keys:
        stacked = torch.stack([out[key] for out in output_dicts])
        if method == "mean":
            aggregated_elements[key] = torch.mean(stacked, dim=0)
        elif method == "median":
            aggregated_elements[key] = torch.median(stacked, dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    cls = type(outputs[0])
    out_int = outputs[0].out_interval

    logger.debug(f"Aggregated {len(outputs)} model outputs using '{method}'")
    return cls(**aggregated_elements, out_interval=out_int)

def compute_total_log_counts(model_output: ModelOutput, log_counts_include_pseudocount: bool = False, pseudocount: float = 1.0) -> torch.Tensor:
    """Computes total log counts across all channels, returning shape (B,).

    Supports ProfileCountOutput (explicit log_counts) and ProfileLogRates
    (total count derived by summing per-bin rates).

    Args:
        model_output: A ProfileCountOutput or ProfileLogRates instance.
        log_counts_include_pseudocount: If True, per-channel log_counts are in
            log(count + pseudocount) space (MSE losses). Aggregation inverts
            the transform per channel, sums, then reapplies — avoiding the
            incorrect log(n_channels * pseudocount + total) that logsumexp
            would give. If False (default, Poisson/NB losses), logsumexp
            correctly gives log(total).
        pseudocount: Offset for the pseudocount inversion. Must match the
            count_pseudocount used during training. Default 1.0.

    Returns:
        Tensor of shape (B,) with total log counts per sample.
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
        return torch.logsumexp(log_rates.float().flatten(start_dim=1), dim=-1)
            
    raise ValueError(f"Model output type {type(model_output)} not supported for total log counts extraction.")


def compute_obs_log_counts(
    raw_counts: torch.Tensor,
    target_scale: float,
    log_counts_include_pseudocount: bool,
    pseudocount: float = 0.0,
) -> torch.Tensor:
    """Computes observed total log-counts from raw target counts.

    Applies target_scale to raw counts (to match the training target space),
    then applies the appropriate log transform matching the loss function.

    Args:
        raw_counts: Raw observed counts, shape (B, C, L).
        target_scale: Multiplicative scaling factor from data_config.
        log_counts_include_pseudocount: If True (MSE losses), uses
            log(total + pseudocount).  If False (Poisson/NB), uses
            log(total) with a clamp floor of 1.0 to avoid log(0).
        pseudocount: The scaled pseudocount value (from ``get_log_count_params``).

    Returns:
        Tensor of shape (B,) with observed total log-counts.
    """
    obs_total = raw_counts.sum(dim=(1, 2)) * target_scale
    if log_counts_include_pseudocount:
        return torch.log(obs_total + pseudocount)
    else:
        return torch.log(obs_total.clamp_min(1.0))
