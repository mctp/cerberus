import torch
import torch.nn.functional as F
from torchmetrics import Metric, PearsonCorrCoef, MeanSquaredError, MetricCollection
from cerberus.output import ProfileCountOutput, ProfileLogRates, ProfileLogits


def _per_example_pearson(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Pearson correlation per example along the last dimension.

    Args:
        preds: (B, C, L) predicted values.
        target: (B, C, L) target values.
        eps: Small constant to avoid division by zero.

    Returns:
        (B, C) tensor of per-example, per-channel Pearson correlations.
        Returns NaN where either preds or target has near-zero variance.
    """
    preds_c = preds - preds.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    cov = (preds_c * target_c).sum(dim=-1)
    denom = preds_c.pow(2).sum(dim=-1).sqrt() * target_c.pow(2).sum(dim=-1).sqrt()
    return torch.where(denom > eps, cov / denom, torch.tensor(float("nan"), device=preds.device))

class ProfilePearsonCorrCoef(PearsonCorrCoef):
    """
    Pearson Correlation Coefficient for profile probabilities.
    
    Flattens dimensions (Batch, Length) for each channel to compute correlation
    per channel, then averages across channels.
    
    Operates on probabilities (Softmax of logits or log_rates).
    """
    def __init__(self, num_channels=1, implicit_log_targets=False, **kwargs):
        super().__init__(num_outputs=num_channels, **kwargs)
        self.num_channels = num_channels
        self.implicit_log_targets = implicit_log_targets

    def update(self, preds: ProfileLogRates | ProfileLogits, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileLogRates):
            logits = preds.log_rates
        elif isinstance(preds, ProfileLogits):
            logits = preds.logits
        else:
            raise TypeError("ProfilePearsonCorrCoef requires ProfileLogRates or ProfileLogits")
        
        probs = F.softmax(logits, dim=-1)

        if self.implicit_log_targets:
            target = torch.expm1(target)
            
        # Assumes preds, target: (Batch, Channels, Length)
        # (B, C, L) -> (B, L, C) -> (B*L, C)
        # We ensure channel dimension is last, then flatten batch and length dimensions.
        # This is NOT equivalent to flattening all dimensions and mix channels together.
        preds_flat = probs.detach().permute(0, 2, 1).reshape(-1, self.num_channels)
        target_flat = target.detach().permute(0, 2, 1).reshape(-1, self.num_channels)

        PearsonCorrCoef.update(self, preds_flat, target_flat)

    def compute(self):
        val = super().compute()
        if val.numel() > 1:
            # for multiple channels, skip NaNs when averaging
            val = torch.nanmean(val)
        return val.float()


class CountProfilePearsonCorrCoef(ProfilePearsonCorrCoef):
    """
    Pearson Correlation for reconstructed profile counts (BPNet-style).
    
    Reconstructs predicted profile counts from (logits, log_counts) before
    computing correlation.
    Preds = Softmax(logits) * Exp(log_counts).
    """
    def __init__(self, num_channels=1, implicit_log_targets=False, **kwargs):
        super().__init__(num_channels=num_channels, implicit_log_targets=implicit_log_targets, **kwargs)

    def update(self, preds: ProfileCountOutput, target: torch.Tensor): # type: ignore[override]
        if not isinstance(preds, ProfileCountOutput):
             raise TypeError("CountProfilePearsonCorrCoef requires ProfileCountOutput")

        logits = preds.logits
        log_counts = preds.log_counts

        probs = F.softmax(logits, dim=-1)
        total_counts = torch.expm1(log_counts.float()).clamp_min(0.0)
        
        # Handle (Batch,) edge case
        if total_counts.dim() == 1:
            total_counts = total_counts.unsqueeze(1)
        
        # Broadcasting handles (B, 1, 1) * (B, C, L) -> (B, C, L)
        preds_counts = probs * total_counts.unsqueeze(-1)
        
        if self.implicit_log_targets:
            target = torch.expm1(target)

        # Flatten and call PearsonCorrCoef directly to avoid Softmax in parent update
        preds_flat = preds_counts.detach().permute(0, 2, 1).reshape(-1, self.num_channels)
        target_flat = target.detach().permute(0, 2, 1).reshape(-1, self.num_channels)

        PearsonCorrCoef.update(self, preds_flat, target_flat)


class CountProfileMeanSquaredError(MeanSquaredError):
    """
    Mean Squared Error for reconstructed profile counts (BPNet-style).
    
    Reconstructs predicted profile counts from (logits, log_counts) before
    computing MSE against targets.
    Preds = Softmax(logits) * Exp(log_counts).
    """
    def __init__(self, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.implicit_log_targets = implicit_log_targets

    def update(self, preds: ProfileCountOutput, target: torch.Tensor): # type: ignore[override]
        if not isinstance(preds, ProfileCountOutput):
             raise TypeError("CountProfileMeanSquaredError requires ProfileCountOutput")

        logits = preds.logits
        log_counts = preds.log_counts

        probs = F.softmax(logits, dim=-1)
        total_counts = torch.expm1(log_counts.float()).clamp_min(0.0)
        
        # Handle (Batch,) edge case
        if total_counts.dim() == 1:
            total_counts = total_counts.unsqueeze(1)

        # Broadcasting handles (B, 1, 1) * (B, C, L) -> (B, C, L)
        preds_counts = probs * total_counts.unsqueeze(-1)
        
        if self.implicit_log_targets:
            target = torch.expm1(target)
            
        super().update(preds_counts, target)


class ProfileMeanSquaredError(MeanSquaredError):
    """
    Mean Squared Error on Probability Profiles.
    
    Computes MSE between:
    1. Predicted Probabilities (Softmax of logits)
    2. Target Probabilities (Target Counts / Profile Counts)
    """
    def __init__(self, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.implicit_log_targets = implicit_log_targets

    def update(self, preds: ProfileLogRates | ProfileLogits, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileLogRates):
            logits = preds.log_rates
        elif isinstance(preds, ProfileLogits):
            logits = preds.logits
        else:
             raise TypeError("ProfileMeanSquaredError requires ProfileLogRates or ProfileLogits")
        
        probs = F.softmax(logits, dim=-1)
        
        if self.implicit_log_targets:
            target = torch.expm1(target)

        # Normalize targets to be probabilities (sum to 1 along length)
        # Add epsilon to avoid division by zero
        target_sum = target.sum(dim=-1, keepdim=True)
        target_probs = target / (target_sum + 1e-8)
        
        super().update(probs, target_probs)


class LogCountsMeanSquaredError(MeanSquaredError):
    """
    Mean Squared Error on Log Counts.
    
    Computes MSE between:
    1. Predicted Log Counts (from log_counts or logsumexp of log_rates)
    2. Target Log Counts (log1p of sum of targets)
    """
    def __init__(self, count_per_channel=False, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.count_per_channel = count_per_channel
        self.implicit_log_targets = implicit_log_targets

    def update(self, preds: ProfileCountOutput | ProfileLogRates, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileCountOutput):
            pred_log_counts = preds.log_counts
            # If we want global count but have per-channel counts, aggregate them
            if not self.count_per_channel and pred_log_counts.ndim == 2 and pred_log_counts.shape[1] > 1:
                pred_log_counts = torch.logsumexp(pred_log_counts, dim=1)
                
        elif isinstance(preds, ProfileLogRates):
            if self.count_per_channel:
                pred_log_counts = torch.logsumexp(preds.log_rates, dim=2)
            else:
                pred_log_counts = torch.logsumexp(preds.log_rates.flatten(start_dim=1), dim=-1)
        else:
             raise TypeError("LogCountsMeanSquaredError requires ProfileCountOutput or ProfileLogRates")
        
        if self.implicit_log_targets:
            target = torch.expm1(target)
            
        if self.count_per_channel:
            target_counts = target.sum(dim=2)
            target_log_counts = torch.log1p(target_counts)
        else:
            target_global_count = target.sum(dim=(1, 2))
            target_log_counts = torch.log1p(target_global_count)
            
        # Ensure dimensions match (flatten to 1D if global)
        if not self.count_per_channel:
            pred_log_counts = pred_log_counts.flatten()
            target_log_counts = target_log_counts.flatten()
            
        super().update(pred_log_counts, target_log_counts)


class LogCountsPearsonCorrCoef(PearsonCorrCoef):
    """
    Pearson Correlation on Log Counts.
    
    Computes Correlation between:
    1. Predicted Log Counts (from log_counts or logsumexp of log_rates)
    2. Target Log Counts (log1p of sum of targets)
    """
    def __init__(self, count_per_channel=False, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.count_per_channel = count_per_channel
        self.implicit_log_targets = implicit_log_targets

    def update(self, preds: ProfileCountOutput | ProfileLogRates, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileCountOutput):
            pred_log_counts = preds.log_counts
            # If we want global count but have per-channel counts, aggregate them
            if not self.count_per_channel and pred_log_counts.ndim == 2 and pred_log_counts.shape[1] > 1:
                pred_log_counts = torch.logsumexp(pred_log_counts, dim=1)
                
        elif isinstance(preds, ProfileLogRates):
            if self.count_per_channel:
                pred_log_counts = torch.logsumexp(preds.log_rates, dim=2)
            else:
                pred_log_counts = torch.logsumexp(preds.log_rates.flatten(start_dim=1), dim=-1)
        else:
             raise TypeError("LogCountsPearsonCorrCoef requires ProfileCountOutput or ProfileLogRates")
        
        if self.implicit_log_targets:
            target = torch.expm1(target)
            
        if self.count_per_channel:
            target_counts = target.sum(dim=2)
            target_log_counts = torch.log1p(target_counts)
        else:
            target_global_count = target.sum(dim=(1, 2))
            target_log_counts = torch.log1p(target_global_count)
            
        # Ensure dimensions match (flatten to 1D if global)
        if not self.count_per_channel:
            pred_log_counts = pred_log_counts.flatten()
            target_log_counts = target_log_counts.flatten()
            
        super().update(pred_log_counts, target_log_counts)


class PerExampleProfilePearsonCorrCoef(Metric):
    """
    Per-example Pearson Correlation Coefficient for profile probabilities.

    Computes Pearson correlation for each (example, channel) pair along the
    sequence length dimension, then averages across channels and examples.
    Numerically stable in float32 (no cross-batch accumulation of raw sums).

    Operates on probabilities (Softmax of logits or log_rates).
    """
    full_state_update: bool = False

    def __init__(self, num_channels=1, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.implicit_log_targets = implicit_log_targets
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: ProfileLogRates | ProfileLogits, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileLogRates):
            logits = preds.log_rates
        elif isinstance(preds, ProfileLogits):
            logits = preds.logits
        else:
            raise TypeError("PerExampleProfilePearsonCorrCoef requires ProfileLogRates or ProfileLogits")

        probs = F.softmax(logits, dim=-1)

        if self.implicit_log_targets:
            target = torch.expm1(target)

        corr = _per_example_pearson(probs.detach(), target.detach())  # (B, C)
        corr_mean = torch.nanmean(corr, dim=-1)  # (B,) — average over channels
        valid = ~torch.isnan(corr_mean)
        self.sum_corr += corr_mean[valid].sum()
        self.count += valid.sum()

    def compute(self):
        if self.count == 0:
            return torch.tensor(float("nan"))
        return (self.sum_corr / self.count).float()


class PerExampleCountProfilePearsonCorrCoef(Metric):
    """
    Per-example Pearson Correlation for reconstructed profile counts (BPNet-style).

    Reconstructs predicted profile counts from (logits, log_counts) before
    computing per-example Pearson correlation along the sequence length.
    Numerically stable in float32.
    Preds = Softmax(logits) * Expm1(log_counts).
    """
    full_state_update: bool = False

    def __init__(self, num_channels=1, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.implicit_log_targets = implicit_log_targets
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: ProfileCountOutput, target: torch.Tensor): # type: ignore[override]
        if not isinstance(preds, ProfileCountOutput):
            raise TypeError("PerExampleCountProfilePearsonCorrCoef requires ProfileCountOutput")

        logits = preds.logits
        log_counts = preds.log_counts

        probs = F.softmax(logits, dim=-1)
        total_counts = torch.expm1(log_counts.float()).clamp_min(0.0)

        if total_counts.dim() == 1:
            total_counts = total_counts.unsqueeze(1)

        preds_counts = probs * total_counts.unsqueeze(-1)  # (B, C, L)

        if self.implicit_log_targets:
            target = torch.expm1(target)

        corr = _per_example_pearson(preds_counts.detach(), target.detach())  # (B, C)
        corr_mean = torch.nanmean(corr, dim=-1)  # (B,) — average over channels
        valid = ~torch.isnan(corr_mean)
        self.sum_corr += corr_mean[valid].sum()
        self.count += valid.sum()

    def compute(self):
        if self.count == 0:
            return torch.tensor(float("nan"))
        return (self.sum_corr / self.count).float()


class PerExampleLogCountsPearsonCorrCoef(Metric):
    """
    Pearson Correlation on Log Counts, accumulated as scalar pairs.

    Collects per-example (pred_log_count, target_log_count) pairs and computes
    a single Pearson correlation at epoch end. Numerically stable because counts
    are 1 scalar per example (not L=1024), so the accumulation is small.
    """
    full_state_update: bool = False

    def __init__(self, count_per_channel=False, implicit_log_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.count_per_channel = count_per_channel
        self.implicit_log_targets = implicit_log_targets
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("targets_list", default=[], dist_reduce_fx="cat")

    def update(self, preds: ProfileCountOutput | ProfileLogRates, target: torch.Tensor): # type: ignore[override]
        if isinstance(preds, ProfileCountOutput):
            pred_log_counts = preds.log_counts
            if not self.count_per_channel and pred_log_counts.ndim == 2 and pred_log_counts.shape[1] > 1:
                pred_log_counts = torch.logsumexp(pred_log_counts, dim=1)
        elif isinstance(preds, ProfileLogRates):
            if self.count_per_channel:
                pred_log_counts = torch.logsumexp(preds.log_rates, dim=2)
            else:
                pred_log_counts = torch.logsumexp(preds.log_rates.flatten(start_dim=1), dim=-1)
        else:
            raise TypeError("PerExampleLogCountsPearsonCorrCoef requires ProfileCountOutput or ProfileLogRates")

        if self.implicit_log_targets:
            target = torch.expm1(target)

        if self.count_per_channel:
            target_counts = target.sum(dim=2)
            target_log_counts = torch.log1p(target_counts)
        else:
            target_global_count = target.sum(dim=(1, 2))
            target_log_counts = torch.log1p(target_global_count)

        if not self.count_per_channel:
            pred_log_counts = pred_log_counts.flatten()
            target_log_counts = target_log_counts.flatten()

        self.preds_list.append(pred_log_counts.detach())
        self.targets_list.append(target_log_counts.detach())

    def compute(self):
        # After DDP reduce with dist_reduce_fx="cat", the list may already
        # be a single concatenated tensor rather than a list of tensors.
        if isinstance(self.preds_list, torch.Tensor):
            all_preds = self.preds_list
            all_targets = self.targets_list
        elif len(self.preds_list) == 0:
            return torch.tensor(float("nan"))
        else:
            all_preds = torch.cat(self.preds_list)
            all_targets = torch.cat(self.targets_list)
        if all_preds.numel() < 2:
            return torch.tensor(float("nan"))
        preds_c = all_preds - all_preds.mean()
        target_c = all_targets - all_targets.mean()
        cov = (preds_c * target_c).sum()
        denom = preds_c.pow(2).sum().sqrt() * target_c.pow(2).sum().sqrt()
        if denom < 1e-8:
            return torch.tensor(float("nan"))
        return (cov / denom).float()


class DefaultMetricCollection(MetricCollection):
    """
    Default MetricCollection used for training/validation.
    Includes Pearson Correlation, Profile MSE, and Log Counts MSE.
    """
    def __init__(self, num_channels: int = 1, implicit_log_targets: bool = False):
        super().__init__({
            "pearson": ProfilePearsonCorrCoef(num_channels=num_channels, implicit_log_targets=implicit_log_targets),
            # MSE is element-wise, so Global MSE is equivalent to Mean Per-Channel MSE
            # (assuming equal number of elements per channel). Thus no custom flattening is needed.
            "mse_profile": ProfileMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "mse_log_counts": LogCountsMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "pearson_log_counts": LogCountsPearsonCorrCoef(implicit_log_targets=implicit_log_targets),
        })
