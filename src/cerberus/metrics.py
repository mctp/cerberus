from typing import Any, cast

import torch
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, Metric, MetricCollection

from cerberus.output import ProfileCountOutput, ProfileLogits, ProfileLogRates


def _per_example_pearson(
    preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
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
    return torch.where(
        denom > eps, cov / denom, torch.tensor(float("nan"), device=preds.device)
    )


class ProfilePearsonCorrCoef(Metric):
    """
    Pearson Correlation Coefficient for profile probabilities.

    Computes Pearson correlation for each (example, channel) pair along the
    sequence length dimension, then averages across channels and examples.
    Numerically stable in float32 (no cross-batch accumulation of raw sums).

    Operates on probabilities (Softmax of logits or log_rates).
    """

    full_state_update: bool | None = False
    sum_corr: torch.Tensor
    count: torch.Tensor

    def __init__(self, log1p_targets: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self, preds: ProfileLogRates | ProfileLogits, target: torch.Tensor
    ) -> None:  # type: ignore[override]
        if isinstance(preds, ProfileLogRates):
            logits = preds.log_rates
        elif isinstance(preds, ProfileLogits):
            logits = preds.logits
        else:
            raise TypeError(
                "ProfilePearsonCorrCoef requires ProfileLogRates or ProfileLogits"
            )

        probs = F.softmax(logits, dim=-1)

        if self.log1p_targets:
            target = torch.expm1(target)

        corr = _per_example_pearson(probs.detach(), target.detach())  # (B, C)
        corr_mean = torch.nanmean(corr, dim=-1)  # (B,) — average over channels
        valid = ~torch.isnan(corr_mean)
        self.sum_corr += corr_mean[valid].sum()
        self.count += valid.sum()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.sum_corr.device)
        return (self.sum_corr / self.count).float()


class CountProfilePearsonCorrCoef(Metric):
    """
    Pearson Correlation for reconstructed profile counts (BPNet-style).

    Reconstructs predicted profile counts from (logits, log_counts) before
    computing per-example Pearson correlation along the sequence length.
    Numerically stable in float32.
    Preds = Softmax(logits) * (Exp(log_counts) - count_pseudocount).
    """

    full_state_update: bool | None = False
    sum_corr: torch.Tensor
    count: torch.Tensor

    def __init__(
        self, log1p_targets: bool = False, count_pseudocount: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets
        self.count_pseudocount = count_pseudocount
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: ProfileCountOutput, target: torch.Tensor) -> None:  # type: ignore[override]
        if not isinstance(preds, ProfileCountOutput):
            raise TypeError("CountProfilePearsonCorrCoef requires ProfileCountOutput")

        logits = preds.logits
        log_counts = preds.log_counts

        probs = F.softmax(logits, dim=-1)
        total_counts = (
            torch.exp(log_counts.float()) - self.count_pseudocount
        ).clamp_min(0.0)

        if total_counts.dim() == 1:
            total_counts = total_counts.unsqueeze(1)

        # When log_counts is global (B, 1) but logits has C > 1 channels,
        # distribute the total count equally across channels so that
        # sum(preds_counts) = total, not C * total.
        n_channels = logits.shape[1]
        if total_counts.shape[1] == 1 and n_channels > 1:
            total_counts = total_counts / n_channels

        preds_counts = probs * total_counts.unsqueeze(-1)  # (B, C, L)

        target = target.float()
        if self.log1p_targets:
            target = torch.expm1(target)

        corr = _per_example_pearson(preds_counts.detach(), target.detach())  # (B, C)
        corr_mean = torch.nanmean(corr, dim=-1)  # (B,) — average over channels
        valid = ~torch.isnan(corr_mean)
        self.sum_corr += corr_mean[valid].sum()
        self.count += valid.sum()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"), device=self.sum_corr.device)
        return (self.sum_corr / self.count).float()


class CountProfileMeanSquaredError(MeanSquaredError):
    """
    Mean Squared Error for reconstructed profile counts (BPNet-style).

    Reconstructs predicted profile counts from (logits, log_counts) before
    computing MSE against targets.
    Preds = Softmax(logits) * (Exp(log_counts) - count_pseudocount).
    """

    def __init__(
        self, log1p_targets: bool = False, count_pseudocount: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets
        self.count_pseudocount = count_pseudocount

    def update(self, preds: ProfileCountOutput, target: torch.Tensor) -> None:  # type: ignore[override]
        if not isinstance(preds, ProfileCountOutput):
            raise TypeError("CountProfileMeanSquaredError requires ProfileCountOutput")

        logits = preds.logits
        log_counts = preds.log_counts

        probs = F.softmax(logits, dim=-1)
        total_counts = (
            torch.exp(log_counts.float()) - self.count_pseudocount
        ).clamp_min(0.0)

        # Handle (Batch,) edge case
        if total_counts.dim() == 1:
            total_counts = total_counts.unsqueeze(1)

        # When log_counts is global (B, 1) but logits has C > 1 channels,
        # distribute the total count equally across channels so that
        # sum(preds_counts) = total, not C * total.
        n_channels = logits.shape[1]
        if total_counts.shape[1] == 1 and n_channels > 1:
            total_counts = total_counts / n_channels

        # Broadcasting handles (B, 1, 1) * (B, C, L) -> (B, C, L)
        preds_counts = probs * total_counts.unsqueeze(-1)

        if self.log1p_targets:
            target = torch.expm1(target)

        super().update(preds_counts, target)


class ProfileMeanSquaredError(MeanSquaredError):
    """
    Mean Squared Error on Probability Profiles.

    Computes MSE between:
    1. Predicted Probabilities (Softmax of logits)
    2. Target Probabilities (Target Counts / Profile Counts)
    """

    def __init__(self, log1p_targets: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets

    def update(
        self, preds: ProfileLogRates | ProfileLogits, target: torch.Tensor
    ) -> None:  # type: ignore[override]
        if isinstance(preds, ProfileLogRates):
            logits = preds.log_rates
        elif isinstance(preds, ProfileLogits):
            logits = preds.logits
        else:
            raise TypeError(
                "ProfileMeanSquaredError requires ProfileLogRates or ProfileLogits"
            )

        probs = F.softmax(logits, dim=-1)

        target = target.float()
        if self.log1p_targets:
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
    2. Target Log Counts: log(sum(targets) + count_pseudocount)
    """

    def __init__(
        self,
        count_per_channel: bool = False,
        log1p_targets: bool = False,
        count_pseudocount: float = 1.0,
        log_counts_include_pseudocount: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.count_per_channel = count_per_channel
        self.log1p_targets = log1p_targets
        self.count_pseudocount = count_pseudocount
        self.log_counts_include_pseudocount = log_counts_include_pseudocount

    def _aggregate_pred_log_counts(
        self, preds: ProfileCountOutput | ProfileLogRates
    ) -> torch.Tensor:
        """Extract and aggregate predicted log-counts from model output."""
        if isinstance(preds, ProfileCountOutput):
            pred_log_counts = preds.log_counts
            if (
                not self.count_per_channel
                and pred_log_counts.ndim == 2
                and pred_log_counts.shape[1] > 1
            ):
                if self.log_counts_include_pseudocount:
                    # Invert per-channel offset-log, sum, reapply to avoid
                    # log(total + C*p) that logsumexp would give.
                    total = (
                        (torch.exp(pred_log_counts.float()) - self.count_pseudocount)
                        .clamp_min(0.0)
                        .sum(dim=1)
                    )
                    pred_log_counts = torch.log(total + self.count_pseudocount)
                else:
                    pred_log_counts = torch.logsumexp(pred_log_counts.float(), dim=1)
            return pred_log_counts

        elif isinstance(preds, ProfileLogRates):
            if self.count_per_channel:
                return torch.logsumexp(preds.log_rates.float(), dim=2)
            else:
                return torch.logsumexp(
                    preds.log_rates.float().flatten(start_dim=1), dim=-1
                )

        raise TypeError("requires ProfileCountOutput or ProfileLogRates")

    def update(
        self, preds: ProfileCountOutput | ProfileLogRates, target: torch.Tensor
    ) -> None:  # type: ignore[override]
        pred_log_counts = self._aggregate_pred_log_counts(preds)

        target = target.float()
        if self.log1p_targets:
            target = torch.expm1(target)

        if self.count_per_channel:
            target_counts = target.sum(dim=2)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
        else:
            target_global_count = target.sum(dim=(1, 2))
            target_log_counts = torch.log(target_global_count + self.count_pseudocount)

        # Ensure dimensions match (flatten to 1D if global)
        if not self.count_per_channel:
            pred_log_counts = pred_log_counts.flatten()
            target_log_counts = target_log_counts.flatten()

        super().update(pred_log_counts, target_log_counts)


class LogCountsPearsonCorrCoef(Metric):
    """
    Pearson Correlation on Log Counts.

    Collects per-example (pred_log_count, target_log_count) pairs and computes
    a single Pearson correlation at epoch end. Numerically stable because counts
    are 1 scalar per example (not L=1024), so the accumulation is small.
    """

    full_state_update: bool | None = False
    preds_list: list[torch.Tensor]
    targets_list: list[torch.Tensor]

    def __init__(
        self,
        count_per_channel: bool = False,
        log1p_targets: bool = False,
        count_pseudocount: float = 1.0,
        log_counts_include_pseudocount: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.count_per_channel = count_per_channel
        self.log1p_targets = log1p_targets
        self.count_pseudocount = count_pseudocount
        self.log_counts_include_pseudocount = log_counts_include_pseudocount
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")
        self.add_state("targets_list", default=[], dist_reduce_fx="cat")

    def update(
        self, preds: ProfileCountOutput | ProfileLogRates, target: torch.Tensor
    ) -> None:  # type: ignore[override]
        if isinstance(preds, ProfileCountOutput):
            pred_log_counts = preds.log_counts
            if (
                not self.count_per_channel
                and pred_log_counts.ndim == 2
                and pred_log_counts.shape[1] > 1
            ):
                if self.log_counts_include_pseudocount:
                    total = (
                        (torch.exp(pred_log_counts.float()) - self.count_pseudocount)
                        .clamp_min(0.0)
                        .sum(dim=1)
                    )
                    pred_log_counts = torch.log(total + self.count_pseudocount)
                else:
                    pred_log_counts = torch.logsumexp(pred_log_counts.float(), dim=1)
        elif isinstance(preds, ProfileLogRates):
            if self.count_per_channel:
                pred_log_counts = torch.logsumexp(preds.log_rates.float(), dim=2)
            else:
                pred_log_counts = torch.logsumexp(
                    preds.log_rates.float().flatten(start_dim=1), dim=-1
                )
        else:
            raise TypeError(
                "LogCountsPearsonCorrCoef requires ProfileCountOutput or ProfileLogRates"
            )

        target = target.float()
        if self.log1p_targets:
            target = torch.expm1(target)

        if self.count_per_channel:
            target_counts = target.sum(dim=2)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
        else:
            target_global_count = target.sum(dim=(1, 2))
            target_log_counts = torch.log(target_global_count + self.count_pseudocount)

        if not self.count_per_channel:
            pred_log_counts = pred_log_counts.flatten()
            target_log_counts = target_log_counts.flatten()

        self.preds_list.append(pred_log_counts.detach())
        self.targets_list.append(target_log_counts.detach())

    def compute(self) -> torch.Tensor:
        # After DDP reduce with dist_reduce_fx="cat", the list may already
        # be a single concatenated tensor rather than a list of tensors.
        if isinstance(self.preds_list, torch.Tensor):
            all_preds = cast(torch.Tensor, self.preds_list)
            all_targets = cast(torch.Tensor, self.targets_list)
        elif len(self.preds_list) == 0:
            return torch.tensor(float("nan"), device=self.device)
        else:
            all_preds = torch.cat(self.preds_list)
            all_targets = torch.cat(self.targets_list)
        if all_preds.numel() < 2:
            return torch.tensor(float("nan"), device=all_preds.device)
        preds_c = all_preds - all_preds.mean()
        target_c = all_targets - all_targets.mean()
        cov = (preds_c * target_c).sum()
        denom = preds_c.pow(2).sum().sqrt() * target_c.pow(2).sum().sqrt()
        if denom < 1e-8:
            return torch.tensor(float("nan"), device=all_preds.device)
        return (cov / denom).float()


class DefaultMetricCollection(MetricCollection):
    """
    Default MetricCollection used for training/validation.
    Includes Pearson Correlation, Profile MSE, and Log Counts MSE.
    """

    def __init__(
        self,
        log1p_targets: bool = False,
        count_pseudocount: float = 1.0,
        log_counts_include_pseudocount: bool = False,
    ):
        super().__init__(
            {
                "pearson": ProfilePearsonCorrCoef(log1p_targets=log1p_targets),
                # MSE is element-wise, so Global MSE is equivalent to Mean Per-Channel MSE
                # (assuming equal number of elements per channel). Thus no custom flattening is needed.
                "mse_profile": ProfileMeanSquaredError(log1p_targets=log1p_targets),
                "mse_log_counts": LogCountsMeanSquaredError(
                    log1p_targets=log1p_targets,
                    count_pseudocount=count_pseudocount,
                    log_counts_include_pseudocount=log_counts_include_pseudocount,
                ),
                "pearson_log_counts": LogCountsPearsonCorrCoef(
                    log1p_targets=log1p_targets,
                    count_pseudocount=count_pseudocount,
                    log_counts_include_pseudocount=log_counts_include_pseudocount,
                ),
            }
        )
