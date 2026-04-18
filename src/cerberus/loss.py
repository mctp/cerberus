from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerberus.output import (
    FactorizedProfileCountOutput,
    ProfileCountOutput,
    ProfileLogRates,
)
from cerberus.utils import import_class


class CerberusLoss(Protocol):
    """Protocol for all cerberus loss functions.

    Every loss must implement ``loss_components`` (returning a dict of named
    scalar tensors) and ``__call__`` (returning the combined scalar loss).
    """

    def loss_components(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]: ...

    def __call__(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor: ...


class ProfilePoissonNLLLoss(nn.PoissonNLLLoss):
    """
    Poisson NLL Loss on Profile LogRates.

    Interprets input as unnormalized log-counts (log-intensity).
    Computes Poisson NLL between exp(log_rates) and targets.

    Args:
        log1p_targets (bool): If True, assumes targets are log1p transformed.
        count_pseudocount (float): Accepted for compatibility with propagate_pseudocount.
            Not used (PoissonNLLLoss handles count loss directly).
    """

    uses_count_pseudocount: bool = False

    def __init__(
        self, log1p_targets: bool = False, count_pseudocount: float = 1.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets
        # count_pseudocount is accepted for compatibility with propagate_pseudocount
        # but not used (PoissonNLLLoss handles count loss directly).

    def loss_components(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()

        if not isinstance(outputs, ProfileLogRates):
            raise TypeError("ProfilePoissonNLLLoss requires ProfileLogRates")

        log_input = outputs.log_rates

        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        return {"poisson_nll_loss": super().forward(log_input, targets)}

    def forward(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:  # type: ignore[override]
        components = self.loss_components(outputs, targets, **kwargs)
        return components["poisson_nll_loss"]


class MSEMultinomialLoss(nn.Module):
    """
    Multinomial NLL Profile Loss + MSE Count Loss.
    Also known as BPNet Loss.

    Objective:
      1. Profile Loss: Multinomial NLL (using logits as unnormalized log-probs).
      2. Count Loss: MSE of log(global_count) (Sum over all channels and length).

    Requires ProfileCountOutput (logits, log_counts).
    """

    uses_count_pseudocount: bool = True

    def __init__(
        self,
        count_weight: float = 1.0,
        profile_weight: float = 1.0,
        flatten_channels: bool = False,
        count_per_channel: bool = False,
        average_channels: bool = False,
        log1p_targets: bool = False,
        epsilon: float = 1e-8,
        count_pseudocount: float = 1.0,
    ) -> None:
        """
        Args:
            count_weight (float): Weight for the count loss component.
            profile_weight (float): Weight for the profile loss component.
            flatten_channels (bool): Controls how profile loss is computed across channels.
                - If False (default): Computes Multinomial NLL independently for each channel.
                  Each channel's profile is normalized separately (Softmax over Length).
                  Loss aggregation across channels depends on `average_channels`.
                  Useful when channels represent independent distributions (e.g., stranded profiles).
                - If True: Flattens channels and length into a single dimension.
                  Computes Multinomial NLL over the entire (Channels x Length) output.
                  Profile is normalized globally (Softmax over Channels * Length).
                  Useful when relative abundance between channels matters in the profile shape.
            count_per_channel (bool): Controls how count loss is computed.
                - If False (default): Computes count loss on global count (sum over channels and length).
                - If True: Computes count loss per channel (sum over length only).
            average_channels (bool): If True, averages profile loss across channels (when flatten_channels=False).
                                     If False (default), sums profile loss across channels.
            log1p_targets (bool): If True, assumes targets are log1p transformed.
            epsilon (float): Small constant for numerical stability.
            count_pseudocount (float): Additive offset applied before log-transforming count
                targets: log(count + count_pseudocount). Prevents log(0) for silent regions
                and encodes the minimum meaningful count. Must be in the same units as the
                scaled targets. Set via data_config.count_pseudocount in hparams (propagated
                automatically by propagate_pseudocount in instantiate()). Default 1.0 reproduces log1p behaviour.
        """
        super().__init__()
        self.count_weight = count_weight
        self.profile_weight = profile_weight
        self.flatten_channels = flatten_channels
        self.count_per_channel = count_per_channel
        self.average_channels = average_channels
        self.log1p_targets = log1p_targets
        self.epsilon = epsilon
        self.count_pseudocount = count_pseudocount

    def _compute_profile_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.flatten_channels:
            logits_flat = logits.flatten(start_dim=1)
            targets_flat = targets.flatten(start_dim=1)

            profile_counts = targets_flat.sum(dim=-1)

            log_fact_sum = torch.lgamma(profile_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets_flat + 1), dim=-1)

            log_probs = F.log_softmax(logits_flat, dim=-1)
            log_prod_exp = torch.sum(targets_flat * log_probs, dim=-1)

            profile_loss = -log_fact_sum + log_prod_fact - log_prod_exp
            return profile_loss.mean()

        else:
            profile_counts = targets.sum(dim=-1)

            log_fact_sum = torch.lgamma(profile_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets + 1), dim=-1)

            log_probs = F.log_softmax(logits, dim=-1)
            log_prod_exp = torch.sum(targets * log_probs, dim=-1)

            profile_loss_per_channel = -log_fact_sum + log_prod_fact - log_prod_exp

            if self.average_channels:
                return profile_loss_per_channel.mean()
            else:
                return profile_loss_per_channel.sum(dim=-1).mean()

    def loss_components(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if not isinstance(outputs, ProfileCountOutput):
            raise TypeError("MSEMultinomialLoss requires ProfileCountOutput")
        logits = outputs.logits
        pred_log_counts = outputs.log_counts
        profile_loss = self._compute_profile_loss(logits, targets)
        if self.count_per_channel:
            target_counts = targets.sum(dim=2)  # (B, C)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
            if pred_log_counts.shape != target_log_counts.shape:
                raise ValueError(
                    f"count_per_channel=True requires per-channel log_counts "
                    f"{target_log_counts.shape}, but model predicted "
                    f"{pred_log_counts.shape}. Set predict_total_count=False "
                    f"in model_args when using count_per_channel=True."
                )
            count_loss = F.mse_loss(pred_log_counts, target_log_counts)
        else:
            target_global_count = targets.sum(dim=(1, 2))
            target_log_global_count = torch.log(
                target_global_count + self.count_pseudocount
            )
            count_loss = F.mse_loss(pred_log_counts.flatten(), target_log_global_count)
        return {"profile_loss": profile_loss, "count_loss": count_loss}

    def forward(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(outputs, targets, **kwargs)
        return (
            self.profile_weight * components["profile_loss"]
            + self.count_weight * components["count_loss"]
        )


class CoupledMSEMultinomialLoss(MSEMultinomialLoss):
    """
    Multinomial NLL Profile Loss + MSE Count Loss (Coupled).
    Mathematically equivalent to MSEMultinomialLoss but derives counts from log_rates.

    Objective:
      1. Profile Loss: Multinomial NLL.
      2. Count Loss: MSE of log(global_count).

    Accepts ProfileLogRates only. Simulates log_counts via LogSumExp
    (interpreting inputs as log-intensities) over all channels and bins.
    Does NOT accept ProfileCountOutput (to avoid ambiguity with MSEMultinomialLoss).
    """

    def loss_components(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if isinstance(outputs, ProfileCountOutput):
            raise TypeError(
                "CoupledMSEMultinomialLoss does not accept ProfileCountOutput. Use MSEMultinomialLoss instead."
            )
        if not isinstance(outputs, ProfileLogRates):
            raise TypeError("CoupledMSEMultinomialLoss requires ProfileLogRates")
        logits = outputs.log_rates
        if self.count_per_channel:
            pred_log_counts = torch.logsumexp(logits.float(), dim=2)
            target_counts = targets.sum(dim=2)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
            count_loss = F.mse_loss(pred_log_counts, target_log_counts)
        else:
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1)
            target_global_count = targets.sum(dim=(1, 2))
            target_log_global_count = torch.log(
                target_global_count + self.count_pseudocount
            )
            count_loss = F.mse_loss(pred_log_counts, target_log_global_count)
        profile_loss = self._compute_profile_loss(logits, targets)
        return {"profile_loss": profile_loss, "count_loss": count_loss}

    def forward(
        self, outputs: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(outputs, targets, **kwargs)
        return (
            self.profile_weight * components["profile_loss"]
            + self.count_weight * components["count_loss"]
        )


class PoissonMultinomialLoss(nn.Module):
    """
    Poisson Multinomial Loss (Global Count).

    Objective:
      1. Profile Loss: Cross Entropy (Multinomial NLL form).
      2. Count Loss: Poisson NLL on Global Count.

    Requires ProfileCountOutput.
    """

    uses_count_pseudocount: bool = False

    def __init__(
        self,
        count_weight: float = 0.2,
        profile_weight: float = 1.0,
        flatten_channels: bool = False,
        count_per_channel: bool = False,
        average_channels: bool = True,
        log1p_targets: bool = False,
        epsilon: float = 1e-8,
        count_pseudocount: float = 1.0,
    ) -> None:
        """
        Args:
            count_weight (float): Weight for the count loss component.
            profile_weight (float): Weight for the profile loss component.
            flatten_channels (bool): Controls how profile loss is computed across channels.
                - If False (default): Computes Cross Entropy independently for each channel.
                  (Softmax over Length). Summed across channels.
                - If True: Flattens channels and length. Computes Cross Entropy over
                  (Channels x Length). (Softmax over Channels * Length).
            count_per_channel (bool): Controls how count loss is computed.
                - If False (default): Computes count loss on global count (sum over channels and length).
                - If True: Computes count loss per channel (sum over length only).
            average_channels (bool): If True (default), averages profile loss across channels (when flatten_channels=False).
                                     If False, sums profile loss across channels.
            log1p_targets (bool): If True, assumes targets are log1p transformed.
            epsilon (float): Small constant for numerical stability.
            count_pseudocount (float): Accepted for compatibility with propagate_pseudocount.
                Not used by Poisson count loss (which uses PoissonNLLLoss directly).
        """
        super().__init__()
        self.count_weight = count_weight
        self.profile_weight = profile_weight
        self.flatten_channels = flatten_channels
        self.count_per_channel = count_per_channel
        self.average_channels = average_channels
        self.log1p_targets = log1p_targets
        self.epsilon = epsilon
        self.count_loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)

    def _compute_profile_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.flatten_channels:
            logits_flat = logits.flatten(start_dim=1)
            targets_flat = targets.flatten(start_dim=1)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss_shape = -torch.sum(targets_flat * log_probs, dim=-1).mean()
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            loss_shape_per_channel = -torch.sum(targets * log_probs, dim=-1)
            if self.average_channels:
                loss_shape = loss_shape_per_channel.mean()
            else:
                loss_shape = loss_shape_per_channel.sum(dim=-1).mean()
        return loss_shape

    def loss_components(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if not isinstance(predictions, ProfileCountOutput):
            raise TypeError("PoissonMultinomialLoss requires ProfileCountOutput")
        logits = predictions.logits
        pred_log_counts = predictions.log_counts
        if self.count_per_channel:
            target_counts = targets.sum(dim=2)  # (B, C)
            if pred_log_counts.shape != target_counts.shape:
                raise ValueError(
                    f"count_per_channel=True requires per-channel log_counts "
                    f"{target_counts.shape}, but model predicted "
                    f"{pred_log_counts.shape}. Set predict_total_count=False "
                    f"in model_args when using count_per_channel=True."
                )
            count_loss = self.count_loss_fn(pred_log_counts, target_counts)
        else:
            target_global_count = targets.sum(dim=(1, 2))
            count_loss = self.count_loss_fn(
                pred_log_counts.flatten(), target_global_count
            )
        profile_loss = self._compute_profile_loss(logits, targets)
        return {"profile_loss": profile_loss, "count_loss": count_loss}

    def forward(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(predictions, targets, **kwargs)
        return (
            self.count_weight * components["count_loss"]
            + self.profile_weight * components["profile_loss"]
        )


class CoupledPoissonMultinomialLoss(PoissonMultinomialLoss):
    """
    Poisson Multinomial Loss (Coupled/Global Count).
    Mathematically equivalent to PoissonMultinomialLoss but derives counts from log_rates.

    Objective:
      1. Profile Loss: Cross Entropy.
      2. Count Loss: Poisson NLL on Global Count.

    Accepts ProfileLogRates only. Simulates log_counts via LogSumExp of logits over all channels.
    Does NOT accept ProfileCountOutput (to avoid ambiguity with PoissonMultinomialLoss).
    """

    def loss_components(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if isinstance(predictions, ProfileCountOutput):
            raise TypeError(
                "CoupledPoissonMultinomialLoss does not accept ProfileCountOutput. Use PoissonMultinomialLoss instead."
            )
        if not isinstance(predictions, ProfileLogRates):
            raise TypeError("CoupledPoissonMultinomialLoss requires ProfileLogRates")
        logits = predictions.log_rates
        if self.count_per_channel:
            pred_log_counts = torch.logsumexp(logits.float(), dim=2)
            target_counts = targets.sum(dim=2)
            count_loss = self.count_loss_fn(pred_log_counts, target_counts)
        else:
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1)
            target_global_count = targets.sum(dim=(1, 2))
            count_loss = self.count_loss_fn(pred_log_counts, target_global_count)
        profile_loss = self._compute_profile_loss(logits, targets)
        return {"profile_loss": profile_loss, "count_loss": count_loss}

    def forward(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(predictions, targets, **kwargs)
        return (
            self.count_weight * components["count_loss"]
            + self.profile_weight * components["profile_loss"]
        )


class NegativeBinomialMultinomialLoss(PoissonMultinomialLoss):
    """
    Negative Binomial Multinomial Loss.

    Objective:
      1. Profile Loss: Cross Entropy (Multinomial NLL form).
      2. Count Loss: Negative Binomial NLL on Global Count.

    Args:
        total_count (float): The 'r' dispersion parameter for Negative Binomial.
                             Fixed hyperparameter. Controls saturation of gradients.
                             Smaller 'total_count' implies higher dispersion (variance >> mean).
                             NB Variance = mu + mu^2 / total_count.
    """

    def __init__(self, total_count: float = 10.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.total_count = float(total_count)

    def loss_components(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if not isinstance(predictions, ProfileCountOutput):
            raise TypeError(
                "NegativeBinomialMultinomialLoss requires ProfileCountOutput"
            )
        logits = predictions.logits
        pred_log_counts = predictions.log_counts
        if self.count_per_channel:
            target_counts = targets.sum(dim=2)  # (B, C)
            if pred_log_counts.shape != target_counts.shape:
                raise ValueError(
                    f"count_per_channel=True requires per-channel log_counts "
                    f"{target_counts.shape}, but model predicted "
                    f"{pred_log_counts.shape}. Set predict_total_count=False "
                    f"in model_args when using count_per_channel=True."
                )
        else:
            target_counts = targets.sum(dim=(1, 2))
            if pred_log_counts.ndim > 1:
                pred_log_counts = pred_log_counts.flatten()
        r_tensor = torch.tensor(
            self.total_count, device=pred_log_counts.device, dtype=pred_log_counts.dtype
        )
        nb_logits = pred_log_counts - torch.log(r_tensor)
        nb_dist = torch.distributions.NegativeBinomial(
            total_count=r_tensor, logits=nb_logits
        )
        loss_count = -nb_dist.log_prob(target_counts).mean()
        loss_shape = self._compute_profile_loss(logits, targets)
        return {"profile_loss": loss_shape, "count_loss": loss_count}

    def forward(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(predictions, targets, **kwargs)
        return (
            self.count_weight * components["count_loss"]
            + self.profile_weight * components["profile_loss"]
        )


class CoupledNegativeBinomialMultinomialLoss(NegativeBinomialMultinomialLoss):
    """
    Negative Binomial Multinomial Loss (Coupled).
    Mathematically equivalent to NegativeBinomialMultinomialLoss but derives counts from log_rates.
    """

    def loss_components(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components."""
        targets = targets.float()
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)
        if isinstance(predictions, ProfileCountOutput):
            raise TypeError(
                "CoupledNegativeBinomialMultinomialLoss does not accept ProfileCountOutput. Use NegativeBinomialMultinomialLoss instead."
            )
        if not isinstance(predictions, ProfileLogRates):
            raise TypeError(
                "CoupledNegativeBinomialMultinomialLoss requires ProfileLogRates"
            )
        logits = predictions.log_rates
        if self.count_per_channel:
            pred_log_counts = torch.logsumexp(logits.float(), dim=2)
            target_counts = targets.sum(dim=2)
        else:
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1)
            target_counts = targets.sum(dim=(1, 2))
        r_tensor = torch.tensor(
            self.total_count, device=pred_log_counts.device, dtype=pred_log_counts.dtype
        )
        nb_logits = pred_log_counts - torch.log(r_tensor)
        nb_dist = torch.distributions.NegativeBinomial(
            total_count=r_tensor, logits=nb_logits
        )
        loss_count = -nb_dist.log_prob(target_counts).mean()
        loss_shape = self._compute_profile_loss(logits, targets)
        return {"profile_loss": loss_shape, "count_loss": loss_count}

    def forward(
        self, predictions: object, targets: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        components = self.loss_components(predictions, targets, **kwargs)
        return (
            self.count_weight * components["count_loss"]
            + self.profile_weight * components["profile_loss"]
        )


class DifferentialCountLoss(nn.Module):
    """Phase 2 fine-tuning loss supervising ``log_counts[:, B] - log_counts[:, A]``.

    The delta target is derived inline from the ``(B, N, L)`` ``targets``
    tensor (the per-condition absolute-signal tracks Phase 1 already
    supervised against): for each sample,

    .. math::

        \\Delta_{\\mathrm{target}} = \\log_2\\left(
          \\frac{\\sum_\\ell \\mathrm{targets}[:, B, \\ell] + \\mathrm{pc}}{
                 \\sum_\\ell \\mathrm{targets}[:, A, \\ell] + \\mathrm{pc}}
        \\right)

    where the sum is over the length axis and ``pc`` is
    ``count_pseudocount`` (use the same value Phase 1 used so ``log_counts``
    live in the same log-space). The returned loss is
    ``MSE(log_counts[:, B] - log_counts[:, A], target_delta)``.

    Profile loss is disabled (weight 0) following Naqvi et al. (2025): only
    the count head needs to be retargeted. The profile heads already encode
    condition-specific TF footprint grammar from Phase 1 multi-task
    training.

    Single code path: ``forward(outputs, targets)`` with no optional
    kwargs, no shape overloading, no absolute-count regularizer.

    Args:
        cond_a_idx: Index of condition A in the ``log_counts`` output. Default 0.
        cond_b_idx: Index of condition B in the ``log_counts`` output. Default 1.
        count_pseudocount: Additive pseudocount used in the log2FC
            derivation. Must match the value Phase 1 used so the two
            phases share a log-space. Default 1.0.

    References:
        - Naqvi et al. (2025). *Transfer learning reveals sequence
          determinants of the quantitative response to transcription factor
          dosage.* Cell Genomics. PMC11160683.
        - bpAI-TAC: Chandra et al. (2025). *Refining sequence-to-activity
          models by increasing model resolution.* bioRxiv 2025.01.24.634804.
    """

    uses_count_pseudocount: bool = True

    def __init__(
        self,
        cond_a_idx: int = 0,
        cond_b_idx: int = 1,
        count_pseudocount: float = 1.0,
    ) -> None:
        super().__init__()
        if cond_a_idx == cond_b_idx:
            raise ValueError(
                f"cond_a_idx and cond_b_idx must differ, got both={cond_a_idx}"
            )
        for name, idx in (("cond_a_idx", cond_a_idx), ("cond_b_idx", cond_b_idx)):
            if idx < 0:
                raise ValueError(f"{name} must be non-negative, got {idx}")
        self.cond_a_idx = cond_a_idx
        self.cond_b_idx = cond_b_idx
        self.count_pseudocount = count_pseudocount

    def _delta_loss(
        self, outputs: object, targets: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(outputs, ProfileCountOutput):
            raise TypeError(
                f"DifferentialCountLoss requires ProfileCountOutput, "
                f"got {type(outputs).__name__}"
            )

        log_counts = outputs.log_counts  # (B, N)
        n_channels = log_counts.shape[-1]
        a, b = self.cond_a_idx, self.cond_b_idx
        for name, idx in (("cond_a_idx", a), ("cond_b_idx", b)):
            if idx >= n_channels:
                raise ValueError(
                    f"{name}={idx} is out of range for log_counts with "
                    f"{n_channels} channels"
                )

        if targets.ndim != 3:
            raise ValueError(
                f"DifferentialCountLoss requires targets of shape (B, N, L); "
                f"got {tuple(targets.shape)}"
            )
        n_cond_needed = max(a, b) + 1
        if targets.shape[1] < n_cond_needed:
            raise ValueError(
                f"targets must have at least {n_cond_needed} channels to "
                f"cover cond_a_idx={a} and cond_b_idx={b}, got shape "
                f"{tuple(targets.shape)}"
            )

        counts = targets.float().sum(dim=-1)  # (B, N)
        pc = self.count_pseudocount
        target_delta = torch.log2(
            (counts[:, b] + pc) / (counts[:, a] + pc)
        )  # (B,)
        delta_pred = log_counts[:, b] - log_counts[:, a]  # (B,)
        return F.mse_loss(delta_pred, target_delta)

    def loss_components(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        return {"delta_loss": self._delta_loss(outputs, targets)}

    def forward(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return self._delta_loss(outputs, targets)


class DalmatianLoss(nn.Module):
    """Peak-conditioned factorized loss for the Dalmatian architecture.

    Two loss terms:
      1. L_recon: Combined reconstruction loss on all examples (profile + count).
      2. L_bias: Bias-only reconstruction loss on non-peak (background) examples.

    Gradient separation is handled architecturally (bias outputs are detached
    before combining in Dalmatian.forward), so no explicit signal suppression
    term is needed. Exp21 confirmed that L_signal_bg had zero measurable effect.

    The base loss (e.g. MSEMultinomialLoss) is instantiated internally from
    ``base_loss_cls`` / ``base_loss_args``, enabling nested config via YAML.

    Args:
        base_loss_cls: Fully qualified class name for the base loss
            (e.g. ``"cerberus.loss.MSEMultinomialLoss"``).
        base_loss_args: Keyword arguments forwarded to the base loss constructor.
        bias_weight: Weight for the bias-only reconstruction term.
        count_pseudocount: Forwarded into ``base_loss_args`` (via setdefault)
            so that ``propagate_pseudocount`` works transparently.
    """

    uses_count_pseudocount: bool = True

    def __init__(
        self,
        base_loss_cls: str,
        base_loss_args: dict[str, object] | None = None,
        bias_weight: float = 1.0,
        count_pseudocount: float = 1.0,
        **kwargs: object,
    ):
        super().__init__()
        self.bias_weight = bias_weight

        loss_cls = import_class(base_loss_cls)
        args = dict(base_loss_args or {})
        args.setdefault("count_pseudocount", count_pseudocount)
        self.base_loss: nn.Module = loss_cls(**args)

    def loss_components(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Returns named loss components.

        Args:
            outputs: Model output with combined and decomposed fields.
            targets: Ground-truth target tensor (B, C, L).
            **kwargs: Batch context. Must contain ``interval_source``
                (list of str) with the class name of each interval's
                originating sub-sampler.  Intervals from ``"IntervalSampler"``
                are treated as peaks; all others as background.
        """
        if not isinstance(outputs, FactorizedProfileCountOutput):
            raise TypeError("DalmatianLoss requires FactorizedProfileCountOutput")
        interval_source: list[str] = kwargs["interval_source"]  # type: ignore[assignment]

        # 1. Combined reconstruction loss (all examples)
        combined = ProfileCountOutput(
            logits=outputs.logits,
            log_counts=outputs.log_counts,
        )
        l_recon = self.base_loss(combined, targets)

        # 2. Bias-only reconstruction (non-peak examples)
        non_peak = torch.tensor(
            [s != "IntervalSampler" for s in interval_source],
            dtype=torch.bool,
            device=targets.device,
        )
        l_bias = torch.tensor(0.0, device=targets.device)

        if non_peak.any():
            bias_logits = outputs.bias_logits[non_peak]
            bias_log_counts = outputs.bias_log_counts[non_peak]
            bias_targets = targets[non_peak]
            # shared_bias: 1-channel bias vs N-channel targets — sum targets
            # to get the bulk signal that BiasNet should reconstruct.
            if bias_logits.shape[1] < bias_targets.shape[1]:
                bias_targets = bias_targets.sum(dim=1, keepdim=True)
            bias_out = ProfileCountOutput(
                logits=bias_logits, log_counts=bias_log_counts
            )
            l_bias = self.base_loss(bias_out, bias_targets)

        return {"recon_loss": l_recon, "bias_loss": l_bias}

    def forward(
        self,
        outputs: object,
        targets: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute factorized loss.

        Args:
            outputs: Model output with combined and decomposed fields.
            targets: Ground-truth target tensor (B, C, L).
            **kwargs: Batch context. Must contain ``interval_source``
                (list of str) identifying each interval's originating
                sub-sampler.

        Returns:
            Scalar loss tensor.
        """
        components = self.loss_components(outputs, targets, **kwargs)
        return components["recon_loss"] + self.bias_weight * components["bias_loss"]
