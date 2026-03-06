import torch
import torch.nn as nn
import torch.nn.functional as F

from cerberus.config import import_class
from cerberus.output import FactorizedProfileCountOutput, ProfileCountOutput, ProfileLogRates

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
    def __init__(self, log1p_targets=False, count_pseudocount=1.0, **kwargs):
        super().__init__(**kwargs)
        self.log1p_targets = log1p_targets
        # count_pseudocount is accepted for compatibility with propagate_pseudocount
        # but not used (PoissonNLLLoss handles count loss directly).

    def forward(self, log_input, target):
        target = target.float()
        
        if not isinstance(log_input, ProfileLogRates):
             raise TypeError("ProfilePoissonNLLLoss requires ProfileLogRates")

        log_input = log_input.log_rates

        if self.log1p_targets:
            target = torch.expm1(target).clamp_min(0.0)

        return super().forward(log_input, target)


class MSEMultinomialLoss(nn.Module):
    """
    Multinomial NLL Profile Loss + MSE Count Loss.
    Also known as BPNet Loss.
    
    Objective:
      1. Profile Loss: Multinomial NLL (using logits as unnormalized log-probs).
      2. Count Loss: MSE of log(global_count) (Sum over all channels and length).
      
    Requires ProfileCountOutput (logits, log_counts).
    """
    def __init__(self, count_weight=1.0, profile_weight=1.0, flatten_channels=False, count_per_channel=False, average_channels=False, log1p_targets=False, epsilon=1e-8, count_pseudocount=1.0):
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

    def _compute_profile_loss(self, logits, targets):
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

    def forward(self, outputs, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if not isinstance(outputs, ProfileCountOutput):
             raise TypeError("MSEMultinomialLoss requires ProfileCountOutput")

        logits = outputs.logits
        pred_log_counts = outputs.log_counts

        # --- Profile Loss ---
        profile_loss = self._compute_profile_loss(logits, targets)

        # --- Count Loss (MSE) ---
        if self.count_per_channel:
            target_counts = targets.sum(dim=2) # (B, C)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
            # pred_log_counts should be (B, C)
            count_loss = F.mse_loss(pred_log_counts, target_log_counts)
        else:
            target_global_count = targets.sum(dim=(1, 2))
            target_log_global_count = torch.log(target_global_count + self.count_pseudocount)
            pred_log_counts = pred_log_counts.flatten()
            count_loss = F.mse_loss(pred_log_counts, target_log_global_count)
        
        return self.profile_weight * profile_loss + self.count_weight * count_loss


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
    def forward(self, outputs, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if isinstance(outputs, ProfileCountOutput):
            raise TypeError("CoupledMSEMultinomialLoss does not accept ProfileCountOutput. Use MSEMultinomialLoss instead.")

        if not isinstance(outputs, ProfileLogRates):
             raise TypeError("CoupledMSEMultinomialLoss requires ProfileLogRates")
        
        logits = outputs.log_rates

        if self.count_per_channel:
            # Simulate log_counts per channel (Sum over Length)
            pred_log_counts = torch.logsumexp(logits.float(), dim=2) # (B, C)
            target_counts = targets.sum(dim=2) # (B, C)
            target_log_counts = torch.log(target_counts + self.count_pseudocount)
            count_loss = F.mse_loss(pred_log_counts, target_log_counts)
        else:
            # Simulate log_counts from logits (Global Sum)
            # Flatten channels and length to sum over everything
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1) # (B,)
            target_global_count = targets.sum(dim=(1, 2))
            target_log_global_count = torch.log(target_global_count + self.count_pseudocount)
            count_loss = F.mse_loss(pred_log_counts, target_log_global_count)
        
        # --- Profile Loss ---
        profile_loss = self._compute_profile_loss(logits, targets)
        
        return self.profile_weight * profile_loss + self.count_weight * count_loss


class PoissonMultinomialLoss(nn.Module):
    """
    Poisson Multinomial Loss (Global Count).
    
    Objective:
      1. Profile Loss: Cross Entropy (Multinomial NLL form).
      2. Count Loss: Poisson NLL on Global Count.
      
    Requires ProfileCountOutput.
    """
    def __init__(self, count_weight=0.2, profile_weight=1.0, flatten_channels=False, count_per_channel=False, average_channels=True, log1p_targets=False, epsilon=1e-8, count_pseudocount=1.0):
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

    def _compute_profile_loss(self, logits, targets):
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

    def forward(self, predictions, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if not isinstance(predictions, ProfileCountOutput):
             raise TypeError("PoissonMultinomialLoss requires ProfileCountOutput")

        logits = predictions.logits
        pred_log_counts = predictions.log_counts
        
        if self.count_per_channel:
            target_counts = targets.sum(dim=2) # (B, C)
            # pred_log_counts should be (B, C)
            loss_count = self.count_loss_fn(pred_log_counts, target_counts)
        else:
            # --- Count Loss (Global Count) ---
            target_global_count = targets.sum(dim=(1, 2)) # (B,)
            pred_log_counts = pred_log_counts.flatten() # (B,)
            loss_count = self.count_loss_fn(pred_log_counts, target_global_count)

        # --- Profile Loss ---
        loss_shape = self._compute_profile_loss(logits, targets)

        return self.count_weight * loss_count + self.profile_weight * loss_shape


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
    def forward(self, predictions, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if isinstance(predictions, ProfileCountOutput):
            raise TypeError("CoupledPoissonMultinomialLoss does not accept ProfileCountOutput. Use PoissonMultinomialLoss instead.")

        if not isinstance(predictions, ProfileLogRates):
             raise TypeError("CoupledPoissonMultinomialLoss requires ProfileLogRates")

        logits = predictions.log_rates

        if self.count_per_channel:
            # Simulate log_counts per channel
            pred_log_counts = torch.logsumexp(logits.float(), dim=2) # (B, C)
            target_counts = targets.sum(dim=2) # (B, C)
            loss_count = self.count_loss_fn(pred_log_counts, target_counts)
        else:
            # Simulate log_counts from logits (Global Sum)
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1) # (B,)
            target_global_count = targets.sum(dim=(1, 2)) # (B,)
            loss_count = self.count_loss_fn(pred_log_counts, target_global_count)

        # --- Profile Loss ---
        loss_shape = self._compute_profile_loss(logits, targets)

        return self.count_weight * loss_count + self.profile_weight * loss_shape


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
    def __init__(self, total_count=10.0, **kwargs):
        super().__init__(**kwargs)
        self.total_count = float(total_count)
        
    def forward(self, predictions, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if not isinstance(predictions, ProfileCountOutput):
             raise TypeError("NegativeBinomialMultinomialLoss requires ProfileCountOutput")

        logits = predictions.logits
        pred_log_counts = predictions.log_counts
        
        # Determine target counts
        if self.count_per_channel:
            target_counts = targets.sum(dim=2) # (B, C)
        else:
            target_counts = targets.sum(dim=(1, 2)) # (B,)
            if pred_log_counts.ndim > 1:
                pred_log_counts = pred_log_counts.flatten()
        
        # --- Count Loss (Negative Binomial) ---
        # pred_log_counts is log(mu)
        # We need to construct NB distribution
        # PyTorch NB parameterization: total_count (r), logits (log-odds)
        # logits = pred_log_counts - log(r)
        
        r_tensor = torch.tensor(self.total_count, device=pred_log_counts.device, dtype=pred_log_counts.dtype)
        nb_logits = pred_log_counts - torch.log(r_tensor)
        
        nb_dist = torch.distributions.NegativeBinomial(total_count=r_tensor, logits=nb_logits)
        loss_count = -nb_dist.log_prob(target_counts).mean()

        # --- Profile Loss ---
        loss_shape = self._compute_profile_loss(logits, targets)

        return self.count_weight * loss_count + self.profile_weight * loss_shape


class CoupledNegativeBinomialMultinomialLoss(NegativeBinomialMultinomialLoss):
    """
    Negative Binomial Multinomial Loss (Coupled).
    Mathematically equivalent to NegativeBinomialMultinomialLoss but derives counts from log_rates.
    """
    def forward(self, predictions, targets):
        targets = targets.float()
        
        if self.log1p_targets:
            targets = torch.expm1(targets).clamp_min(0.0)

        if isinstance(predictions, ProfileCountOutput):
            raise TypeError("CoupledNegativeBinomialMultinomialLoss does not accept ProfileCountOutput. Use NegativeBinomialMultinomialLoss instead.")

        if not isinstance(predictions, ProfileLogRates):
             raise TypeError("CoupledNegativeBinomialMultinomialLoss requires ProfileLogRates")

        logits = predictions.log_rates

        if self.count_per_channel:
            # Simulate log_counts per channel
            pred_log_counts = torch.logsumexp(logits.float(), dim=2) # (B, C)
            target_counts = targets.sum(dim=2) # (B, C)
        else:
            # Simulate log_counts from logits (Global Sum)
            logits_flat = logits.flatten(start_dim=1)
            pred_log_counts = torch.logsumexp(logits_flat.float(), dim=-1) # (B,)
            target_counts = targets.sum(dim=(1, 2)) # (B,)

        # --- Count Loss (Negative Binomial) ---
        r_tensor = torch.tensor(self.total_count, device=pred_log_counts.device, dtype=pred_log_counts.dtype)
        nb_logits = pred_log_counts - torch.log(r_tensor)
        
        nb_dist = torch.distributions.NegativeBinomial(total_count=r_tensor, logits=nb_logits)
        loss_count = -nb_dist.log_prob(target_counts).mean()

        # --- Profile Loss ---
        loss_shape = self._compute_profile_loss(logits, targets)

        return self.count_weight * loss_count + self.profile_weight * loss_shape


class DalmatianLoss(nn.Module):
    """Peak-conditioned factorized loss for the Dalmatian architecture.

    Three loss terms:
      1. L_recon: Combined reconstruction loss on all examples (profile + count).
      2. L_bias: Bias-only reconstruction loss on non-peak (background) examples.
      3. L_signal_bg: Signal suppression penalty on non-peak examples --
         L1 on signal logits and log_counts, pushing them toward zero
         (the identity elements for logit addition and logsumexp).

    The base loss (e.g. MSEMultinomialLoss) is instantiated internally from
    ``base_loss_cls`` / ``base_loss_args``, enabling nested config via YAML.

    Args:
        base_loss_cls: Fully qualified class name for the base loss
            (e.g. ``"cerberus.loss.MSEMultinomialLoss"``).
        base_loss_args: Keyword arguments forwarded to the base loss constructor.
        bias_weight: Weight for the bias-only reconstruction term.
        signal_background_weight: Weight for the signal suppression term on
            background regions.
        count_pseudocount: Forwarded into ``base_loss_args`` (via setdefault)
            so that ``propagate_pseudocount`` works transparently.
    """

    def __init__(
        self,
        base_loss_cls: str,
        base_loss_args: dict[str, object] | None = None,
        bias_weight: float = 1.0,
        signal_background_weight: float = 0.1,
        count_pseudocount: float = 1.0,
    ):
        super().__init__()
        self.bias_weight = bias_weight
        self.signal_background_weight = signal_background_weight

        loss_cls = import_class(base_loss_cls)
        args = dict(base_loss_args or {})
        args.setdefault("count_pseudocount", count_pseudocount)
        self.base_loss: nn.Module = loss_cls(**args)

    def forward(
        self,
        output: FactorizedProfileCountOutput,
        target: torch.Tensor,
        peak_status: torch.Tensor,
    ) -> torch.Tensor:
        """Compute factorized loss.

        Args:
            output: Model output with combined and decomposed fields.
            target: Ground-truth target tensor (B, C, L).
            peak_status: Per-example indicator -- 1 for peak, 0 for background.

        Returns:
            Scalar loss tensor.
        """
        # 1. Combined reconstruction loss (all examples)
        combined = ProfileCountOutput(
            logits=output.logits, log_counts=output.log_counts,
        )
        l_recon = self.base_loss(combined, target)

        # 2. Bias-only reconstruction + signal suppression (non-peak examples)
        non_peak = peak_status == 0
        l_bias = torch.tensor(0.0, device=target.device)
        l_signal_bg = torch.tensor(0.0, device=target.device)

        if non_peak.any():
            bias_out = ProfileCountOutput(
                logits=output.bias_logits[non_peak],
                log_counts=output.bias_log_counts[non_peak],
            )
            l_bias = self.base_loss(bias_out, target[non_peak])

            l_signal_bg = (
                output.signal_logits[non_peak].abs().mean()
                + output.signal_log_counts[non_peak].abs().mean()
            )

        return l_recon + self.bias_weight * l_bias + self.signal_background_weight * l_signal_bg
