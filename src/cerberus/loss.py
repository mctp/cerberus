import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef, MeanSquaredError, MetricCollection

class FlattenedPearsonCorrCoef(PearsonCorrCoef):
    """
    PearsonCorrCoef that automatically flattens input dimensions (Batch, Channels, Length) 
    to (Batch*Length, Channels) to compute per-channel correlation, 
    and then averages the result across channels.
    """
    def __init__(self, num_channels=1, **kwargs):
        super().__init__(num_outputs=num_channels, **kwargs)
        self.num_channels = num_channels

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if isinstance(preds, (tuple, list)) and len(preds) == 1:
            preds = preds[0]
            
        # Assumes preds, target: (Batch, Channels, Length)
        # (B, C, L) -> (B, L, C) -> (B*L, C)
        # We ensure channel dimension is last, then flatten batch and length dimensions.
        # This is NOT equivalent to flattening all dimensions and mix channels together.
        preds_flat = preds.detach().permute(0, 2, 1).reshape(-1, self.num_channels)
        target_flat = target.detach().permute(0, 2, 1).reshape(-1, self.num_channels)
        
        super().update(preds_flat, target_flat)

    def compute(self):
        val = super().compute()
        if val.numel() > 1:
            return val.mean()
        return val


class DecoupledFlattenedPearsonCorrCoef(FlattenedPearsonCorrCoef):
    """
    PearsonCorrCoef for BPNet style models returning (logits, log_counts).
    Converts outputs to counts before computing correlation.
    """
    def update(self, preds, target: torch.Tensor):
        # preds[0] = logits (Batch, Channels, Length)
        # preds[1] = log_counts (Batch, Channels)
        logits, log_counts = preds
        probs = F.softmax(logits, dim=-1)
        total_counts = torch.exp(log_counts)
        # Broadcast total_counts: (Batch, Channels) -> (Batch, Channels, 1)
        if total_counts.dim() == 2 and total_counts.shape[1] == 1:
            total_counts = total_counts.expand(-1, self.num_channels)
        elif total_counts.dim() == 1: # (Batch,) case
            total_counts = total_counts.view(-1, 1).expand(-1, self.num_channels)
        
        preds_counts = probs * total_counts.unsqueeze(-1)
        
        super().update(preds_counts, target)


class DecoupledMeanSquaredError(MeanSquaredError):
    """
    MeanSquaredError for BPNet style models returning (logits, log_counts).
    Converts outputs to counts before computing MSE.
    """
    def update(self, preds, target: torch.Tensor):
        # preds[0] = logits (Batch, Channels, Length)
        # preds[1] = log_counts (Batch, Channels)
        logits, log_counts = preds
        probs = F.softmax(logits, dim=-1)
        total_counts = torch.exp(log_counts)
        
        num_channels = logits.shape[1]
        
        # Broadcast total_counts: (Batch, Channels) -> (Batch, Channels, 1)
        if total_counts.dim() == 2 and total_counts.shape[1] == 1:
            total_counts = total_counts.expand(-1, num_channels)
        elif total_counts.dim() == 1: # (Batch,) case
            total_counts = total_counts.view(-1, 1).expand(-1, num_channels)
        
        preds_counts = probs * total_counts.unsqueeze(-1)
        
        super().update(preds_counts, target)


class TupleAwareMeanSquaredError(MeanSquaredError):
    """
    MeanSquaredError that handles tuple inputs by unpacking single-element tuples.
    """
    def update(self, preds, target: torch.Tensor):
        if isinstance(preds, (tuple, list)) and len(preds) == 1:
            preds = preds[0]
        super().update(preds, target)


def get_default_metrics(num_channels: int = 1) -> MetricCollection:
    """
    Returns the default MetricCollection used for training/validation.
    
    Args:
        num_channels: Number of output channels.
    """
    return MetricCollection({
        "pearson": FlattenedPearsonCorrCoef(num_channels=num_channels),
        # MSE is element-wise, so Global MSE is equivalent to Mean Per-Channel MSE
        # (assuming equal number of elements per channel). Thus no custom flattening is needed.
        "mse": TupleAwareMeanSquaredError(),
    })


def get_bpnet_metrics(num_channels: int = 1) -> MetricCollection:
    """
    Returns MetricCollection for BPNet models (using DecoupledFlattenedPearsonCorrCoef).
    """
    return MetricCollection({
        "pearson": DecoupledFlattenedPearsonCorrCoef(num_channels=num_channels),
        "mse": DecoupledMeanSquaredError(),
    })


class TupleAwarePoissonNLLLoss(nn.PoissonNLLLoss):
    """
    PoissonNLLLoss that handles tuple inputs by unpacking single-element tuples.
    """
    def forward(self, log_input, target):
        if isinstance(log_input, (tuple, list)) and len(log_input) == 1:
            log_input = log_input[0]
        return super().forward(log_input, target)


def get_default_loss() -> nn.Module:
    """Returns the default loss function (PoissonNLLLoss)."""
    return TupleAwarePoissonNLLLoss(log_input=True, full=False)


class BPNetLoss(nn.Module):
    """
    Standard BPNet Loss function (matching chrombpnet-pytorch reference).
    
    Decomposes loss into:
    1. Profile Loss: Exact Multinomial Negative Log-Likelihood (including combinatorial terms).
    2. Count Loss: Mean Squared Error on log(total_counts + 1).
    
    Supports both Tuple (Logits, LogCounts) and Tensor (Counts) inputs.
    
    """
    def __init__(self, count_weight=1.0, flatten_channels=False, implicit_log_targets=False, epsilon=1e-8):
        """
        Args:
            count_weight (float): Weight for the count loss component.
            flatten_channels (bool): If True, flattens channels and length for profile loss.
                                   Default False corresponds to independent profile loss per channel (strand).
            implicit_log_targets (bool): If True, assumes targets are log1p transformed.
            epsilon (float): Small constant for numerical stability.
        """
        super().__init__()
        self.count_weight = count_weight
        self.flatten_channels = flatten_channels
        self.implicit_log_targets = implicit_log_targets
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        if self.implicit_log_targets:
            targets = torch.expm1(targets)

        # 1. Determine Logits and LogCounts
        if isinstance(outputs, (tuple, list)):
            logits, pred_log_counts = outputs
        else:
            # Tensor Input (Counts)
            logits = torch.log(outputs + self.epsilon)
            pred_log_counts = torch.log1p(outputs.sum(dim=-1))

        # --- Profile Loss (Exact Multinomial NLL) ---
        if self.flatten_channels:
            logits_flat = logits.flatten(start_dim=1)
            targets_flat = targets.flatten(start_dim=1)
            
            total_counts = targets_flat.sum(dim=-1)
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets_flat + 1), dim=-1)
            
            log_probs = F.log_softmax(logits_flat, dim=-1)
            log_prod_exp = torch.sum(targets_flat * log_probs, dim=-1)
            
            profile_loss = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss.mean()
            
        else:
            total_counts = targets.sum(dim=-1) 
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets + 1), dim=-1) 
            
            log_probs = F.log_softmax(logits, dim=-1)
            log_prod_exp = torch.sum(targets * log_probs, dim=-1)
            
            profile_loss_per_channel = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss_per_channel.sum(dim=-1).mean()

        # --- Count Loss (MSE) ---
        if pred_log_counts.dim() == 2 and pred_log_counts.shape[1] > 1:
            true_counts = targets.sum(dim=-1)
            true_log_counts = torch.log1p(true_counts)
            count_loss = F.mse_loss(pred_log_counts, true_log_counts)
        else:
            true_total_counts = targets.sum(dim=(1, 2))
            true_log_counts = torch.log1p(true_total_counts)
            pred_log_counts = pred_log_counts.flatten()
            count_loss = F.mse_loss(pred_log_counts, true_log_counts)
        
        return profile_loss + self.count_weight * count_loss


class PoissonMultinomialLoss(nn.Module):
    """
    Poisson Multinomial Loss.
    
    Decomposes loss into:
    1. Profile Loss: Multinomial Negative Log-Likelihood (Cross-Entropy form).
    2. Count Loss: Poisson Negative Log-Likelihood.
    
    Supports both Tuple (Logits, LogCounts) and Tensor (Counts) inputs.
    
    """
    def __init__(self, count_weight=0.2, flatten_channels=False, implicit_log_targets=False, epsilon=1e-8):
        """
        Args:
            count_weight (float): Weight for the count loss component.
            flatten_channels (bool): If True, flattens channels and length for profile loss.
            implicit_log_targets (bool): If True, assumes targets are log1p transformed.
            epsilon (float): Small constant for numerical stability.
        """
        super().__init__()
        self.count_weight = count_weight
        self.flatten_channels = flatten_channels
        self.implicit_log_targets = implicit_log_targets
        self.epsilon = epsilon
        self.count_loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)
        self.count_loss_fn_linear = nn.PoissonNLLLoss(log_input=False, full=False)

    def forward(self, predictions, targets):
        if self.implicit_log_targets:
            targets = torch.expm1(targets)

        # 1. Determine Logits and LogCounts
        if isinstance(predictions, (tuple, list)):
            logits, log_counts = predictions
            count_true = targets.sum(dim=-1)
            loss_count = self.count_loss_fn(log_counts, count_true)
        else:
            # Tensor Input (Counts)
            logits = torch.log(predictions + self.epsilon)
            count_pred = predictions.sum(dim=-1)
            count_true = targets.sum(dim=-1)
            loss_count = self.count_loss_fn_linear(count_pred, count_true)

        # --- Profile Loss (Cross Entropy) ---
        if self.flatten_channels:
             logits_flat = logits.flatten(start_dim=1)
             targets_flat = targets.flatten(start_dim=1)
             log_probs = F.log_softmax(logits_flat, dim=-1)
             loss_shape = -torch.sum(targets_flat * log_probs, dim=-1).mean()
        else:
             log_probs = F.log_softmax(logits, dim=-1)
             loss_shape = -torch.sum(targets * log_probs, dim=-1).mean()

        return self.count_weight * loss_count + loss_shape
