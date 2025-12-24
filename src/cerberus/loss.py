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
        "mse": MeanSquaredError(),
    })


def get_default_loss() -> nn.Module:
    """Returns the default loss function (PoissonNLLLoss)."""
    return nn.PoissonNLLLoss(log_input=False, full=False)


class BPNetLoss(nn.Module):
    """
    BPNet Loss function combining Multinomial NLL for profile and MSE for counts.
    
    Follows the implementation in bpnet-lite:
    1. Profile Loss: Multinomial Negative Log-Likelihood with exact combinatorial terms (lgamma).
       Flattens channels and length to compute a single multinomial distribution per example.
    2. Count Loss: Mean Squared Error on log(total_counts + 1).
    """
    def __init__(self, alpha=1.0, flatten_channels=True, implicit_log_targets=False):
        """
        Args:
            alpha (float): Weight for the count loss. Total loss = profile_loss + alpha * count_loss.
            flatten_channels (bool): If True, flattens channels and length dimensions for profile loss 
                                     (single multinomial over all tracks). 
                                     If False, computes multinomial over length dimension for each channel independently.
                                     BPNet-lite defaults to True (single softmax over strands).
            implicit_log_targets (bool): If True, assumes targets have been log1p transformed (log(x+1))
                                         and reverses this transform (expm1) before computing loss.
                                         Useful if data pipeline includes Log1p transform.
        """
        super().__init__()
        self.alpha = alpha
        self.flatten_channels = flatten_channels
        self.implicit_log_targets = implicit_log_targets

    def forward(self, outputs, targets):
        """
        Args:
            outputs (tuple): (profile_logits, pred_log_counts)
                profile_logits: (Batch, Channels, Length)
                pred_log_counts: (Batch, Channels) or (Batch, 1) depending on model. 
                                 If (Batch, Channels), we might sum them or expect model to output total log counts.
                                 BPNet-lite predicts a single log-count scalar for the total sum.
            targets (torch.Tensor): Observed counts (Batch, Channels, Length).
        
        Returns:
            loss: scalar
        """
        if self.implicit_log_targets:
            targets = torch.expm1(targets)

        profile_logits, pred_log_counts = outputs
        
        # --- Profile Loss ---
        if self.flatten_channels:
            # Flatten channels and length: (Batch, Channels * Length)
            # This matches BPNet-lite behavior: "single log softmax ... across both strands"
            logits_flat = profile_logits.flatten(start_dim=1)
            targets_flat = targets.flatten(start_dim=1)
            
            # Multinomial NLL
            # log_prob(k) = log(N!) - sum(log(k_i!)) + sum(k_i * log(p_i))
            # Loss = -log_prob
            
            total_counts = targets_flat.sum(dim=-1)
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets_flat + 1), dim=-1)
            
            log_probs = F.log_softmax(logits_flat, dim=-1)
            log_prod_exp = torch.sum(targets_flat * log_probs, dim=-1)
            
            profile_loss = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss.mean()
            
        else:
            # Independent multinomial per channel
            # logits: (Batch, Channels, Length)
            
            total_counts = targets.sum(dim=-1) # (Batch, Channels)
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets + 1), dim=-1) # sum over length
            
            log_probs = F.log_softmax(profile_logits, dim=-1) # softmax over length
            log_prod_exp = torch.sum(targets * log_probs, dim=-1)
            
            profile_loss_per_channel = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss_per_channel.sum(dim=-1).mean() # Sum over channels, mean over batch

        # --- Count Loss ---
        # BPNet-lite predicts total counts across all strands.
        # "The count prediction task is predicting the total counts across both strands."
        
        # Calculate true total log counts
        # targets: (Batch, Channels, Length) -> sum over all dims except batch
        true_total_counts = targets.sum(dim=(1, 2)) # (Batch,)
        true_log_counts = torch.log1p(true_total_counts) # log(x+1)
        
        # pred_log_counts might be (Batch, 1) or (Batch,)
        pred_log_counts = pred_log_counts.flatten()
        
        count_loss = F.mse_loss(pred_log_counts, true_log_counts)
        
        # --- Total Loss ---
        loss = profile_loss + self.alpha * count_loss
        
        return loss


class BPNetPoissonLoss(nn.Module):
    """
    BPNet-style Loss function combining Multinomial NLL for profile and Poisson NLL for total counts.
    
    1. Profile Loss: Multinomial Negative Log-Likelihood (same as BPNetLoss).
    2. Count Loss: Poisson Negative Log-Likelihood on total counts.
    """
    def __init__(self, alpha=1.0, flatten_channels=True, implicit_log_targets=False):
        """
        Args:
            alpha (float): Weight for the count loss. Total loss = profile_loss + alpha * count_loss.
            flatten_channels (bool): If True, flattens channels and length dimensions for profile loss.
            implicit_log_targets (bool): If True, assumes targets have been log1p transformed.
        """
        super().__init__()
        self.alpha = alpha
        self.flatten_channels = flatten_channels
        self.implicit_log_targets = implicit_log_targets
        self.count_loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)

    def forward(self, outputs, targets):
        """
        Args:
            outputs (tuple): (profile_logits, pred_log_counts)
                profile_logits: (Batch, Channels, Length)
                pred_log_counts: (Batch, Channels) or (Batch, 1). Interpreted as log(total_counts).
            targets (torch.Tensor): Observed counts (Batch, Channels, Length).
        
        Returns:
            loss: scalar
        """
        if self.implicit_log_targets:
            targets = torch.expm1(targets)

        profile_logits, pred_log_counts = outputs
        
        # --- Profile Loss ---
        if self.flatten_channels:
            # Flatten channels and length: (Batch, Channels * Length)
            logits_flat = profile_logits.flatten(start_dim=1)
            targets_flat = targets.flatten(start_dim=1)
            
            total_counts = targets_flat.sum(dim=-1)
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets_flat + 1), dim=-1)
            
            log_probs = F.log_softmax(logits_flat, dim=-1)
            log_prod_exp = torch.sum(targets_flat * log_probs, dim=-1)
            
            profile_loss = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss.mean()
            
        else:
            # Independent multinomial per channel
            total_counts = targets.sum(dim=-1) # (Batch, Channels)
            
            log_fact_sum = torch.lgamma(total_counts + 1)
            log_prod_fact = torch.sum(torch.lgamma(targets + 1), dim=-1) 
            
            log_probs = F.log_softmax(profile_logits, dim=-1)
            log_prod_exp = torch.sum(targets * log_probs, dim=-1)
            
            profile_loss_per_channel = -log_fact_sum + log_prod_fact - log_prod_exp
            profile_loss = profile_loss_per_channel.sum(dim=-1).mean()

        # --- Count Loss ---
        # Calculate true total counts
        true_total_counts = targets.sum(dim=(1, 2)) # (Batch,)
        
        pred_log_counts = pred_log_counts.flatten()
        
        count_loss = self.count_loss_fn(pred_log_counts, true_total_counts)
        
        # --- Total Loss ---
        loss = profile_loss + self.alpha * count_loss
        
        return loss
