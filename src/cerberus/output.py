from dataclasses import dataclass
import torch
from cerberus.interval import Interval

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
