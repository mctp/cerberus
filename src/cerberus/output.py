from dataclasses import dataclass
import torch

@dataclass
class ModelOutput:
    """Base class for model outputs."""
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
        return ProfileLogits(logits=self.logits.detach())

@dataclass
class ProfileLogRates(ModelOutput):
    """
    Output for models predicting log-rates (log-intensities).
    Interpretation: exp(log_rates) = counts.
    """
    log_rates: torch.Tensor # (Batch, Channels, Length)

    def detach(self):
        return ProfileLogRates(log_rates=self.log_rates.detach())

@dataclass
class ProfileCountOutput(ProfileLogits):
    """Output for models predicting profile (logits) and total counts."""
    log_counts: torch.Tensor # (Batch, Channels)

    def detach(self):
        return ProfileCountOutput(logits=self.logits.detach(), log_counts=self.log_counts.detach())
