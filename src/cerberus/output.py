from dataclasses import dataclass
import torch

@dataclass
class ModelOutput:
    """Base class for model outputs."""
    def detach(self):
        """Returns a new instance with all tensors detached from the graph."""
        raise NotImplementedError

@dataclass
class ProfileOutput(ModelOutput):
    """Output for models predicting only a profile (shape)."""
    logits: torch.Tensor # (Batch, Channels, Length)

    def detach(self):
        return ProfileOutput(logits=self.logits.detach())

@dataclass
class ProfileCountOutput(ProfileOutput):
    """Output for models predicting profile and total counts."""
    log_counts: torch.Tensor # (Batch, Channels)

    def detach(self):
        return ProfileCountOutput(logits=self.logits.detach(), log_counts=self.log_counts.detach())
