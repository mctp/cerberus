from typing import Protocol

import torch
import torch.nn.functional as F

from .config import DataConfig
from .interval import Interval


class DataTransform(Protocol):
    """
    Protocol for data transformations.
    
    Transforms operate on the input/target tensors and the interval metadata.
    They must return the modified inputs, targets, and interval.
    """
    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]: ...


class Compose:
    """
    Composes multiple transforms together.
    
    Sequentially applies a list of transforms to the data.
    """
    def __init__(self, transforms: list[DataTransform]):
        """
        Args:
            transforms: List of DataTransform objects to apply in order.
        """
        self.transforms = transforms

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        for t in self.transforms:
            inputs, targets, interval = t(inputs, targets, interval)
        return inputs, targets, interval


class Jitter:
    """
    Randomly crops input/target tensors to input_len.
    
    This augmentation simulates random shifting of the genomic window.
    
    IMPORTANT: This transform updates the `interval.start` and `interval.end` 
    attributes of the passed Interval object to reflect the new cropped region.
    Downstream components should use these updated coordinates.
    """

    def __init__(self, input_len: int, max_jitter: int | None = None):
        """
        Args:
            input_len: Desired length of output tensors (model input length).
            max_jitter: Max deviation from center in bp. If None, uses full available slack.
        """
        self.input_len = input_len
        self.max_jitter = max_jitter

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        """
        Applies jitter crop.
        
        Args:
            inputs: Tensor of shape (Channels, Current_Length).
            targets: Tensor of shape (Target_Channels, Current_Length) or different.
                     If targets length matches inputs, it is also cropped.
            interval: Genomic interval.
            
        Returns:
            Tuple of (cropped_inputs, cropped_targets, updated_interval).
        """
        current_len = inputs.shape[-1]
        slack = current_len - self.input_len

        if self.max_jitter is not None:
            center = slack // 2
            start_min = max(0, center - self.max_jitter)
            start_max = min(slack, center + self.max_jitter)
        else:
            start_min = 0
            start_max = slack

        start = torch.randint(start_min, start_max + 1, (1,)).item()
        end = start + self.input_len

        inputs = inputs[..., start:end]

        if targets.shape[-1] == current_len:
            targets = targets[..., start:end]
            
        # Update Interval
        # The tensor corresponds to [interval.start, interval.end)
        # We cropped by 'start' from the left.
        
        # New genomic start = old genomic start + crop offset
        interval.start = interval.start + start  # type: ignore
        # New genomic end = new genomic start + input_len
        interval.end = interval.start + self.input_len

        return inputs, targets, interval


class TargetCrop:
    """
    Crops target tensors to the center to match output_len.
    Used when the model output is smaller than the input (e.g., valid padding convolutions).
    """

    def __init__(self, output_len: int):
        """
        Args:
            output_len: Desired length of target tensors.
        """
        self.output_len = output_len

    def _crop(self, tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.shape[-1]
        if length <= self.output_len:
            return tensor

        start = (length - self.output_len) // 2
        end = start + self.output_len
        return tensor[..., start:end]

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        targets = self._crop(targets)
        return inputs, targets, interval


class ReverseComplement:
    """
    Randomly reverse complements the sequence and reverses the signal.
    Updates interval strand.
    
    Assumes inputs are (Channels, Length) where first 4 channels are DNA (ACGT).
    """

    def __init__(
        self, probability: float = 0.5, dna_channels: slice | list[int] = slice(0, 4)
    ):
        """
        Args:
            probability: Probability of applying the transformation.
            dna_channels: Slice or list of indices indicating DNA channels.
                          Assumes channel order corresponds to ACGT (or similar symmetric mapping).
        """
        self.probability = probability
        self.dna_channels = dna_channels

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if torch.rand(1).item() > self.probability:
            return inputs, targets, interval

        # Reverse Length
        inputs = torch.flip(inputs, dims=[-1])
        targets = torch.flip(targets, dims=[-1])

        # Complement DNA (Flip Channels)
        # Reversing ACGT (0123) -> TGCA (3210) maps A->T, C->G.
        if isinstance(self.dna_channels, slice):
            dna = inputs[self.dna_channels]
            inputs[self.dna_channels] = torch.flip(dna, dims=[-2])
            
        # Flip strand
        interval.strand = "-" if interval.strand == "+" else "+"

        return inputs, targets, interval


class Log1p:
    """
    Applies log1p transformation (log(x + 1)).
    """

    def __init__(self, apply_to: str = "targets", safe_check: bool = False):
        """
        Args:
            apply_to: Which tensor to apply transform to: 'inputs', 'targets', or 'both'.
            safe_check: If True, checks for negative values before log.
        """
        self.apply_to = apply_to
        self.safe_check = safe_check

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if self.apply_to in ("inputs", "both"):
            if self.safe_check and (inputs < 0).any():
                raise ValueError("Log1p input contains negative values")
            inputs = torch.log1p(inputs)

        if self.apply_to in ("targets", "both"):
            if self.safe_check and (targets < 0).any():
                raise ValueError("Log1p target contains negative values")
            targets = torch.log1p(targets)

        return inputs, targets, interval


class Sqrt:
    """
    Applies sqrt transformation.
    """

    def __init__(self, apply_to: str = "targets", safe_check: bool = False):
        """
        Args:
            apply_to: Which tensor to apply transform to: 'inputs', 'targets', or 'both'.
            safe_check: If True, checks for negative values before sqrt.
        """
        self.apply_to = apply_to
        self.safe_check = safe_check

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if self.apply_to in ("inputs", "both"):
            if self.safe_check and (inputs < 0).any():
                raise ValueError("Sqrt input contains negative values")
            inputs = torch.sqrt(inputs)

        if self.apply_to in ("targets", "both"):
            if self.safe_check and (targets < 0).any():
                raise ValueError("Sqrt target contains negative values")
            targets = torch.sqrt(targets)

        return inputs, targets, interval


class Arcsinh:
    """
    Applies arcsinh transformation (log(x + sqrt(x^2 + 1))).
    """

    def __init__(self, apply_to: str = "targets"):
        """
        Args:
            apply_to: Which tensor to apply transform to: 'inputs', 'targets', or 'both'.
        """
        self.apply_to = apply_to

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if self.apply_to in ("inputs", "both"):
            inputs = torch.arcsinh(inputs)

        if self.apply_to in ("targets", "both"):
            targets = torch.arcsinh(targets)

        return inputs, targets, interval


class Scale:
    """
    Multiplies targets (or inputs) by a constant factor.
    Useful for rescaling fractional/normalized BigWig signal to integer-like counts.
    """

    def __init__(self, factor: float, apply_to: str = "targets"):
        self.factor = factor
        self.apply_to = apply_to

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if self.apply_to in ("inputs", "both"):
            inputs = inputs * self.factor
        if self.apply_to in ("targets", "both"):
            targets = targets * self.factor
        return inputs, targets, interval


class Bin:
    """
    Bins the signal by pooling.
    Reduces resolution by `bin_size`.
    """

    def __init__(self, bin_size: int, method: str = "max", apply_to: str = "targets"):
        """
        Args:
            bin_size: Size of the bin.
            method: Pooling method: 'max', 'avg', or 'sum'.
            apply_to: Which tensor to apply transform to: 'inputs', 'targets', or 'both'.
        """
        valid_methods = ("max", "avg", "sum")
        if method not in valid_methods:
            raise ValueError(
                f"Bin method must be one of {valid_methods}, got {method!r}"
            )
        self.bin_size = bin_size
        self.method = method
        self.apply_to = apply_to

    def _bin(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.unsqueeze(0)  # (1, C, L)

        if self.method == "max":
            x = F.max_pool1d(x, kernel_size=self.bin_size, stride=self.bin_size)
        elif self.method == "avg":
            x = F.avg_pool1d(x, kernel_size=self.bin_size, stride=self.bin_size)
        elif self.method == "sum":
            x = (
                F.avg_pool1d(x, kernel_size=self.bin_size, stride=self.bin_size)
                * self.bin_size
            )

        return x.squeeze(0)

    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, interval: Interval
    ) -> tuple[torch.Tensor, torch.Tensor, Interval]:
        if self.apply_to in ("inputs", "both"):
            inputs = self._bin(inputs)
        if self.apply_to in ("targets", "both"):
            targets = self._bin(targets)
        return inputs, targets, interval


def create_default_transforms(
    data_config: DataConfig, deterministic: bool = False
) -> list[DataTransform]:
    """
    Creates a list of default transforms based on DataConfig options.
    
    Standard pipeline:
    1. Jitter
    2. Reverse Complement
    3. Target Crop
    4. Binning
    5. Log Transform

    Args:
        data_config: Dictionary containing transform parameters.
        deterministic: If True, disables random augmentations (jitter=0, no RC).
    
    Returns:
        list[DataTransform]: List of instantiated transforms.
    """
    transforms: list[DataTransform] = []

    # 1. Jitter (random)
    # If deterministic, force max_jitter=0 (center crop)
    max_jitter = 0 if deterministic else data_config.max_jitter
    transforms.append(Jitter(input_len=data_config.input_len, max_jitter=max_jitter))

    # 2. Reverse Complement (random)
    # If deterministic, skip RC
    if data_config.reverse_complement and not deterministic:
        transforms.append(ReverseComplement(dna_channels=slice(0, 4)))

    # 3. Target Cropping (deterministic)
    if data_config.output_len < data_config.input_len:
        transforms.append(TargetCrop(output_len=data_config.output_len))

    # 4. Target Scaling (deterministic)
    target_scale = data_config.target_scale
    if target_scale != 1.0:
        transforms.append(Scale(factor=target_scale, apply_to="targets"))

    # 5. Binning (deterministic)
    if data_config.output_bin_size > 1:
        transforms.append(Bin(bin_size=data_config.output_bin_size, apply_to="targets"))

    # 6. Log Transform (deterministic)
    if data_config.log_transform:
        transforms.append(Log1p(apply_to="targets"))

    return transforms
