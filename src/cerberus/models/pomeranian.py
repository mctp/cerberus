import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from cerberus.output import ProfileCountOutput
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError
from cerberus.layers import ConvNeXtV2Block, PGCBlock

class Pomeranian(nn.Module):
    """
    Pomeranian: A lightweight model mirroring BPNet (valid padding) but using GemiNet components.
    
    Structure:
    - Input: One-hot sequence (Batch, 4, Length)
    - Stem: ConvNeXtV2 Block (Valid Padding)
    - Body: Stack of Dilated PGC Blocks (Valid Padding)
    - Head 1 (Profile): Decoupled Head (Pointwise -> GELU -> Spatial Valid)
    - Head 2 (Counts): MLP Head (Global Avg Pool -> Linear -> GELU -> Linear -> Log(Total Counts))
    
    Notes:
    - Kernel sizes should be ODD numbers to ensure symmetric padding/cropping and perfect alignment.
    - Output Length Formula:
      Output = Input - (Stem_Shrinkage + Tower_Shrinkage + Head_Shrinkage)
      
      Where:
      - Stem_Shrinkage = conv_kernel_size - 1
      - Tower_Shrinkage = Sum((dil_kernel_size - 1) * dilation) for all layers
      - Head_Shrinkage = profile_kernel_size - 1
      
      For defaults (8 layers, dilations 2^1..2^8, k=3):
      - Tower_Shrinkage = Sum(2 * 2^i) = 2 * (2^9 - 2) = 1020
      - Total Shrinkage = (21-1) + 1020 + (49-1) = 20 + 1020 + 48 = 1088
    
    Args:
        input_len (int): Length of input sequence.
        output_len (int): Length of output sequence.
        output_bin_size (int): Output resolution bin size.
        input_channels (list[str]): List of input channel names.
        output_channels (list[str]): List of output channel names.
        filters (int): Model dimension. Default: 64.
        n_dilated_layers (int): Number of dilated PGC blocks. Default: 8.
        conv_kernel_size (int): Kernel size for initial convolution (stem). Default: 21.
        dil_kernel_size (int): Kernel size for dilated convolutions. Default: 3.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 75.
        expansion (int): Expansion factor for PGC blocks. Default: 1.
        dropout (float): Dropout rate. Default: 0.1.
        predict_total_count (bool): If True, predicts a single total count scalar. Default: True.
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 64,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        expansion: int = 1,
        dropout: float = 0.1,
        predict_total_count: bool = True,
    ):
        super().__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.n_input_channels = len(input_channels)
        self.n_output_channels = len(output_channels)
        self.predict_total_count = predict_total_count
        
        # 1. Stem
        # Use ConvNeXtV2Block with valid padding
        self.stem = ConvNeXtV2Block(
            channels_in=self.n_input_channels, 
            channels_out=filters, 
            kernel_size=conv_kernel_size,
            padding='valid'
        )
        
        # 2. Dilated PGC Tower
        self.layers = nn.ModuleList()
        for i in range(1, n_dilated_layers + 1):
            dilation = 2**i
            self.layers.append(
                PGCBlock(
                    dim=filters,
                    kernel_size=dil_kernel_size,
                    dilation=dilation,
                    expansion=expansion,
                    dropout=dropout,
                    padding='valid'
                )
            )
            
        # 3. Profile Head (Decoupled)
        # We use a decoupled design instead of a single large convolution.
        # 1. Pointwise Conv (1x1): Mixes channels to refine features per position without spatial smoothing.
        # 2. GELU: Adds non-linearity to the head (standard BPNet head is linear).
        # 3. Spatial Conv (Valid): Performs the final spatial smoothing/aggregation over the profile window.
        # This separation allows the model to learn complex feature interactions before committing to a specific shape,
        # improving expressivity with minimal parameter cost compared to a single large kernel.
        self.profile_pointwise = nn.Conv1d(filters, filters, kernel_size=1)
        self.profile_act = nn.GELU()
        self.profile_spatial = nn.Conv1d(
            filters, self.n_output_channels, 
            kernel_size=profile_kernel_size, 
            padding='valid'
        )
        
        # 4. Counts Head (MLP)
        num_count_outputs = 1 if self.predict_total_count else self.n_output_channels
        hidden_dim = filters // 2
        self.count_mlp = nn.Sequential(
            nn.Linear(filters, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_count_outputs)
        )

    def forward(self, x) -> ProfileCountOutput:
        # 1. Stem
        x = self.stem(x)
        
        # 2. PGC Layers
        for layer in self.layers:
            x = layer(x)
            
        # --- Profile Head ---
        # Decoupled: Pointwise -> Act -> Spatial
        profile_x = self.profile_pointwise(x)
        profile_x = self.profile_act(profile_x)
        profile_logits = self.profile_spatial(profile_x) # (B, Out_Channels, Length)
        
        # Crop to target output_len if needed
        current_len = profile_logits.shape[-1]
        target_len = self.output_len
        
        if current_len > target_len:
            diff = current_len - target_len
            crop_l = diff // 2
            crop_r = diff - crop_l
            profile_logits = profile_logits[..., crop_l:-crop_r]
        elif current_len < target_len:
            raise ValueError(f"Output length {current_len} is smaller than requested {target_len}")
        
        if self.output_bin_size > 1:
            profile_logits = F.avg_pool1d(
                profile_logits, 
                kernel_size=self.output_bin_size, 
                stride=self.output_bin_size
            )

        # --- Counts Head ---
        # Global Average Pooling over the VALID latent representation x (before profile spatial conv)
        # Note: 'x' here has length equal to Input - (Stem + Tower Shrinkage).
        # Profile logits have length x - Profile Shrinkage.
        # BPNet typically pools over the latent representation that feeds into the profile head.
        # However, to be perfectly aligned with the output region, we might want to crop 'x'
        # to correspond to the output region (center) before pooling.
        # The profile head shrinkage is 74 bp.
        # If we pool 'x' directly, we include 37bp on each side that don't contribute to the valid profile output.
        # Standard BPNet usually pools the input to the profile head directly (GlobalAvgPool).
        # GemiNet also pools x directly.
        
        # Crop x to match the target output length to ensure we pool features
        # corresponding to the predicted region.
        current_len_x = x.shape[-1]
        if current_len_x > target_len:
            diff = current_len_x - target_len
            crop_l = diff // 2
            crop_r = diff - crop_l
            x_for_counts = x[..., crop_l:-crop_r]
        else:
            x_for_counts = x

        x_pooled = x_for_counts.mean(dim=-1) # (B, Filters)
        
        log_counts = self.count_mlp(x_pooled) # (B, Out_Channels)
        
        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)

class Pomeranian1k(Pomeranian):
    """
    Pomeranian1k: A 1024bp output version of Pomeranian aligned to powers of 2.
    Input: 2112 bp. Output: 1024 bp.
    """
    def __init__(
        self,
        input_len: int = 2112,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 64,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 49,
        expansion: int = 1,
        dropout: float = 0.1,
        predict_total_count: bool = True,
    ):
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=input_channels,
            output_channels=output_channels,
            filters=filters,
            n_dilated_layers=n_dilated_layers,
            conv_kernel_size=conv_kernel_size,
            dil_kernel_size=dil_kernel_size,
            profile_kernel_size=profile_kernel_size,
            expansion=expansion,
            dropout=dropout,
            predict_total_count=predict_total_count,
        )


class PomeranianMetricCollection(MetricCollection):
    """
    MetricCollection for Pomeranian models.
    """
    def __init__(self, num_channels: int = 1, implicit_log_targets: bool = False):
        super().__init__({
            "pearson": CountProfilePearsonCorrCoef(num_channels=num_channels, implicit_log_targets=implicit_log_targets),
            "mse_profile": CountProfileMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "mse_log_counts": LogCountsMeanSquaredError(implicit_log_targets=implicit_log_targets),
        })
