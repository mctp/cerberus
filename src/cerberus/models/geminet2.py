import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from cerberus.output import ProfileCountOutput
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError, LogCountsPearsonCorrCoef
from cerberus.layers import PGCBlock, ConvNeXtV2Block

class GemiNet2(nn.Module):
    """
    GemiNet2: A variant of GemiNet using ConvNeXtV2 for the stem.
    
    Architecture:
    - Input: One-hot sequence (Batch, 4, Length)
    - Stem: ConvNeXtV2Block (captures motifs and local features)
    - Body: Stack of Dilated PGC Blocks
    - Head 1 (Profile): Conv1D -> Logits
    - Head 2 (Counts): Global Avg Pool -> Dense -> Log(Total Counts)
    
    Args:
        input_len (int): Length of input sequence.
        output_len (int): Length of output sequence.
        output_bin_size (int): Output resolution bin size.
        input_channels (list[str]): List of input channel names.
        output_channels (list[str]): List of output channel names.
        filters (int): Model dimension. Default: 64.
        n_dilated_layers (int): Number of dilated PGC blocks. Default: 8.
        conv_kernel_size (int): Kernel size for initial convolution. Default: 21.
        dil_kernel_size (int): Kernel size for dilated convolutions. Default: 3.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 75.
        expansion (int): Expansion factor for PGC blocks. Default: 1 (keeps params low).
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
        # Use ConvNeXtV2Block instead of standard Conv1d
        self.stem = ConvNeXtV2Block(
            channels_in=self.n_input_channels, 
            channels_out=filters, 
            kernel_size=conv_kernel_size
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
                    dropout=dropout
                )
            )
            
        # 3. Profile Head
        self.profile_conv = nn.Conv1d(
            filters, self.n_output_channels, 
            kernel_size=profile_kernel_size, 
            padding='same'
        )
        
        # 4. Counts Head
        num_count_outputs = 1 if self.predict_total_count else self.n_output_channels
        self.count_dense = nn.Linear(filters, num_count_outputs)

    def forward(self, x) -> ProfileCountOutput:
        # 1. Stem
        x = self.stem(x)
        
        # 2. PGC Layers
        for layer in self.layers:
            x = layer(x)
            
        # --- Profile Head ---
        profile_logits = self.profile_conv(x) # (B, Out_Channels, Length)
        
        # Crop to target output_len
        current_len = profile_logits.shape[-1]
        target_len = self.output_len
        
        crop_l = 0
        crop_r = 0
        
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
        # Global Average Pooling
        # We crop the feature map to the target output length before pooling to ensure
        # the counts prediction corresponds to the target region.
        if current_len > target_len:
            x_for_counts = x[..., crop_l:-crop_r]
        else:
            x_for_counts = x
            
        x_pooled = x_for_counts.mean(dim=-1) # (B, Filters)
        
        log_counts = self.count_dense(x_pooled) # (B, Out_Channels)
        
        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)

class GemiNet2Medium(GemiNet2):
    """
    Medium version of GemiNet2 (~600k params).
    
    Changes from GemiNet2:
    - Filters: 64 -> 128
    - Layers: 8 -> 11
    - Dropout: 0.1 -> 0.15
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 128,
        n_dilated_layers: int = 11,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        expansion: int = 1,
        dropout: float = 0.15,
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

class GemiNet2Large(GemiNet2):
    """
    Large version of GemiNet2 (~2.2M params).
    
    Changes from GemiNet2:
    - Filters: 64 -> 128
    - Expansion: 1 -> 4
    - Layers: 8 -> 11
    - Dropout: 0.1 -> 0.2
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 128,
        n_dilated_layers: int = 11,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        expansion: int = 4,
        dropout: float = 0.2,
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

class GemiNet2ExtraLarge(GemiNet2):
    """
    Extra Large version of GemiNet2 (~5.0M params).
    
    Changes from GemiNet2:
    - Filters: 64 -> 224
    - Expansion: 1 -> 3
    - Layers: 8 -> 11
    - Dropout: 0.1 -> 0.35
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 224,
        n_dilated_layers: int = 11,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        expansion: int = 3,
        dropout: float = 0.35,
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


class GemiNet2MetricCollection(MetricCollection):
    """
    MetricCollection for GemiNet2 models.
    Includes Decoupled Pearson Correlation and Decoupled MSE (operating on reconstructed counts).
    """
    def __init__(self, num_channels: int = 1, implicit_log_targets: bool = False):
        super().__init__({
            "pearson": CountProfilePearsonCorrCoef(num_channels=num_channels, implicit_log_targets=implicit_log_targets),
            "mse_profile": CountProfileMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "mse_log_counts": LogCountsMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "pearson_log_counts": LogCountsPearsonCorrCoef(implicit_log_targets=implicit_log_targets),
        })
