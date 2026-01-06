import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from cerberus.output import ProfileCountOutput
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError

class PGCBlock(nn.Module):
    """
    Projected Gated Convolution Block.
    
    Structure:
    1. Projection (Pointwise) -> Expansion
    2. RMSNorm
    3. Split into X, V
    4. Dilated Depthwise Conv on X
    5. Gating: G = X_conv * V
    6. Projection (Pointwise) -> Compression
    7. RMSNorm
    8. Dropout
    9. Residual Connection
    
    Args:
        dim (int): Input/Output dimension.
        kernel_size (int): Kernel size for depthwise convolution.
        dilation (int): Dilation rate.
        expansion (int): Expansion factor. Internal dimension = dim * expansion.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim: int, kernel_size: int, dilation: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = dim * expansion
        
        # 1. Input Projection (Expansion)
        # We project to 2 * hidden_dim to split into X and V
        self.in_proj = nn.Conv1d(dim, 2 * self.hidden_dim, kernel_size=1)
        
        # 2. Norm after projection
        self.norm1 = nn.RMSNorm(2 * self.hidden_dim)
        
        # 3. Depthwise Conv
        # Applied to X part only (first half)
        self.conv = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            groups=self.hidden_dim, # Depthwise
            padding='same'
        )
        
        # 4. Output Projection
        self.out_proj = nn.Conv1d(self.hidden_dim, dim, kernel_size=1)
        
        # 5. Norm & Dropout
        self.norm2 = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, L)
        residual = x
        
        # 1. Project
        x = self.in_proj(x)
        
        # 2. Norm (requires B, L, C)
        x = x.transpose(1, 2) # (B, L, C)
        x = self.norm1(x.float()).type_as(x)
        x = x.transpose(1, 2) # (B, C, L)
        
        # 3. Split
        x, v = torch.chunk(x, 2, dim=1)
        
        # 4. Depthwise Conv on X
        x = self.conv(x)
        
        # 5. Gating
        x = x * v
        
        # 6. Output Project
        x = self.out_proj(x)
        
        # 7. Norm & Dropout (requires B, L, C)
        x = x.transpose(1, 2)
        x = self.norm2(x.float()).type_as(x)
        x = x.transpose(1, 2)
        
        x = self.dropout(x)
        
        # 8. Residual
        return residual + x

class GemiNet(nn.Module):
    """
    GemiNet: A modern replacement for BPNet using Projected Gated Convolutions.
    
    Architecture:
    - Input: One-hot sequence (Batch, 4, Length)
    - Stem: Standard Conv1d (captures motifs)
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
        # We use 'same' padding to maintain length, unlike BPNet which uses 'valid'.
        # This simplifies length handling.
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.n_input_channels, filters, 
                kernel_size=conv_kernel_size, 
                padding='same'
            ),
            nn.ReLU()
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
        # Note: We pool over the full (uncropped) length to use all context for counts
        x_pooled = x.mean(dim=-1) # (B, Filters)
        
        log_counts = self.count_dense(x_pooled) # (B, Out_Channels)
        
        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)

class GeminetLarge(GemiNet):
    """
    Larger version of GemiNet (~600k params).
    
    Changes from GemiNet:
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

class GeminetExtraLarge(GemiNet):
    """
    Extra Large version of GemiNet (~2.2M params).
    
    Changes from GemiNet:
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


class GemiNetMetricCollection(MetricCollection):
    """
    MetricCollection for GemiNet models.
    Includes Decoupled Pearson Correlation and Decoupled MSE (operating on reconstructed counts).
    """
    def __init__(self, num_channels: int = 1, implicit_log_targets: bool = False):
        super().__init__({
            "pearson": CountProfilePearsonCorrCoef(num_channels=num_channels, implicit_log_targets=implicit_log_targets),
            "mse_profile": CountProfileMeanSquaredError(implicit_log_targets=implicit_log_targets),
            "mse_log_counts": LogCountsMeanSquaredError(implicit_log_targets=implicit_log_targets),
        })
