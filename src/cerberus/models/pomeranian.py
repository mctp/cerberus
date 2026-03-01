import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from cerberus.output import ProfileCountOutput
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError, LogCountsPearsonCorrCoef
from cerberus.layers import ConvNeXtV2Block, PGCBlock

class Pomeranian(nn.Module):
    """
    Pomeranian: A lightweight model mirroring BPNet (valid padding) but using modern components.
    
    Default Configuration (aka PomeranianK9):
    - Input: 2112 bp
    - Output: 1024 bp
    - Stem: Factorized [11, 11] (Expansion 2)
    - Body: 8 Layers, Kernel 9, Dilations [1, 1, 2, 4, 8, 16, 32, 64]
    - Head: Kernel 45
    
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
    
    Args:
        input_len (int): Length of input sequence. Default: 2112.
        output_len (int): Length of output sequence. Default: 1024.
        output_bin_size (int): Output resolution bin size.
        input_channels (list[str]): List of input channel names.
        output_channels (list[str]): List of output channel names.
        filters (int): Model dimension. Default: 64.
        n_dilated_layers (int): Number of dilated PGC blocks. Default: 8.
        conv_kernel_size (int | list[int]): Kernel size for initial convolution (stem). Default: [11, 11].
        dil_kernel_size (int | list[int]): Kernel size for dilated convolutions. Default: 9.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 45.
        expansion (int): Expansion factor for PGC blocks. Default: 1.
        dropout (float): Dropout rate. Default: 0.1.
        predict_total_count (bool): If True, predicts a single total count scalar. Default: True.
        stem_expansion (int): Expansion factor for Stem blocks. Default: 2.
        dilations (list[int] | None): Dilation schedule. Default: [1, 1, 2, 4, 8, 16, 32, 64].
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
        conv_kernel_size: int | list[int] = [11, 11],
        dil_kernel_size: int | list[int] = 9,
        profile_kernel_size: int = 45,
        expansion: int = 1,
        dropout: float = 0.1,
        predict_total_count: bool = True,
        stem_expansion: int = 2,
        dilations: list[int] | None = [1, 1, 2, 4, 8, 16, 32, 64],
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
        if isinstance(conv_kernel_size, int):
            self.stem = ConvNeXtV2Block(
                channels_in=self.n_input_channels, 
                channels_out=filters, 
                kernel_size=conv_kernel_size,
                padding='valid',
                inv_bottleneckscale=stem_expansion
            )
        else:
            # Multi-layer stem
            layers = []
            for i, k_size in enumerate(conv_kernel_size):
                # First layer takes input channels, subsequent layers take filters
                c_in = self.n_input_channels if i == 0 else filters
                # First layer is dense (Stem), subsequent layers are depthwise (Block)
                use_groups = (i > 0)
                layers.append(
                    ConvNeXtV2Block(
                        channels_in=c_in,
                        channels_out=filters,
                        kernel_size=k_size,
                        padding='valid',
                        groups=use_groups,
                        inv_bottleneckscale=stem_expansion
                    )
                )
            self.stem = nn.Sequential(*layers)
        
        # 2. Dilated PGC Tower
        self.layers = nn.ModuleList()
        if dilations is not None:
            dilation_schedule = dilations
        else:
            dilation_schedule = [2**i for i in range(1, n_dilated_layers + 1)]
            
        if isinstance(dil_kernel_size, int):
            kernel_schedule = [dil_kernel_size] * len(dilation_schedule)
        else:
            kernel_schedule = dil_kernel_size
            if len(kernel_schedule) != len(dilation_schedule):
                raise ValueError(f"Kernel schedule length {len(kernel_schedule)} != Dilation schedule length {len(dilation_schedule)}")

        for dilation, k_size in zip(dilation_schedule, kernel_schedule):
            self.layers.append(
                PGCBlock(
                    dim=filters,
                    kernel_size=k_size,
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
            nn.Linear(hidden_dim, num_count_outputs),
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


class PomeranianK5(Pomeranian):
    """
    PomeranianK5 (Medium Kernel): K5 body with a 2-layer factorized stem.
    
    Structure:
    - Stem: [11, 11] (Shrinkage 20)
    - Body: K=5, Dilations [1..128]
    - Head: K=49
    
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
        conv_kernel_size: list[int] = [11, 11],
        dil_kernel_size: int = 5,
        profile_kernel_size: int = 49,
        expansion: int = 1,
        dropout: float = 0.1,
        predict_total_count: bool = True,
        stem_expansion: int = 2,
        dilations: list[int] | None = [1, 2, 4, 8, 16, 32, 64, 128],
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
            stem_expansion=stem_expansion,
            dilations=dilations,
        )


class PomeranianMetricCollection(MetricCollection):
    """
    MetricCollection for Pomeranian models.
    """
    def __init__(self, num_channels: int = 1, log1p_targets: bool = False, count_pseudocount: float = 1.0):
        super().__init__({
            "pearson": CountProfilePearsonCorrCoef(num_channels=num_channels, log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "mse_profile": CountProfileMeanSquaredError(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "mse_log_counts": LogCountsMeanSquaredError(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "pearson_log_counts": LogCountsPearsonCorrCoef(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
        })
