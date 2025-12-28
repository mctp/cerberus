import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualBlock(nn.Module):
    """
    Dilated Residual Block for BPNet.
    Structure: Input -> Dilated Conv -> ReLU -> Add to Input
    """
    def __init__(self, filters, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            filters, filters, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding='valid'
        )
        
    def forward(self, x):
        # Post-Activation Residual Block
        # 1. Dilated Conv
        out = self.conv(x)
        # 2. ReLU
        out = F.relu(out)
        # 3. Residual Connection
        # Center crop x to match out
        diff = x.shape[-1] - out.shape[-1]
        if diff > 0:
            crop_l = diff // 2
            crop_r = diff - crop_l
            x = x[..., crop_l:-crop_r]
                
        return x + out


class BPNet(nn.Module):
    """
    BPNet: Base-Resolution Prediction Net.
    
    Architecture based on the Consensus BPNet specification:
    - Input: One-hot sequence (Batch, 4, Length)
    - Body: Initial Conv -> N Dilated Residual Blocks
    - Head 1 (Profile): Conv1D -> Logits
    - Head 2 (Counts): Global Avg Pool -> Dense -> Log(Total Counts)
    
    Uses 'valid' padding with center cropping to match the reference implementation.
    
    Args:
        input_len (int): Length of input sequence.
        output_len (int): Length of output sequence.
        output_bin_size (int): Output resolution bin size.
        input_channels (list[str]): List of input channel names.
        output_channels (list[str]): List of output channel names.
        filters (int): Number of filters in convolutional layers. Default: 64.
        n_dilated_layers (int): Number of dilated residual layers. Default: 9.
        conv_kernel_size (int): Kernel size for initial convolution. Default: 21.
        dil_kernel_size (int): Kernel size for dilated convolutions. Default: 3.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 75.
        predict_total_count (bool): If True, predicts a single total count scalar (sum of all channels).
                                    If False, predicts per-channel counts. Default: True.
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 64,
        n_dilated_layers: int = 9,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        predict_total_count: bool = True,
    ):
        super().__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.n_input_channels = len(input_channels)
        self.n_output_channels = len(output_channels)
        self.predict_total_count = predict_total_count
        
        # 1. Initial Convolution
        self.iconv = nn.Conv1d(
            self.n_input_channels, filters, 
            kernel_size=conv_kernel_size, 
            padding='valid'
        )
        
        # 2. Dilated Residual Tower
        self.res_layers = nn.ModuleList()
        for i in range(1, n_dilated_layers + 1):
            # Dilation increases exponentially: 2^1, 2^2, ...
            dilation = 2**i
            self.res_layers.append(
                _ResidualBlock(filters, dil_kernel_size, dilation)
            )
            
        # 3. Profile Head
        # Predicts shape (logits)
        self.profile_conv = nn.Conv1d(
            filters, self.n_output_channels, 
            kernel_size=profile_kernel_size, 
            padding='valid'
        )
        
        # 4. Counts Head
        # Predicts total count (log space)
        # Global Average Pooling is performed in forward()
        # If predict_total_count is True (default), we output a single scalar (total counts)
        # regardless of the number of profile output channels, matching chrombpnet/bpnet-lite.
        num_count_outputs = 1 if self.predict_total_count else self.n_output_channels
        self.count_dense = nn.Linear(filters, num_count_outputs)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input sequence (Batch, Channels, Input_Len)
            
        Returns:
            Tuple[Tensor, Tensor]: (profile_logits, log_counts)
                profile_logits: (Batch, Out_Channels, Out_Len)
                log_counts: (Batch, Out_Channels) - representing log(total_counts)
        """
        # 1. Initial Conv + ReLU
        x = F.relu(self.iconv(x))
        
        # 2. Residual Tower
        for layer in self.res_layers:
            x = layer(x)
            
        # --- Profile Head ---
        profile_logits = self.profile_conv(x) # (B, Out_Channels, Length)
        
        # Crop to target output_len if needed
        # We assume output_len is set to the desired length after all valid convolutions
        # Typically chrombpnet expects 1000 output from 2114 input
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
            # Average Pooling if binning is requested
            # Note: This reduces resolution from Input_Len to Output_Len
            profile_logits = F.avg_pool1d(
                profile_logits, 
                kernel_size=self.output_bin_size, 
                stride=self.output_bin_size
            )

        # --- Counts Head ---
        # Global Average Pooling over the sequence length of the latent representation
        # x is (B, Filters, Input_Len) (or cropped length)
        x_pooled = x.mean(dim=-1) # (B, Filters)
        
        log_counts = self.count_dense(x_pooled) # (B, Out_Channels)
        
        return profile_logits, log_counts
