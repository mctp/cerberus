import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from cerberus.loss import MSEMultinomialLoss
from cerberus.output import ProfileCountOutput
from cerberus.metrics import CountProfilePearsonCorrCoef, CountProfileMeanSquaredError, LogCountsMeanSquaredError, LogCountsPearsonCorrCoef
from cerberus.layers import DilatedResidualBlock


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
        n_dilated_layers (int): Number of dilated residual layers. Default: 8.
        conv_kernel_size (int): Kernel size for initial convolution. Default: 21.
        dil_kernel_size (int): Kernel size for dilated convolutions. Default: 3.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 75.
        predict_total_count (bool): If True, predicts a single total count scalar (sum of all channels).
                                    If False, predicts per-channel counts. Default: True.
    """
    def __init__(
        self,
        input_len: int = 2114,
        output_len: int = 1000,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 64,
        n_dilated_layers: int = 8,
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
                DilatedResidualBlock(filters, dil_kernel_size, dilation)
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

        self._tf_style_reinit()

    def _tf_style_reinit(self):
        """Re-initialize weights using Xavier uniform (Glorot) and zero biases.

        Matches the TensorFlow/Keras default initialization used by the original
        BPNet implementation and chrombpnet-pytorch. Without this, PyTorch defaults
        to Kaiming uniform which is calibrated for deeper ReLU networks and produces
        different activation scales at initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> ProfileCountOutput:
        """
        Forward pass.
        
        Args:
            x (Tensor): Input sequence (Batch, Channels, Input_Len)
            
        Returns:
            ProfileCountOutput: Contains profile_logits and log_counts.
                logits: (Batch, Out_Channels, Out_Len)
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
        # Typically bpnet expects 1000 output from 2114 input
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
        
        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)


class BPNet1024(BPNet):
    """
    BPNet1024: A configuration of BPNet optimized for 2112 -> 1024 prediction without cropping.
    
    Key Features:
    - Input: 2112 bp
    - Output: 1024 bp
    - Receptive Field Shrinkage: Exactly 1088 bp (2112 - 1024), achieved via tuned kernels.
      - Initial Conv Reduction: 20 (Kernel 21)
      - Tower Reduction: 1020 (8 layers, K=3, Dilations 2^1..2^8)
      - Profile Head Reduction: 48 (Kernel 49)
      - Total: 20 + 1020 + 48 = 1088.
    - Parameter Count: ~152k (roughly 50% increase over standard BPNet).
      - Achieved by increasing filters to 77.
    """
    def __init__(
        self,
        input_len: int = 2112,
        output_len: int = 1024,
        output_bin_size: int = 1,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 77,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 49,
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
            predict_total_count=predict_total_count,
        )


class BPNetLoss(MSEMultinomialLoss):
    """
    BPNet Loss with parameters fixed to match chrombpnet-pytorch implementation.
    
    Objective:
      1. Profile Loss: Multinomial NLL (using logits as unnormalized log-probs).
      2. Count Loss: MSE of log(global_count).
      
    Weights:
      Loss = beta * profile_loss + alpha * count_loss
      
    Differences from MSEMultinomialLoss:
      - Profile loss is averaged over channels (instead of summed) (average_channels=True).
      - Uses alpha/beta parameterization map to count_weight/profile_weight.

    Compatibility Note:
        This loss is mathematically equivalent to the loss in `bpnet-lite` and `chrombpnet-pytorch`.
        
        1. Profile Loss:
           - chrombpnet-pytorch: Calculates Multinomial NLL per channel (summing over sequence length),
             resulting in a (Batch, Channels) tensor. Then takes the MEAN over all elements 
             (batch and channels).
           - BPNetLoss: Sets `average_channels=True` and `flatten_channels=False`. This computes 
             NLL per channel (summing over length) and then takes the MEAN over batch and channels.
             Result: Identical.
             
        2. Count Loss:
           - chrombpnet-pytorch: Sums target counts over all channels and length. Calculates MSE between 
             predicted log-counts and log(total_counts + 1). Takes MEAN over batch.
           - BPNetLoss: Sets `count_per_channel=False`. This sums target counts over channels and length,
             computes log1p, and calculates MSE with predicted log-counts. Takes MEAN over batch.
             Result: Identical.
    """
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        """
        Args:
            alpha (float): Weight for count loss. Default: 1.0.
            beta (float): Weight for profile loss. Default: 1.0.
            **kwargs: Other arguments passed to MSEMultinomialLoss.
        """
        # Remove constrained arguments from kwargs if they exist to avoid multiple values error
        kwargs.pop("average_channels", None)
        kwargs.pop("flatten_channels", None)
        kwargs.pop("count_per_channel", None)
        kwargs.pop("log1p_targets", None)
        kwargs.pop("count_weight", None)
        kwargs.pop("profile_weight", None)

        # chrombpnet: loss = beta * profile + alpha * count
        # MSEMultinomialLoss: loss = profile_weight * profile + count_weight * count
        # We explicitly set all parameters to ensure strict compatibility regardless of defaults
        super().__init__(
            count_weight=alpha, 
            profile_weight=beta, 
            average_channels=True, 
            flatten_channels=False,
            count_per_channel=False,
            log1p_targets=False,
            **kwargs
        )


class BPNetMetricCollection(MetricCollection):
    """
    MetricCollection for BPNet models.
    Includes Decoupled Pearson Correlation and Decoupled MSE (operating on reconstructed counts).
    """
    def __init__(self, num_channels: int = 1, log1p_targets: bool = False, count_pseudocount: float = 1.0):
        super().__init__({
            "pearson": CountProfilePearsonCorrCoef(num_channels=num_channels, log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "mse_profile": CountProfileMeanSquaredError(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "mse_log_counts": LogCountsMeanSquaredError(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
            "pearson_log_counts": LogCountsPearsonCorrCoef(log1p_targets=log1p_targets, count_pseudocount=count_pseudocount),
        })
