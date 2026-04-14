import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as _apply_weight_norm
from torchmetrics import MetricCollection

from cerberus.loss import MSEMultinomialLoss

logger = logging.getLogger(__name__)
from cerberus.layers import DilatedResidualBlock
from cerberus.metrics import (
    CountProfileMeanSquaredError,
    CountProfilePearsonCorrCoef,
    LogCountsMeanSquaredError,
    LogCountsPearsonCorrCoef,
)
from cerberus.output import ProfileCountOutput


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
        activation (str): Activation function used throughout the dilated tower. One of ``"relu"``
            or ``"gelu"``. GELU provides smoother gradients, reducing dying-neuron risk in deep
            stacks and working better with cosine LR schedules. Default: ``"relu"``.
        weight_norm (bool): If True, applies :func:`torch.nn.utils.weight_norm` to the initial
            convolution and all dilated residual block convolutions. This decouples weight magnitude
            from direction, stabilising gradient norms across the deep tower and enabling effective
            AdamW weight decay and cosine LR scheduling. Safe for DeepLIFT/DeepSHAP: weight
            normalisation is a weight reparameterisation, not an activation nonlinearity.
            Default: ``False``.
        residual_architecture (str): Residual block formulation. One of:
            ``"residual_pre-activation_conv"`` (``x + conv(act(x))``, default),
            ``"residual_post-activation_conv"`` (``x + act(conv(x))``),
            ``"activated_residual_pre-activation_conv"`` (``act(x) + conv(act(x))``).
            The initial convolution output is activated before entering the tower
            only for ``"residual_post-activation_conv"``.
            To match ``bpnet-refactor`` semantics, the two pre-activation variants
            apply an additional final ``ReLU`` after the full dilated tower.
    """

    def __init__(
        self,
        input_len: int = 2114,
        output_len: int = 1000,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        filters: int = 64,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        predict_total_count: bool = True,
        activation: str = "relu",
        weight_norm: bool = False,
        residual_architecture: str = "residual_pre-activation_conv",
    ):
        super().__init__()
        if input_channels is None:
            input_channels = ["A", "C", "G", "T"]
        if output_channels is None:
            output_channels = ["signal"]

        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.n_input_channels = len(input_channels)
        self.n_output_channels = len(output_channels)
        self.predict_total_count = predict_total_count
        self.residual_architecture = residual_architecture
        self._activate_iconv_before_tower = (
            residual_architecture == "residual_post-activation_conv"
        )
        self._apply_final_tower_relu = residual_architecture in {
            "residual_pre-activation_conv",
            "activated_residual_pre-activation_conv",
        }
        # Use an nn.Module activation (instead of functional F.relu) so hook-based
        # attribution methods can register on this operation.
        self.final_tower_relu = nn.ReLU()

        # 1. Initial Convolution (plain — weight_norm applied after reinit if requested)
        self.iconv: nn.Module = nn.Conv1d(
            self.n_input_channels,
            filters,
            kernel_size=conv_kernel_size,
            padding="valid",
        )

        # Activation module used for the initial conv output when the selected
        # residual architecture is post-activation.
        if activation == "relu":
            self.iconv_act: nn.Module = nn.ReLU()
        elif activation == "gelu":
            self.iconv_act = nn.GELU()
        else:
            raise ValueError(
                f"BPNet: unsupported activation {activation!r}. Must be 'relu' or 'gelu'."
            )

        # 2. Dilated Residual Tower (built plain — weight_norm applied after reinit if requested)
        self.res_layers = nn.ModuleList()
        for i in range(1, n_dilated_layers + 1):
            # Dilation increases exponentially: 2^1, 2^2, ...
            dilation = 2**i
            self.res_layers.append(
                DilatedResidualBlock(
                    filters,
                    dil_kernel_size,
                    dilation,
                    activation=activation,
                    weight_norm=False,
                    residual_architecture=residual_architecture,
                )
            )

        # 3. Profile Head
        # Predicts shape (logits)
        self.profile_conv = nn.Conv1d(
            filters,
            self.n_output_channels,
            kernel_size=profile_kernel_size,
            padding="valid",
        )

        # 4. Counts Head
        # Predicts total count (log space)
        # Global Average Pooling is performed in forward()
        # If predict_total_count is True (default), we output a single scalar (total counts)
        # regardless of the number of profile output channels, matching chrombpnet/bpnet-lite.
        num_count_outputs = 1 if self.predict_total_count else self.n_output_channels
        self.count_dense = nn.Linear(filters, num_count_outputs)

        # Xavier reinit on plain weights first, then apply weight_norm.
        # This follows PyTorch's recommended pattern: right_inverse decomposes the
        # Xavier-initialized weight into weight_g and weight_v, preserving the intent.
        self._tf_style_reinit()
        if weight_norm:
            _apply_weight_norm(self.iconv)
            for block in self.res_layers:
                if isinstance(block, DilatedResidualBlock):
                    _apply_weight_norm(block.conv)

        logger.info(
            "BPNet initialized: filters=%d, n_dilated_layers=%d, activation=%s, "
            "weight_norm=%s, residual_architecture=%s, iconv_activation_before_tower=%s, "
            "final_tower_relu=%s",
            filters,
            n_dilated_layers,
            activation,
            weight_norm,
            residual_architecture,
            self._activate_iconv_before_tower,
            self._apply_final_tower_relu,
        )

    @staticmethod
    def compute_shrinkage(
        conv_kernel_size: int = 21,
        n_dilated_layers: int = 8,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
    ) -> int:
        """Compute total shrinkage (in bp) for BPNet's valid-padding conv stack.

        Shrinkage = stem + tower + profile_head, where each valid-padding layer
        shrinks by ``dilation * (kernel_size - 1)``.  BPNet uses exponential
        dilations ``2**i`` for ``i`` in ``1..n_dilated_layers``.

        Args:
            conv_kernel_size: Initial conv kernel size. Default: ``21``.
            n_dilated_layers: Number of dilated residual layers. Default: ``8``.
            dil_kernel_size: Dilated conv kernel size. Default: ``3``.
            profile_kernel_size: Profile head kernel size. Default: ``75``.

        Returns:
            Total input-to-output shrinkage in bp.
        """
        stem = conv_kernel_size - 1
        dilations = [2**i for i in range(1, n_dilated_layers + 1)]
        tower = sum(d * (dil_kernel_size - 1) for d in dilations)
        head = profile_kernel_size - 1
        return stem + tower + head

    def _tf_style_reinit(self):
        """Re-initialize weights using Xavier uniform (Glorot) and zero biases.

        Matches the TensorFlow/Keras default initialization used by the original
        BPNet implementation and chrombpnet-pytorch. Without this, PyTorch defaults
        to Kaiming uniform which is calibrated for deeper ReLU networks and produces
        different activation scales at initialization.

        Called before weight normalization is applied (if ``weight_norm=True``).
        PyTorch's ``right_inverse`` then decomposes the Xavier-initialized weight
        into ``weight_g`` and ``weight_v``, preserving the initialization intent.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d | nn.Linear):
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
        # Center-crop or reject input based on expected input_len
        if x.shape[-1] > self.input_len:
            crop = (x.shape[-1] - self.input_len) // 2
            x = x[..., crop : crop + self.input_len]
        elif x.shape[-1] < self.input_len:
            raise ValueError(
                f"Input length {x.shape[-1]} is shorter than required {self.input_len}"
            )

        # 1. Initial Conv
        # bpnet-refactor pre-activation variants pass an unactivated tensor into
        # the first dilated block. The post-activation variant applies activation here.
        x = self.iconv(x)
        if self._activate_iconv_before_tower:
            x = self.iconv_act(x)

        # 2. Residual Tower
        for layer in self.res_layers:
            x = layer(x)

        # bpnet-refactor applies a final ReLU after the dilated tower for both
        # pre-activation variants (syntax_module final activation).
        if self._apply_final_tower_relu:
            x = self.final_tower_relu(x)

        # --- Profile Head ---
        profile_logits = self.profile_conv(x)  # (B, Out_Channels, Length)

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
            raise ValueError(
                f"Output length {current_len} is smaller than requested {target_len}"
            )

        if self.output_bin_size > 1:
            # Average Pooling if binning is requested
            # Note: This reduces resolution from Input_Len to Output_Len
            profile_logits = F.avg_pool1d(
                profile_logits,
                kernel_size=self.output_bin_size,
                stride=self.output_bin_size,
            )

        # --- Counts Head ---
        # Global Average Pooling over the sequence length of the latent representation
        # x is (B, Filters, Input_Len) (or cropped length)
        x_pooled = x.mean(dim=-1)  # (B, Filters)

        log_counts = self.count_dense(x_pooled)  # (B, Out_Channels)

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
        input_channels: list[str] | None = None,
        output_channels: list[str] | None = None,
        filters: int = 77,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 49,
        predict_total_count: bool = True,
        activation: str = "relu",
        weight_norm: bool = False,
        residual_architecture: str = "residual_pre-activation_conv",
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
            activation=activation,
            weight_norm=weight_norm,
            residual_architecture=residual_architecture,
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
        # BPNetLoss enforces fixed values for chrombpnet compatibility.
        # Warn if the caller explicitly passed conflicting values.
        _fixed = {
            "average_channels": True,
            "flatten_channels": False,
            "count_per_channel": False,
            "log1p_targets": False,
            "count_weight": alpha,
            "profile_weight": beta,
        }
        for key, fixed_val in _fixed.items():
            caller_val = kwargs.pop(key, None)
            if caller_val is not None and caller_val != fixed_val:
                logger.warning(
                    f"BPNetLoss: ignoring {key}={caller_val!r}, "
                    f"using fixed value {fixed_val!r} for chrombpnet compatibility"
                )

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
            **kwargs,
        )


class MultitaskBPNet(BPNet):
    """Multi-task BPNet: shared dilated tower with N condition-specific output channels.

    Phase 1 of the two-phase differential accessibility workflow.  Each
    condition gets its own profile head channel and its own count head
    output, enabling cross-condition comparisons that are not confounded by
    separate model training runs — the architecture principle of bpAI-TAC
    (Chandra et al. 2025, bioRxiv 2025.01.24.634804).

    The model is architecturally identical to :class:`BPNet` with
    ``predict_total_count=False`` enforced.  Per-channel count prediction is
    a hard requirement: Phase 2 differential fine-tuning supervises the
    *difference* of two specific count heads, so each condition must produce
    an independent scalar output.

    Args:
        output_channels: List of condition names (one per steady-state
            condition).  Must contain at least 2 entries.
        input_len: Length of input sequence. Default: 2114.
        output_len: Length of output sequence. Default: 1000.
        output_bin_size: Output resolution bin size. Default: 1.
        input_channels: Input channel names. Default: ``["A","C","G","T"]``.
        filters: Number of conv filters. Default: 64.
        n_dilated_layers: Number of dilated residual layers. Default: 8.
        conv_kernel_size: Kernel size for initial conv. Default: 21.
        dil_kernel_size: Kernel size for dilated convs. Default: 3.
        profile_kernel_size: Kernel size for profile head. Default: 75.
        activation: Activation function (``"relu"`` or ``"gelu"``). Default: ``"relu"``.
        weight_norm: Apply weight normalisation. Default: ``False``.
        residual_architecture: Residual block formulation. Default:
            ``"residual_pre-activation_conv"``.

    References:
        - bpAI-TAC: Chandra et al. (2025). *Refining sequence-to-activity
          models by increasing model resolution.* bioRxiv 2025.01.24.634804.
        - Naqvi et al. (2025). *Transfer learning reveals sequence determinants
          of the quantitative response to transcription factor dosage.*
          Cell Genomics. PMC11160683.
    """

    def __init__(
        self,
        output_channels: list[str],
        input_len: int = 2114,
        output_len: int = 1000,
        output_bin_size: int = 1,
        input_channels: list[str] | None = None,
        filters: int = 64,
        n_dilated_layers: int = 8,
        conv_kernel_size: int = 21,
        dil_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        activation: str = "relu",
        weight_norm: bool = False,
        residual_architecture: str = "residual_pre-activation_conv",
    ):
        if len(output_channels) < 2:
            raise ValueError(
                f"MultitaskBPNet requires at least 2 output_channels, "
                f"got {len(output_channels)}: {output_channels!r}. "
                "Each entry represents one steady-state condition."
            )
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
            predict_total_count=False,
            activation=activation,
            weight_norm=weight_norm,
            residual_architecture=residual_architecture,
        )
        logger.info(
            "MultitaskBPNet initialized: n_conditions=%d, conditions=%s",
            len(output_channels),
            output_channels,
        )

    @property
    def n_conditions(self) -> int:
        """Number of steady-state conditions (output channels)."""
        return self.n_output_channels

    @property
    def condition_channels(self) -> int:
        """Alias for n_conditions."""
        return self.n_output_channels


class MultitaskBPNetLoss(MSEMultinomialLoss):
    """Phase 1 loss for :class:`MultitaskBPNet`.

    Applies the BPNet profile + count loss independently per condition
    channel.  Profile loss is averaged across conditions (following
    bpAI-TAC).  Count loss is computed per channel, requiring
    ``predict_total_count=False`` in the model.

    Parameters ``alpha`` and ``beta`` map to ``count_weight`` and
    ``profile_weight`` respectively, matching the :class:`BPNetLoss`
    convention.

    Args:
        alpha: Weight for per-channel count MSE loss. Default: 1.0.
        beta: Weight for per-channel profile multinomial NLL loss. Default: 1.0.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        _fixed = {
            "count_per_channel": True,
            "average_channels": True,
            "flatten_channels": False,
            "count_weight": alpha,
            "profile_weight": beta,
        }
        for key, fixed_val in _fixed.items():
            caller_val = kwargs.pop(key, None)
            if caller_val is not None and caller_val != fixed_val:
                logger.warning(
                    "MultitaskBPNetLoss: ignoring %s=%r, using fixed value %r",
                    key,
                    caller_val,
                    fixed_val,
                )
        super().__init__(
            count_per_channel=True,
            average_channels=True,
            flatten_channels=False,
            count_weight=alpha,
            profile_weight=beta,
            **kwargs,
        )


class BPNetMetricCollection(MetricCollection):
    """
    MetricCollection for BPNet models.
    Includes Decoupled Pearson Correlation and Decoupled MSE (operating on reconstructed counts).
    """

    def __init__(
        self,
        log1p_targets: bool = False,
        count_pseudocount: float = 1.0,
        log_counts_include_pseudocount: bool = False,
    ):
        super().__init__(
            {
                "pearson": CountProfilePearsonCorrCoef(
                    log1p_targets=log1p_targets, count_pseudocount=count_pseudocount
                ),
                "mse_profile": CountProfileMeanSquaredError(
                    log1p_targets=log1p_targets, count_pseudocount=count_pseudocount
                ),
                "mse_log_counts": LogCountsMeanSquaredError(
                    log1p_targets=log1p_targets,
                    count_pseudocount=count_pseudocount,
                    log_counts_include_pseudocount=log_counts_include_pseudocount,
                ),
                "pearson_log_counts": LogCountsPearsonCorrCoef(
                    log1p_targets=log1p_targets,
                    count_pseudocount=count_pseudocount,
                    log_counts_include_pseudocount=log_counts_include_pseudocount,
                ),
            }
        )
