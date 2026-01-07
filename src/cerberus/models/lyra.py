import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from cerberus.output import ProfileCountOutput

# -------------------------------------------------------------------------
# Core Components: S4D and PGC
# -------------------------------------------------------------------------

class S4DKernel(nn.Module):
    """
    S4D Kernel: Generates the convolution kernel for the SSM.
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate H distinct timescales
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        # C: Output projection (viewed as complex)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        
        self.register_parameter_custom("log_dt", nn.Parameter(log_dt), lr)
        
        # Diagonal A matrix parameters: Real (decay) and Imag (frequency)
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * torch.arange(N//2).repeat(H, 1)
        
        self.register_parameter_custom("log_A_real", nn.Parameter(log_A_real), lr)
        self.register_parameter_custom("A_imag", nn.Parameter(A_imag), lr)

    def register_parameter_custom(self, name, param, lr=None):
        """Helper to handle specific learning rates for SSM parameters."""
        self.register_parameter(name, param)
        setattr(getattr(self, name), "_optim", {"weight_decay": 0.0, "lr": lr})

    def forward(self, L):
        """
        Computes the SSM Kernel for sequence length L.
        Returns: (d_model, L)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H,)
        C = torch.view_as_complex(self.C) # (H, N//2)
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag) # (H, N//2)
        
        # Discretization
        dtA = A * dt.unsqueeze(-1) # (H, N//2)
        C = C * (torch.exp(dtA) - 1.) / A
        
        # Compute Power Series: A^t for t in [0, L-1]
        t = torch.arange(L, device=A.device) # (L,)
        K_exponents = dtA.unsqueeze(-1) * t # (H, N//2, L)
        
        # Sum over state dimension N
        vals = C.unsqueeze(-1) * torch.exp(K_exponents) # (H, N//2, L)
        K = 2 * torch.sum(vals, dim=1).real # (H, L)
        
        return K


class S4D(nn.Module):
    """
    S4D Layer: Applies the S4D kernel using FFT convolution.
    """
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, lr=0.001):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed
        self.d_output = self.h
        
        self.D = nn.Parameter(torch.randn(self.h))
        
        # SSM Kernel Generator
        self.kernel = S4DKernel(self.h, N=self.n, lr=lr)
        
        self.activation = nn.GELU()
        
        # Dropout
        # Use Dropout1d for tied dropout across sequence length (drops entire channels)
        # Dropout1d expects (N, C, L). If transposed=True, input is (B, H, L), so C=H. Correct.
        self.dropout = nn.Dropout1d(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Output projection (GLU Activation style)
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2) 
        )

    def forward(self, u, **kwargs):
        """
        u: (B, H, L) if transposed else (B, L, H)
        """
        if not self.transposed:
            u = u.transpose(-1, -2) # Force to (B, H, L)
        
        L = u.size(-1)
        
        # 1. Compute SSM Kernel
        k = self.kernel(L=L) # (H, L)
        
        # 2. Convolution via FFT
        k_f = torch.fft.rfft(k, n=2*L) # (H, 2L)
        # Ensure input is float32 for FFT stability/support on all devices
        u_f = torch.fft.rfft(u.float(), n=2*L) # (B, H, 2L)
        y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=2*L)[..., :L] # (B, H, L)
        y = y.type_as(u) # Cast back to original dtype
        
        # 3. D Skip Connection
        y = y + u * self.D.unsqueeze(-1)
        
        # 4. Activation and Dropout
        y = self.dropout(self.activation(y))
        
        # 5. Output Projection
        y = self.output_linear(y)
        
        if not self.transposed:
            y = y.transpose(-1, -2)
            
        return y


class PGC(nn.Module):
    """
    Projected Gated Convolution (PGC).
    """
    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        
        hidden_dim = int(d_model * expansion_factor)
        
        # Projections
        self.in_proj = nn.Linear(d_model, hidden_dim * 2)
        self.in_norm = nn.RMSNorm(hidden_dim * 2)
        
        # Depthwise Convolution
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        # Output
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.norm = nn.RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # u: (B, L, d_model)
        
        # 1. Expand and Normalize
        x_proj = self.in_proj(u)
        xv = self.in_norm(x_proj.float()).type_as(x_proj) # (B, L, 2 * hidden_dim)
        
        # 2. Split into two paths
        x, v = xv.chunk(2, dim=-1) # x, v: (B, L, hidden_dim)
        
        # 3. Convolution Path
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2) # (B, L, hidden_dim)
        
        # 4. Gating
        gate = x_conv * v
        
        # 5. Projection back
        out_proj = self.out_proj(gate)
        out = self.norm(out_proj.float()).type_as(out_proj)
        out = self.dropout(out)
        
        return out


# -------------------------------------------------------------------------
# LyraNet Architecture
# -------------------------------------------------------------------------

class LyraNet(nn.Module):
    """
    LyraNet: A profile+counts model using the Lyra architecture (PGC + S4D).
    
    Architecture:
    - Input: Sequence (Batch, Channels, Length)
    - Stem: Conv1d (captures motifs)
    - Body: 
        - Local Context: Stack of PGC layers
        - Global Context: Stack of S4D layers
    - Head 1 (Profile): Conv1D -> Logits
    - Head 2 (Counts): Global Avg Pool -> Dense -> Log(Total Counts)
    
    Args:
        input_len (int): Length of input sequence.
        output_len (int): Length of output sequence.
        input_channels (list[str]): List of input channel names.
        output_channels (list[str]): List of output channel names.
        filters (int): Model dimension. Default: 64.
        pgc_layers (int): Number of PGC layers. Default: 4.
        s4_layers (int): Number of S4D layers. Default: 2.
        pgc_expansion (float): Expansion factor for PGC layers. Default: 2.0.
        output_bin_size (int): Output resolution bin size. Default: 1.
        profile_kernel_size (int): Kernel size for profile head convolution. Default: 75.
        predict_total_count (bool): If True, predicts a single total count scalar. Default: True.
        dropout (float): Dropout rate. Default: 0.1.
        s4_lr (float) Ignored. Default: 0.001.
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_channels: list[str] = ["A", "C", "G", "T"],
        output_channels: list[str] = ["signal"],
        filters: int = 64,
        pgc_layers: int = 4,
        s4_layers: int = 3,
        pgc_expansion: float = 1.5,
        output_bin_size: int = 1,
        conv_kernel_size: int = 21,
        profile_kernel_size: int = 75,
        predict_total_count: bool = True,
        dropout: float = 0.1,
        s4_lr: float = 0.001, # ignored unless optimizer is made aware of this
    ):
        super().__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.output_bin_size = output_bin_size
        self.n_input_channels = len(input_channels)
        self.n_output_channels = len(output_channels)
        self.predict_total_count = predict_total_count
        self.prenorm = True
        
        # 1. Stem
        # Adapts input channels to model filters
        # Using 'same' padding to maintain length
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.n_input_channels, filters, 
                kernel_size=conv_kernel_size, 
                padding='same'
            ),
            nn.GELU()
        )
        
        # 2. PGC Layers (Local Interactions)
        self.pgc_layers = nn.ModuleList()
        for _ in range(pgc_layers):
            self.pgc_layers.append(
                PGC(filters, expansion_factor=pgc_expansion, dropout=dropout)
            )
            
        # 3. S4D Layers (Global Interactions)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(s4_layers):
            self.s4_layers.append(
                S4D(filters, dropout=dropout, transposed=True, lr=s4_lr)
            )
            self.norms.append(nn.RMSNorm(filters))
            self.dropouts.append(nn.Dropout(dropout))
            
        # 4. Profile Head
        self.profile_conv = nn.Conv1d(
            filters, self.n_output_channels, 
            kernel_size=profile_kernel_size, 
            padding='same'
        )
        
        # 5. Counts Head
        num_count_outputs = 1 if self.predict_total_count else self.n_output_channels
        self.count_dense = nn.Linear(filters, num_count_outputs)

    def forward(self, x, return_embeddings=False) -> ProfileCountOutput:
        # x: (B, Input_Channels, Input_Len)
        
        # 1. Stem
        x = self.stem(x) # (B, Filters, L)
        
        # 2. PGC Layers (Expects (B, L, Filters))
        x = x.transpose(1, 2) # (B, L, Filters)
        
        for pgc in self.pgc_layers:
            x = pgc(x)
            
        # 3. S4D Layers (Expects (B, Filters, L) because transposed=True)
        x = x.transpose(1, 2) # (B, Filters, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            residual = x
            z = x
            
            if self.prenorm:
                # Norm expects (..., D), so transpose
                z_in = z.transpose(-1, -2)
                z = norm(z_in.float()).type_as(z_in).transpose(-1, -2)
                
            # Apply S4D
            z = layer(z)
            z = dropout(z)
            
            # Residual
            x = z + residual
            
            if not self.prenorm:
                 x_in = x.transpose(-1, -2)
                 x = norm(x_in.float()).type_as(x_in).transpose(-1, -2)

        # --- Profile Head ---
        # x is (B, Filters, L)
        profile_logits = self.profile_conv(x) # (B, Out_Channels, L)
        
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
        # Global Average Pooling over the full sequence
        # x is (B, Filters, L)
        x_pooled = x.mean(dim=-1) # (B, Filters)
        
        log_counts = self.count_dense(x_pooled) # (B, Out_Channels)
        
        return ProfileCountOutput(logits=profile_logits, log_counts=log_counts)
