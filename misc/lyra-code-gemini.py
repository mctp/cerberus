"""
Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences.

Reference: Ramesh et al. 2025
Implementation derived from manuscript supplementary materials.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# -------------------------------------------------------------------------
# Helper Modules
# -------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Standard implementation to match paper description (Source 925).
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (..., d_model)
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed

class DropoutNd(nn.Module):
    """
    N-dimensional dropout as used in the S4D listing.
    Adapted to use pure PyTorch without einops for standard compliance.
    """
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if not self.training or self.p == 0.0:
            return X

        # X shape: (B, D, L) if transposed else (B, L, D)
        # We want to drop entire channels or time steps based on 'tie' logic.
        # The supplementary code implies standard dropout on the feature dimension.
        
        # Simplify to standard dropout for this implementation to ensure stability,
        # unless specific structure is required. The code below mimics the intent
        # of dropping features consistently across the sequence (if tied).
        
        if self.transposed:
            # Input is (B, D, L)
            mask_shape = list(X.shape)
            if self.tie:
                # Drop same channel across all L: mask shape (B, D, 1)
                mask_shape[-1] = 1
        else:
            # Input is (B, L, D)
            mask_shape = list(X.shape)
            if self.tie:
                # Drop same channel across all L: mask shape (B, 1, D)
                mask_shape[1] = 1
                
        mask = torch.empty(mask_shape, dtype=X.dtype, device=X.device).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return X * mask

# -------------------------------------------------------------------------
# Core Components: S4D and PGC
# -------------------------------------------------------------------------

class S4DKernel(nn.Module):
    """
    S4D Kernel: Generates the convolution kernel for the SSM.
    Math: Computes the impulse response h_t = C * A^t * B using a 
    truncated generating function approach (Source 1473-1481).
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate H distinct timescales
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        # C: Output projection (viewed as complex)
        # Listing 2 line 1638: C is complex normal
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
        # Note: In a full training loop, one would inspect these attributes 
        # to set specific optimizer hyperparameters (e.g. weight_decay=0).
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
        
        # Discretization (ZOH approximation logic from Listing 2 Line 1652)
        # C_tilde = C * (exp(dtA) - 1) / A
        dtA = A * dt.unsqueeze(-1) # (H, N//2)
        C = C * (torch.exp(dtA) - 1.) / A
        
        # Compute Power Series: A^t for t in [0, L-1]
        # We do this in log space/exponent space for stability
        # K_exponent = dtA * t
        t = torch.arange(L, device=A.device) # (L,)
        K_exponents = dtA.unsqueeze(-1) * t # (H, N//2, L)
        
        # Sum over state dimension N to get impulse response
        # kernel = 2 * Real( sum( C * exp(dtA * t) ) )
        # The factor of 2 accounts for conjugate symmetry of the real signal
        vals = C.unsqueeze(-1) * torch.exp(K_exponents) # (H, N//2, L)
        K = 2 * torch.sum(vals, dim=1).real # (H, L)
        
        return K


class S4D(nn.Module):
    """
    S4D Layer: Applies the S4D kernel using FFT convolution.
    Source 920: "S4D layer... hidden dimension 64... residual connection and sequence prenormalization".
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
        self.dropout = DropoutNd(dropout, transposed=self.transposed) if dropout > 0.0 else nn.Identity()
        
        # Output projection (GLU Activation style)
        # Listing 2 line 1697: Linear to 2*h -> GLU -> h
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
        u_f = torch.fft.rfft(u, n=2*L) # (B, H, 2L)
        y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=2*L)[..., :L] # (B, H, L)
        
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
    Source 268: "Transform through projection... two parallel pathways... depthwise 1D conv... linear projection... element-wise multiplication".
    """
    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        
        hidden_dim = int(d_model * expansion_factor)
        
        # Projections
        self.in_proj = nn.Linear(d_model, hidden_dim * 2)
        self.in_norm = RMSNorm(hidden_dim * 2)
        
        # Depthwise Convolution
        # Groups=hidden_dim ensures depthwise operation
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        # Output
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.norm = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # u: (B, L, d_model)
        
        # 1. Expand and Normalize
        # Listing 1 line 1561: xv = in_norm(in_proj(u))
        xv = self.in_norm(self.in_proj(u)) # (B, L, 2 * hidden_dim)
        
        # 2. Split into two paths
        x, v = xv.chunk(2, dim=-1) # x, v: (B, L, hidden_dim)
        
        # 3. Convolution Path (local dependencies)
        # Conv1d expects (B, Dim, Length), so we transpose
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2) # (B, L, hidden_dim)
        
        # 4. Gating (Global/Linear mixing conditioned by Local Conv)
        # Source 1562-1563 implied: element-wise multiplication
        gate = x_conv * v
        
        # 5. Projection back
        out = self.norm(self.out_proj(gate))
        out = self.dropout(out)
        
        return out


# -------------------------------------------------------------------------
# Lyra Architecture
# -------------------------------------------------------------------------

class Lyra(nn.Module):
    """
    Lyra: Efficient Biological Sequence Modeling.
    Combines PGC layers for local feature extraction and S4D for global context.
    
    Args:
        model_dimension (int): Hidden dimension.
        pgc_configs (list): List of (hidden_dim, num_layers) for PGC blocks.
        num_s4 (int): Number of S4D layers.
        d_input (int): Input feature dimension (e.g., 4 for DNA OHE, 20 for Protein).
        d_output (int): Output dimension (e.g., 1 for regression, N for classification).
    """
    def __init__(
        self,
        model_dimension,
        pgc_configs,
        num_s4,
        d_input,
        d_output=1,
        dropout=0.2,
        prenorm=True
    ):
        super().__init__()
        self.d_model = model_dimension
        self.prenorm = prenorm
        
        # Encoder: Projects input to model dimension
        self.encoder = nn.Linear(d_input, model_dimension)
        
        # PGC Layers (Local Interactions)
        # Generalized to accept a configuration list
        self.pgc_layers = nn.ModuleList()
        for hidden_dim, num_layers_pgc in pgc_configs:
            # Assuming the config implies a stack of PGCs.
            # In the paper implementation (Listing 3), it adds PGC layers based on config.
            # We map the config to specific expansion factors or dimensions.
            # To match Listing 3 logic exactly:
            # "pgc_hidden_dimension, num_layers = config"
            # It seems Listing 3 iterates config and adds ONE PGC module per config entry?
            # Re-reading Listing 3 lines 32-34: It adds `num_layers` logic inside PGC?
            # No, Listing 1 `PGC` class doesn't have a loop. 
            # We will implement as a stack of PGC layers.
            
            expansion = hidden_dim / model_dimension
            for _ in range(num_layers_pgc):
                self.pgc_layers.append(PGC(model_dimension, expansion_factor=expansion, dropout=dropout))
                
        # S4D Layers (Global Interactions)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(model_dimension, dropout=dropout, transposed=True, lr=0.001)
            )
            self.norms.append(RMSNorm(model_dimension))
            self.dropouts.append(nn.Dropout(dropout))
            
        # Decoder
        self.decoder = nn.Linear(model_dimension, d_output)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x, return_embeddings=False):
        """
        Args:
            x: (B, L, d_input)
            return_embeddings: If True, returns (logits, embeddings)
        """
        # 1. Encode
        x = self.encoder(x) # (B, L, d_model)
        
        # 2. PGC Block (Local)
        for pgc in self.pgc_layers:
            x = pgc(x)
            
        # 3. S4D Block (Global)
        # S4D expects (B, D, L) because transposed=True in init
        x = x.transpose(-1, -2) # (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            residual = x
            z = x
            
            if self.prenorm:
                # Norm expects (..., D), so transpose
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                
            # Apply S4D
            z = layer(z)
            z = dropout(z)
            
            # Residual
            x = z + residual
            
            if not self.prenorm:
                 x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        # 4. Pooling
        # Paper Source 1864: Average pooling over sequence length
        # x is currently (B, d_model, L)
        embeddings = x.mean(dim=-1) # (B, d_model)
        
        # 5. Decode
        out = self.final_dropout(embeddings)
        out = self.decoder(out) # (B, d_output)
        
        if return_embeddings:
            return out, embeddings
        return out

# -------------------------------------------------------------------------
# Usage Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example Configuration based on "Methods and Materials" (Source 919-920)
    # "Lyra includes two PGC blocks... first hidden dim 16, second hidden dim 128... S4D dim 64"
    
    BATCH_SIZE = 2
    SEQ_LEN = 512
    INPUT_DIM = 20   # e.g., Protein OHE
    OUTPUT_DIM = 1   # e.g., Fitness score
    MODEL_DIM = 64
    
    # Config: [(hidden_dim, num_layers)]
    # Note: expansion factor is derived inside class as hidden_dim / MODEL_DIM
    pgc_config = [
        (16, 1),   # First PGC
        (128, 1)   # Second PGC
    ]
    
    model = Lyra(
        model_dimension=MODEL_DIM,
        pgc_configs=pgc_config,
        num_s4=1,
        d_input=INPUT_DIM,
        d_output=OUTPUT_DIM
    )
    
    # Dummy Input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    
    # Forward Pass
    y = model(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
