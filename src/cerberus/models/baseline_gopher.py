import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalProfileCNN(nn.Module):
    """
    GlobalProfileCNN (Baseline CNN) for genomic profile prediction.
    
    This architecture corresponds to the "Baseline CNN" described in the Gopher manuscript
    (specifically `conv_profile_all_base`). It uses a standard convolutional body followed
    by a global dense bottleneck and a rescaling layer that projects the global representation
    back to a spatial grid.
    
    Key Features:
    - 3 Convolutional Blocks with Max Pooling.
    - Global Dense Bottleneck.
    - Global Projection to (Output_Length, Bottleneck_Channels).
    - Shared Convolutional Head.
    
    Constraints:
    - Input length should ideally be divisible by 128 (8*4*4 pooling).
    - The model structure (Dense layers) depends on the specific `input_len` provided at initialization.
      Changing `input_len` requires re-initialization.
    
    Args:
        input_len (int): Length of the input sequence. Defaults to 2048.
        output_len (int): Length of the output sequence/profile (in base pairs). Defaults to 1024.
        output_bin_size (int): Resolution of the output predictions. Defaults to 1.
                               Note: The number of prediction bins will be `output_len // output_bin_size`.
        num_input_channels (int): Number of input channels (e.g., 4 for one-hot DNA). Defaults to 4.
        num_output_channels (int): Number of output tracks/channels to predict. Defaults to 1.
        bottleneck_channels (int): Number of channels in the reshaped feature map. Defaults to 8.
    """
    def __init__(
        self, 
        input_len=2048, 
        output_len=1024, 
        output_bin_size=4, 
        num_input_channels=4, 
        num_output_channels=1,
        bottleneck_channels=8
    ):
        super().__init__()
        
        # Calculate number of bins based on output length (in bp) and bin size
        assert output_len % output_bin_size == 0, "output_len must be divisible by output_bin_size"
        self.nr_bins = output_len // output_bin_size
        
        self.num_output_channels = num_output_channels
        self.bottleneck_channels = bottleneck_channels
        
        # 1. Conv Block 1
        # TF: Conv1D(192, 19, same) -> BN -> Act -> MaxPool(8) -> Drop(0.1)
        self.block1 = nn.Sequential(
            nn.Conv1d(num_input_channels, 192, kernel_size=19, padding='same'),
            nn.BatchNorm1d(192, momentum=0.1), # PT 0.1 ~ TF 0.9
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Dropout(0.1)
        )
        
        # 2. Conv Block 2
        # TF: Conv1D(256, 7, same) -> BN -> Act -> MaxPool(4) -> Drop(0.1)
        self.block2 = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=7, padding='same'),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.1)
        )
        
        # 3. Conv Block 3
        # TF: Conv1D(512, 7, same) -> BN -> Act -> MaxPool(4) -> Drop(0.2)
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=7, padding='same'),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2)
        )
        
        # Calculate Flatten Size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_input_channels, input_len)
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.block3(x)
            flatten_size = x.view(1, -1).size(1)
            
        # 4. Dense Bottleneck
        # TF: Dense(256) -> BN -> Act -> Drop(0.3)
        self.dense1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 5. Global Projection (Rescaling)
        # TF: Dense(Out_Len * Bottleneck) -> BN -> Act -> Reshape -> Drop(0.1)
        # Target shape for Reshape in PT: (Batch, Bottleneck, Out_Len)
        self.projection_size = self.nr_bins * bottleneck_channels
        self.dense2 = nn.Sequential(
            nn.Linear(256, self.projection_size),
            nn.BatchNorm1d(self.projection_size, momentum=0.1),
            nn.ReLU(),
            # Reshape happens in forward
            nn.Dropout(0.1)
        )
        
        # 6. Final Conv Block
        # TF: Conv1D(256, 7, same) -> BN -> Act -> Drop(0.2)
        # Input channels: bottleneck_channels
        self.final_conv = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 256, kernel_size=7, padding='same'),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 7. Output Head
        # TF: Dense(num_tasks, softplus)
        # In TF this is applied to (Batch, Time, Features). 
        # In PT (Batch, Features, Time), we use Conv1d(1) to map Features -> Num_Tasks
        # We output Logits (Linear) to use with PoissonNLLLoss(log_input=True).
        self.head = nn.Conv1d(256, num_output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length)
        
        # Conv Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Dense Bottleneck
        x = self.dense1(x)
        
        # Global Projection
        x = self.dense2(x) # (Batch, Out_Len * Bottleneck)
        
        # Reshape to (Batch, Bottleneck, Out_Len)
        # Note: In TF code it reshapes to (Out_Len, Bottleneck).
        # Since Dense layer connects everything to everything, the specific reshaping layout 
        # is just a convention. We choose (Bottleneck, Out_Len) to match Conv1d expectation.
        x = x.view(x.shape[0], self.bottleneck_channels, self.nr_bins)
        
        # Final Conv
        x = self.final_conv(x)
        
        # Head (Logits)
        x = self.head(x)
        
        # Output: (Batch, Num_Output_Channels, Out_Len)
        return x
