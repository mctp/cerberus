import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    """
    A vanilla Convolutional Neural Network (CNN) for genomic sequence modeling.

    This model consists of a stem, a core body of convolutional layers with pooling,
    and a regression head. It is designed to take one-hot encoded DNA sequences
    as input and predict signal profiles or other genomic properties.

    The network structure dynamically adapts the first linear layer in the head
    to the size of the feature map produced by the convolutional layers, making
    it flexible to different input lengths (subject to a minimum length constraint
    imposed by pooling layers, approx. 1254 bp).

    Args:
        input_len (int): Length of the input sequence. Defaults to 2048.
        output_len (int): Length of the output sequence (in base pairs). Defaults to 1024.
        bin_size (int): Resolution of the output predictions. The final output dimension
            will be `output_len // bin_size`. Defaults to 4.
        num_input_channels (int): Number of input channels (e.g., 4 for one-hot DNA). Defaults to 4.
        num_output_channels (int): Number of output tracks/channels to predict. Defaults to 1.
    """
    def __init__(self, input_len=2048, output_len=1024, output_bin_size=4, num_input_channels=4, num_output_channels=1):
        super().__init__()
        assert output_len % output_bin_size == 0
        
        self.nr_bins = output_len // output_bin_size
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        self.stem = nn.Sequential(
            nn.Conv1d(num_input_channels, 256, 15),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2)
        )
    
        self.core = nn.Sequential(
            nn.Conv1d(256, 512, 15),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            nn.Conv1d(512, 512, 15),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            nn.Conv1d(512, 512, 15),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_input_channels, input_len)
            dummy_out = self.core(self.stem(dummy_input))
            flatten_size = dummy_out.view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 1024),
            nn.ReLU(),
            # Output total number of predictions: bins * output_channels
            # We multiply by num_output_channels to produce predictions for all output channels simultaneously,
            # whereas the previous implementation implicitly assumed a single output channel.
            nn.Linear(1024, self.nr_bins * self.num_output_channels),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected: (Batch, 4, Length)
        x = self.stem(x)
        x = self.core(x)
        # Core output: (Batch, 512, 13)
        # Flatten will make it (Batch, 512 * 13)
        x = self.head(x)
        # Head output: (Batch, nr_bins * num_output_channels)
        # Reshape to (Batch, Output Channels, Bins) to match DataModule target shape
        x = x.view(x.shape[0], self.num_output_channels, self.nr_bins)
        return x
