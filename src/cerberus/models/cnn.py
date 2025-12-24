import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self, window_size=2048, bin_size=4, nr_tracks=1):
        super().__init__()
        assert window_size % bin_size == 0
        nr_bins = window_size // bin_size

        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, 15),
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

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 13, 1024),
            nn.ReLU(),
            nn.Linear(1024, nr_bins),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, -1, -2)
        x = self.stem(x)
        x = self.core(x)
        x = torch.transpose(x, -1, -2)
        x = self.head(x)
        x = x.reshape(x.shape + (1,))
        return x
