# Usage

This guide provides a basic workflow for setting up and training with Cerberus.

## 1. Setup Configuration

Define your genome, data, and sampling strategy. This example assumes you have run `cerberus.download.download_human_reference("tests/data")` and `cerberus.download.download_dataset("tests/data", "mdapca2b_ar")`.

```python
from pathlib import Path
from cerberus.genome import create_genome_config

# Paths (based on download_human_reference and download_dataset defaults)
data_dir = Path("tests/data")
fasta_path = data_dir / "hg38/hg38.fa"
peaks_path = data_dir / "mdapca2b_ar/mdapca2b-ar.narrowPeak.gz"
signal_path = data_dir / "mdapca2b_ar/mdapca2b-ar.bigwig"
blacklist_path = data_dir / "hg38/blacklist.bed"

# 1. Genome Configuration
genome_config = create_genome_config(
    name="hg38",
    fasta_path=fasta_path,
    species="human",
    # Exclude blacklist regions
    exclude_intervals={"blacklist": blacklist_path},
    fold_type="chrom_partition",
    fold_args={"k": 5}
)

# 2. Data Configuration
data_config = {
    "inputs": {},  # Only sequence input
    "targets": {"AR": signal_path},
    "input_len": 2114,
    "output_len": 1000,
    "output_bin_size": 1,
    "encoding": "ACGT",
    "max_jitter": 128,
    "log_transform": True,
    "reverse_complement": True,
    "use_sequence": True
}

# 3. Sampler Configuration (Peaks + Negatives)
sampler_config = {
    "sampler_type": "multi",
    "padded_size": 2114,
    "sampler_args": {
        "samplers": [
            # Positive examples (peaks)
            {
                "type": "interval",
                "args": {"intervals_path": peaks_path},
                "scaling": 1.0 
            },
            # Negative examples (background)
            {
                "type": "sliding_window",
                "args": {"stride": 10000}, # Sparse scan
                "scaling": 0.1 # Downsample heavily
            }
        ]
    }
}
```

## 2. Instantiate DataModule

Create the `CerberusDataModule`. This handles multiprocessing and data splitting.

```python
from cerberus import CerberusDataModule

data_module = CerberusDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    pin_memory=True,            # Auto-disabled on MPS
    persistent_workers=True,    # Faster worker initialization
    multiprocessing_context=None # e.g. 'spawn' for MPS stability
)

# Setup datasets (split into train/val/test)
# You can set the batch size and num_workers here
data_module.setup(batch_size=256, num_workers=8)
```

## 3. Train with PyTorch Lightning

You can use the `cerberus.train` module to instantiate the model and start training easily.

```python
import torch.nn as nn
from cerberus.train import train_single, train_multi
from cerberus.loss import BPNetLoss
from torchmetrics import MetricCollection, PearsonCorrCoef, MeanSquaredError

# 4. Train Configuration
train_config = {
    "batch_size": 256,
    "max_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 10,
    "optimizer": "adamw",
    "scheduler_type": "cosine",
    "scheduler_args": {"warmup_epochs": 5},
    "filter_bias_and_bn": True
}

# 5. Model Configuration
# Define your PyTorch model class (must accept input_channels, output_channels, input_len)
# It will also receive output_len and output_bin_size automatically.
class MyModel(nn.Module):
    def __init__(self, input_channels, output_channels, input_len, **kwargs):
        super().__init__()
        num_in = len(input_channels)
        num_out = len(output_channels)
        self.conv = nn.Conv1d(num_in, num_out, 1)
    def forward(self, x): return self.conv(x)

model_config = {
    "name": "my_bpnet",
    "model_cls": MyModel,
    "loss_cls": BPNetLoss,
    "loss_args": {"alpha": 1.0},
    "metrics_cls": MetricCollection,
    "metrics_args": {
        "metrics": {
            "pearson": PearsonCorrCoef(),
            "mse_profile": MeanSquaredError()
        }
    },
    "model_args": {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["AR"]
    }
}

# Option A: Train a Single Model (Single Split)
# Uses the high-level API to handle instantiation and output structure (creates fold_0)
# (test_fold defaults to 0, val_fold defaults to 1)
trainer = train_single(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,
    train_config=train_config,
    num_workers=8, 
    in_memory=False, 
    precision="16-mixed",      # Enable Mixed Precision
    matmul_precision="high",   # TensorFloat-32 on Ampere+
    root_dir="logs/single_run",
    accelerator="gpu", 
    devices=1
)

# Option B: Cross-Validation Training (Multi-Fold)
# Trains k models (where k is defined in genome_config), each with a different test fold.
trainers = train_multi(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,
    train_config=train_config,
    num_workers=8,
    precision="16-mixed",
    matmul_precision="high",
    root_dir="logs/cross_val", # Models saved in logs/cross_val/fold_0, fold_1, etc.
    accelerator="gpu",
    devices=1
)
```

### Logging

Cerberus automatically logs all configuration dictionaries (`genome_config`, `data_config`, `sampler_config`, `model_config`, `train_config`) to the experiment log directory.
You can find these parameters in:
`lightning_logs/version_{#}/hparams.yaml`

## Manual Usage (PyTorch Dataset)

If you aren't using Lightning, you can use `CerberusDataset` directly.

```python
from cerberus import CerberusDataset
from torch.utils.data import DataLoader

dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config
)

# Access a single item
item = dataset[0]
print(item['inputs'].shape)  # (4, 2114) -> Sequence (one-hot)
print(item['targets'].shape) # (1, 1000) -> Signal

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Examples

The `notebooks/` directory contains complete examples:

*   `notebooks/cerberus_basics.py`: A step-by-step walkthrough of the library components (Configuration, Samplers, Datasets, Transforms).
*   `notebooks/baseline_cnn_train.py`: A complete training example using the `GlobalProfileCNN` model to predict BigWig tracks from DNA sequence.

## Next Steps

Once you have trained a model, you can use the prediction tools to evaluate it or generate genome-wide tracks.
See [Prediction (Inference)](prediction.md) for details on `ModelEnsemble`, `predict_intervals_batched`, and CLI tools like `tools/export_predictions.py`.
