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
    "bin_size": 1,
    "encoding": "ACGT",
    "max_jitter": 128,
    "log_transform": True,
    "reverse_complement": True,
    "in_memory": False
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
    batch_size=256,
    num_workers=8,
    pin_memory=True
)

# Setup datasets (split into train/val/test)
data_module.setup()
```

## 3. Train with PyTorch Lightning

Pass the `data_module` directly to the PyTorch Lightning Trainer.

```python
import pytorch_lightning as pl
from my_model import MyGenomicModel # Your PyTorch Lightning Module

model = MyGenomicModel(...)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100)

# Trainer automatically calls train_dataloader() and val_dataloader()
trainer.fit(model, datamodule=data_module)
```

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
