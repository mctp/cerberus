# %% [markdown]
# # Cerberus DataModule Setup
# 
# This notebook demonstrates how to set up a Cerberus DataModule end-to-end using the example data located in `tests/data`.
# It covers configuration of the genome, dataset, and sampler, and shows how to inspect the resulting data batches.

# %%
import os
import sys
import pprint
from pathlib import Path
import torch

from cerberus.genome import create_genome_config
from cerberus import CerberusDataModule
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig

try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

project_root = get_project_root()
print(f"Project root: {project_root}")

# %% [markdown]
# ## 1. Define Paths
# 
# We assume the data has been downloaded to `tests/data` (as done by `tests/conftest.py` or manually).

# %%
data_dir = project_root / "tests/data"  # Default
print(f"Using data directory: {data_dir.resolve()}")

# Verify paths exist
genome_dir = data_dir / "genome" / "hg38"
dataset_dir = data_dir / "dataset" / "mdapca2b_ar"

fasta_path = genome_dir / "hg38.fa"
blacklist_path = genome_dir / "blacklist.bed"
mappability_path = genome_dir / "mappability.bw"
encode_cre_path = genome_dir / "encode_cre.bb"

peaks_path = dataset_dir / "mdapca2b-ar.narrowPeak.gz"
signal_path = dataset_dir / "mdapca2b-ar.bigwig"

print(f"FASTA path: {fasta_path} (Exists: {fasta_path.exists()})")
print(f"Blacklist path: {blacklist_path} (Exists: {blacklist_path.exists()})")
print(f"Mappability path: {mappability_path} (Exists: {mappability_path.exists()})")
print(f"ENCODE cRE path: {encode_cre_path} (Exists: {encode_cre_path.exists()})")
print(f"Peaks path: {peaks_path} (Exists: {peaks_path.exists()})")
print(f"Signal path: {signal_path} (Exists: {signal_path.exists()})")

# Check if all required paths exist, raise error if any are missing
missing_paths = []
required_paths = {
    "FASTA": fasta_path,
    "Blacklist": blacklist_path,
    "Mappability": mappability_path,
    "ENCODE cRE": encode_cre_path,
    "Peaks": peaks_path,
    "Signal": signal_path
}

for name, path in required_paths.items():
    if not path.exists():
        missing_paths.append(f"{name}: {path}")

if missing_paths:
    raise FileNotFoundError(
        f"The following required files are missing:\n" + "\n".join(f"  - {p}" for p in missing_paths)
    )

print("✓ All required files exist")

# %% [markdown]
# ## 2. Genome Configuration
# 
# Configure the reference genome, including species, fasta file, and exclusion regions (blacklist).
# We also define the fold splitting strategy here.

# %%
genome_config: GenomeConfig = create_genome_config(
    name="hg38",
    fasta_path=fasta_path,
    species="human",
    # Exclude blacklist regions
    exclude_intervals={"blacklist": blacklist_path},
    # Use chromosome partitioning for train/val/test splits
    fold_type="chrom_partition",
    fold_args={"k": 5}
)

print("Genome Config Created:")
pp = pprint.PrettyPrinter(width=160, compact=True)
pp.pprint(genome_config)
# %% [markdown]
# ## 3. Data Configuration
# 
# Define the input and output tracks. 
# - `inputs`: Here we rely on sequence (implied), so it's empty.
# - `targets`: The signal tracks we want to predict (AR ChIP-seq signal).
# - `input_len`: Length of the input sequence (e.g., 2114 bp).
# - `output_len`: Length of the output prediction (e.g., 1000 bp).
# - `bin_size`: Resolution of the output.

# %%
data_config: DataConfig = {
    "inputs": {"mappability": mappability_path},  # Only sequence input
    "targets": {"AR": signal_path},
    "input_len": 2048,
    "output_len": 1024,
    "output_bin_size": 4,
    "encoding": "ACGT",
    "max_jitter": 128,
    "log_transform": True,
    "reverse_complement": True,
    "use_sequence": True,
}
print("Data Config Created:")
pp.pprint(data_config)

# %% [markdown]
# ## 4. Sampler Configuration
# 
# Define how examples are sampled from the genome.
# We use a `multi` sampler to combine positive examples (peaks) and negative examples (background).

# %%
# padded_size must be >= input_len + 2 * max_jitter
# input_len = 2048, max_jitter = 128
# required = 2048 + 2 * 128 = 2304
sampler_config: SamplerConfig = {
    "sampler_type": "interval",
    "padded_size": 2304,
    "sampler_args": {
        "intervals_path": peaks_path
    }
}

# %% [markdown]
# ## 5. Instantiate DataModule
# 
# Create the `CerberusDataModule` which manages the datasets and dataloaders.

# %%
data_module = CerberusDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    pin_memory=False
)

# %% [markdown]
# ## 6. Setup Datasets
# 
# `setup()` prepares the train, validation, and test datasets based on the genome fold configuration.
# We also set the batch size and number of workers here.

# %%
# This initializes the underlying datasets and performs the split
# We can also control memory usage here (in_memory=True/False)
data_module.setup(batch_size=8, num_workers=0, in_memory=False)

if data_module.train_dataset:
    print("Train dataset length:", len(data_module.train_dataset))
if data_module.val_dataset:
    print("Val dataset length:", len(data_module.val_dataset))
if data_module.test_dataset:
    print("Test dataset length:", len(data_module.test_dataset))

# %% [markdown]
# ## 7. Inspect a Batch
# 
# Retrieve a batch from the training dataloader and inspect the shapes and content.

# %%
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

inputs = batch['inputs']
targets = batch['targets']
intervals = batch['intervals']

print("Batch keys:", batch.keys())

print(f"Inputs type: {type(inputs)}")
if isinstance(inputs, torch.Tensor):
    print(f"Inputs shape: {inputs.shape}")

print(f"Targets type: {type(targets)}")
if isinstance(targets, torch.Tensor):
    print(f"Targets shape: {targets.shape}")


# print(f"Targets shape: {targets['AR'].shape if isinstance(targets, dict) else targets.shape}")

# %% [markdown]
# ## 8. Visualize (Optional)
# 
# Simple visualization of the signal.

# %%
import matplotlib.pyplot as plt

target_signal = targets.numpy().mean(axis=0).squeeze()

x_axis = range(0, 256 * 4, 4)
plt.figure(figsize=(12, 4))
plt.plot(x_axis, target_signal, linewidth=1)
plt.xlabel('Genomic Position (bp)')
plt.ylabel('Signal')
plt.title('Target Signal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
