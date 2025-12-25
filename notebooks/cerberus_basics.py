# %%
# # Cerberus Library Walkthrough
# 
# This notebook provides a step-by-step introduction to the `cerberus` library.
# We will cover:
# 1. Genome configuration and sequence extraction
# 2. Signal extraction from BigWig files
# 3. Samplers for defining training intervals
# 4. Creating a Dataset
# 5. Using DataModules and DataLoaders
# 
# We will use the **hg38** genome and the **mdapca2b_ar** dataset (example files included in tests).

# %%
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

from cerberus.genome import download_human_reference, create_human_genome_config
from cerberus.sequence import SequenceExtractor
from cerberus.signal import SignalExtractor
from cerberus.samplers import create_sampler
from cerberus.dataset import CerberusDataset
from cerberus.datamodule import CerberusDataModule
from cerberus.core import Interval
from cerberus.exclude import get_exclude_intervals

# Set up paths
project_root = get_project_root()
DATA_DIR = project_root / "tests/data"
GENOME_DIR = DATA_DIR / "genome"
GENOME_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Genome directory: {GENOME_DIR}")

# %%
# ## 1. Genome Configuration
# 
# First, we need to set up the reference genome. Cerberus provides utilities to download and configure genomes.
# Here we download the hg38 reference (FASTA, blacklist, gaps).
# 
# **Note:** The download might take a few minutes.

# %%
from pprint import pprint

# Download hg38 reference if not already present
# This downloads hg38.fa.gz, blacklist.bed.gz, and gap.txt.gz
try:
    genome_files = download_human_reference(GENOME_DIR, name="hg38")
    print("Genome files:", genome_files)
except Exception as e:
    print(f"Error or already downloaded: {e}")
    # If download fails or you have it elsewhere, ensure the config can be created below.

# Create configuration object
# Note: download_human_reference(genome_dir) downloads into genome_dir directly
# but create_human_genome_config expects the directory containing hg38.fa
genome_config = create_human_genome_config(
    genome_dir=GENOME_DIR / "hg38",
    fold_type="chrom_partition",
    fold_args={"k": 5}  # 5-fold cross validation split
)

print("Genome Config:")
pprint(genome_config)

# %%
# ## 2. Sequence Extraction
# 
# With the genome configured, we can extract DNA sequences. The `SequenceExtractor` handles this efficiently.

# %%
import pyfaidx

# Initialize SequenceExtractor
seq_extractor = SequenceExtractor(
    fasta_path=genome_config["fasta_path"],
    encoding="ACGT"  # Can be 'ACGT'
)

# Define an interval
interval = Interval("chr1", 1000000, 1000050, "+")
print(f"Extracting sequence for: {interval}")

# Extract
seq = seq_extractor.extract(interval)
print("Sequence Shape:", seq.shape) # (4, 50) for one-hot
print("First 10 bases (one-hot):\n", seq[:, :10])

# Extract using pyfaidx directly for comparison
genome = pyfaidx.Fasta(str(genome_config["fasta_path"]))
# pyfaidx uses 0-based indexing like we do, but let's verify exact interval
# Interval is [start, end)
seq_str = genome[interval.chrom][interval.start:interval.end].seq
print("\nCharacter Sequence (pyfaidx):")
print(seq_str[:10])  # First 10 bases

# %%
# ## 3. Signal Extraction
# 
# We can extract continuous signals (e.g., read counts, fold change) from BigWig files using `SignalExtractor`.
# We'll use the example BigWig file from the tests.

# %%
bigwig_path = DATA_DIR / "dataset" / "mdapca2b_ar" / "mdapca2b-ar.bigwig"
print(f"Using BigWig: {bigwig_path}")

# Initialize SignalExtractor
# You can provide multiple tracks mapping channel names to paths
signal_extractor = SignalExtractor(
    bigwig_paths={"atac": bigwig_path}
)

# Extract signal for the same interval
# Note: The interval must be within the BigWig's range
# Let's pick a region that likely has signal or just test functionality
# chr8:123,405,092-123,406,230
interval = Interval("chr8", 123405092, 123406230, "+")
print(f"Extracting signal for: {interval}")
signal = signal_extractor.extract(interval)
print("Signal Shape:", signal.shape) # (Channels, Length)
print("Signal Values (first 10):", signal[0, (100, 500, 750, 1000, 1100)])

# %%
# ## 4. Samplers
# 
# Samplers define *where* in the genome we extract data from. Cerberus supports:
# - `IntervalSampler`: From BED/narrowPeak files (e.g., peaks)
# - `SlidingWindowSampler`: Tiles across the genome
# - `MultiSampler`: Combines multiple samplers (e.g., peaks + background)
# 
# ### Interval Sampler (Peaks)

# %%
peaks_path = DATA_DIR / "dataset" / "mdapca2b_ar" / "mdapca2b-ar.narrowPeak.gz"

import gzip
print(f"Reading first 3 lines of {peaks_path}:")
with gzip.open(peaks_path, "rt") as f:
    count = 0
    for line in f:
        if line.startswith("#"):
            continue
        print(line.strip())
        count += 1
        if count >= 3:
            break
print("-" * 20)

# Basic config for IntervalSampler
sampler_config_peaks = {
    "sampler_type": "interval",
    "padded_size": 1000, # Resize intervals to this width
    "sampler_args": {
        "intervals_path": peaks_path
    }
}

# We need the fold configuration from the genome config
from cerberus.genome import create_genome_folds
folds = create_genome_folds(
    genome_config["chrom_sizes"],
    genome_config["fold_type"],
    genome_config["fold_args"]
)

# Create the sampler
# Note: We need to load exclude intervals (InterLap objects) from paths
exclude_intervals = get_exclude_intervals(genome_config["exclude_intervals"])

peak_sampler = create_sampler(
    sampler_config_peaks,
    genome_config["chrom_sizes"],
    exclude_intervals,
    folds
)

print(f"Number of peaks: {len(peak_sampler)}")
print("First sample:", peak_sampler[0])
print("Second sample:", peak_sampler[1])
print("Third sample:", peak_sampler[2])


# %%
# ### Sliding Window Sampler

# %%
sampler_config_sw = {
    "sampler_type": "sliding_window",
    "padded_size": 1000,
    "sampler_args": {
        "stride": 50000 # Large stride for demo purposes
    }
}

sw_sampler = create_sampler(
    sampler_config_sw,
    genome_config["chrom_sizes"],
    exclude_intervals,
    folds
)

print(f"Number of sliding windows: {len(sw_sampler)}")
print("First window:", sw_sampler[0])
print("Second window:", sw_sampler[1])
print("Third window:", sw_sampler[2])

# %%
# ## 5. CerberusDataset
# 
# The `CerberusDataset` brings it all together:
# 1. Genome Config
# 2. Data Config (Inputs/Targets)
# 3. Sampler Config
# 
# It handles splitting into train/val/test folds automatically.

# %%
data_config = {
    "encoding": "ACGT",
    "inputs": {
        "atac": bigwig_path
    },
    "targets": {
        "atac": bigwig_path # Using same file as target for demo (autoencoder style)
    },
    "input_len": 1000, # Should match padded_size from sampler
    "output_len": 1000,
    "output_bin_size": 1,
    "max_jitter": 0,
    "log_transform": False,
    "reverse_complement": False
}

dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config_peaks,
    in_memory=False, # Set True to load everything into RAM (faster for small datasets)
)

# Split folds
train_ds, val_ds, test_ds = dataset.split_folds(test_fold=0, val_fold=1)

print(f"Train size: {len(train_ds)}")
print(f"Val size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")

# Get an item
item = train_ds[0]
print("\nDataset Item Keys:", item.keys())
print("Inputs shape:", item["inputs"].shape)   # (4 + n_input_tracks, Length)
print("Targets shape:", item["targets"].shape) # (n_target_tracks, Length)
print("Interval:", item["intervals"])

# %%
# ## 6. CerberusDataModule
# 
# For PyTorch Lightning integration, `CerberusDataModule` wraps the dataset and handles DataLoaders.

# %%
dm = CerberusDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config_peaks,
)

dm.setup(batch_size=8, num_workers=0, in_memory=False)

train_loader = dm.train_dataloader()
batch = next(iter(train_loader))

print("Batch Inputs:", batch["inputs"].shape)
print("Batch Targets:", batch["targets"].shape)
print("Batch Intervals:", batch["intervals"])

# %%
# ## 7. Experimenting with Parameters
# 
# Let's see how changing the `padded_size` affects the output shapes.

# %%
# Change padded size to 200
sampler_config_small = sampler_config_peaks.copy()
sampler_config_small["padded_size"] = 200

# Also update data config to match the new size
data_config_small = data_config.copy()
data_config_small["input_len"] = 200
data_config_small["output_len"] = 200

ds_small = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config_small,
    sampler_config=sampler_config_small,
    in_memory=False
)

item_small = ds_small[0]
print("New Inputs shape:", item_small["inputs"].shape)
print("New Targets shape:", item_small["targets"].shape)

# %%
# ## 8. Transforms
# 
# Cerberus provides data transformations like Jitter, ReverseComplement, and signal binning.
# These can be manually composed or automatically created from `DataConfig`.

# %%
from cerberus.transform import Jitter, ReverseComplement, Compose

# Create a sample batch (inputs, targets, interval)
# Inputs: (4+1, 1000) - 4 DNA channels + 1 signal channel
# Targets: (1, 1000)
# Interval: dummy
inputs = torch.randn(5, 1000)
targets = torch.randn(1, 1000)
interval = Interval("chr1", 1000, 2000, "+")

print("Original Interval:", interval)
print("Original Inputs shape:", inputs.shape)

# Create a Jitter transform (crop to 500bp with random offset)
jitter = Jitter(input_len=500, max_jitter=None) # None = use full slack

# Apply transform
t_inputs, t_targets, t_interval = jitter(inputs, targets, interval)

print("\nAfter Jitter(500):")
print("New Interval:", t_interval)
print("New Inputs shape:", t_inputs.shape)
print("Offset:", t_interval.start - 1000)

# Create ReverseComplement transform
rc = ReverseComplement(probability=1.0) # Always apply

t_inputs_rc, t_targets_rc, t_interval_rc = rc(t_inputs, t_targets, t_interval)

print("\nAfter ReverseComplement:")
print("New Interval:", t_interval_rc) # Strand should flip
print("New Inputs shape:", t_inputs_rc.shape)
# %%
