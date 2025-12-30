# %% [markdown]
# # Chip-AR MDAPCA2b Prediction Notebook (BPNet)
#
# This notebook demonstrates how to load a trained BPNet model and run predictions on genomic intervals.
# It is based on the `examples/chip_ar_mdapca2b.py` training script (BPNet configuration).

# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from torchmetrics import MetricCollection

# Add project root to path if needed to import notebooks.paths
import sys
try:
    from paths import get_project_root
except ImportError:
    # If running from root, add notebooks/ to path
    sys.path.append("notebooks")
    from paths import get_project_root

from cerberus.download import download_dataset, download_human_reference
from cerberus.config import (
    GenomeConfig, 
    DataConfig, 
    SamplerConfig, 
    TrainConfig, 
    ModelConfig,
    PredictConfig
)
from cerberus.genome import create_genome_config
from cerberus.models.bpnet import BPNet
from cerberus.loss import BPNetLoss, get_bpnet_metrics
from cerberus.dataset import CerberusDataset
from cerberus.model_manager import ModelManager
from cerberus.predict import predict_intervals

# %% [markdown]
# ## 1. Setup Directories and Data
#
# We define the data and checkpoint directories.
# Make sure you have trained the BPNet model using `python examples/chip_ar_mdapca2b.py --model bpnet` first.

# %%
project_root = get_project_root()
data_dir = project_root / "tests/data"
# BPNet Checkpoint Directory
checkpoint_dir = project_root / "tests/data/models/bpnet-chip_ar_mdapca2b/peaks-single"

print(f"Data Directory: {data_dir}")
print(f"Checkpoint Directory: {checkpoint_dir}")

if not checkpoint_dir.exists():
    raise FileNotFoundError(
        f"Checkpoint directory {checkpoint_dir} does not exist.\n"
        "Please run 'python examples/chip_ar_mdapca2b.py --model bpnet' to train the model first."
    )

# %%
# Download/Check Data
print("Checking Data...")
genome_files = download_human_reference(data_dir / "genome", name="hg38")
dataset_files = download_dataset(data_dir / "dataset", name="mdapca2b_ar")

# %% [markdown]
# ## 2. Configuration
#
# We setup the configuration for BPNet.
# Key differences from Baseline:
# - `input_len=2114`, `output_len=1000`.
# - `output_bin_size=1`.
# - `log_transform=False` (Raw counts).

# %%
# Genome Config
genome_config: GenomeConfig = create_genome_config(
    name="hg38",
    fasta_path=genome_files["fasta"],
    species="human",
    fold_type="chrom_partition",
    fold_args={"k": 5, "val_fold": 1, "test_fold": 0},
    exclude_intervals={
        "blacklist": genome_files["blacklist"],
        "gaps": genome_files["gaps"],
    }
)

# Data Config for BPNet
input_len = 2114
output_len = 1000
output_bin_size = 1
log_transform = False

data_config: DataConfig = {
    "inputs": {}, 
    "targets": {"signal": dataset_files["bigwig"]},
    "input_len": input_len,
    "output_len": output_len, 
    "max_jitter": 0,          # Disabled for prediction
    "output_bin_size": output_bin_size,
    "encoding": "ACGT",
    "log_transform": log_transform,
    "reverse_complement": False, # Disabled for prediction
    "use_sequence": True,
}

# Sampler Config
sampler_config: SamplerConfig = {
    "sampler_type": "interval",
    "padded_size": input_len, # Exact input length
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"]
    }
}

# Model/Train Config
train_config: TrainConfig = {
    "batch_size": 64,
    "max_epochs": 1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "patience": 1,
    "optimizer": "adamw",
    "filter_bias_and_bn": True,
    "scheduler_type": "cosine",
    "scheduler_args": {}
}

model_config: ModelConfig = {
    "name": "BPNet",
    "model_cls": BPNet,
    "loss_cls": BPNetLoss,
    "loss_args": {
        "count_weight": 1.0, 
        "flatten_channels": False,
        "implicit_log_targets": log_transform
    },
    "metrics_cls": cast(type[MetricCollection], get_bpnet_metrics),
    "metrics_args": {"num_channels": 1},
    "model_args": {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "filters": 64,
        "n_dilated_layers": 8
    }
}

# %% [markdown]
# ## 3. Initialize Dataset and Sampler
#
# We initialize the dataset and split it to get the test fold sampler.

# %%
print("Initializing Dataset...")
dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config
)

print("Splitting folds (Test Fold = 0)...")
_, _, test_dataset = dataset.split_folds(test_fold=0, val_fold=1)
test_sampler = test_dataset.sampler

print(f"Number of test intervals: {len(test_sampler)}")

# %% [markdown]
# ## 4. Load Model
#
# We use `ModelManager` to load the trained model from the checkpoint.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

print("Loading Model...")
model_manager = ModelManager(
    checkpoint_path=checkpoint_dir,
    model_config=model_config,
    data_config=data_config,
    train_config=train_config,
    genome_config=genome_config,
    device=device
)

# %% [markdown]
# ## 5. Run Prediction
#
# We predict on a subset of the test intervals.

# %%
# Select a single interval for demonstration
target_idx = 0
target_interval = test_sampler[target_idx]
intervals_to_predict = [target_interval]

print(f"Predicting on interval: {target_interval}")

predict_config: PredictConfig = {
    "stride": output_len,
    "intervals": [],
    "intervals_paths": [],
    "use_folds": ["test"], 
    "aggregation": "mean"
}

outputs, merged_interval = predict_intervals(
    intervals=intervals_to_predict,
    dataset=dataset, 
    model_manager=model_manager,
    predict_config=predict_config,
    device=str(device),
    batch_size=64
)

print("Prediction Complete!")
print(f"Merged Interval: {merged_interval}")

# %% [markdown]
# ## 6. Analyze Results
#
# BPNet returns `(profile_logits, log_counts)`.
# `predict_intervals` aggregates these into tracks.
# - Track 0: `profile_logits` (shape 1x1000)
# - Track 1: `log_counts` (scalar broadcasted to 1x1000)

# %%
# Extract Ground Truth
if dataset.target_signal_extractor is None:
    raise ValueError("Target signal extractor is missing.")

gt_targets = dataset.target_signal_extractor.extract(merged_interval)
print(f"Ground Truth Shape: {gt_targets.shape}")

# Inspect Outputs
for i, track in enumerate(outputs):
    print(f"\nOutput Track {i}:")
    print(f"  Shape: {track.shape}")
    print(f"  Min: {track.min():.4f}, Max: {track.max():.4f}, Mean: {track.mean():.4f}")

# %% [markdown]
# ### Visualization
#
# We compare the predicted profile (converted to counts) with the ground truth.

# %%
# Visualizing the interval
print(f"Visualizing interval: {target_interval}")

output_bin_size = data_config["output_bin_size"]
rel_start_bp = target_interval.center(output_len).start - merged_interval.start
rel_start_bin = rel_start_bp // output_bin_size
n_bins = output_len // output_bin_size

print(f"Relative Start Bin: {rel_start_bin}")
print(f"Number of Bins: {n_bins}")

# Extract prediction tracks
# Track 0: Profile Logits
logits = outputs[0][:, rel_start_bin : rel_start_bin + n_bins]
logits_tensor = torch.from_numpy(logits)

# Track 1: Log Total Counts (constant over interval)
log_counts = outputs[1][:, rel_start_bin : rel_start_bin + n_bins]
log_counts_val = torch.tensor(log_counts[0, 0]) # Take first value
total_counts = torch.exp(log_counts_val)

print(f"Predicted Total Counts: {total_counts.item():.2f}")

# Convert Logits to Probability Profile
probs = F.softmax(logits_tensor, dim=-1)

# Predicted Profile in Counts
pred_counts_profile = probs * total_counts

# Extract Ground Truth (Raw Counts)
# gt_targets is in bp resolution. BPNet output is also bp resolution (bin_size=1).
gt_start_bp = rel_start_bin * output_bin_size
gt_end_bp = gt_start_bp + n_bins * output_bin_size
gt_slice_bp = gt_targets[:, gt_start_bp : gt_end_bp]

# No binning needed if output_bin_size=1
gt_slice = gt_slice_bp

print(f"Ground Truth Total Counts: {gt_slice.sum().item():.2f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. Ground Truth
axes[0].plot(gt_slice[0], label="Ground Truth (Counts)", color="blue")
axes[0].set_title("Ground Truth (Observed Counts)")
axes[0].legend()

# 2. Predicted Profile (Counts)
axes[1].plot(pred_counts_profile[0], label="Predicted (Counts)", color="green")
axes[1].set_title("Predicted Profile (Counts)")
axes[1].legend()

plt.xlabel("Base Pairs")
plt.tight_layout()
plt.show()

# %%
