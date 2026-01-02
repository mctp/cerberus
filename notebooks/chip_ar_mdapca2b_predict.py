# %% [markdown]
# # Chip-AR MDAPCA2b Prediction Notebook
#
# This notebook demonstrates how to load a trained model and run predictions on genomic intervals.
# It is based on the `examples/chip_ar_mdapca2b.py` training script.

# %%
import torch
import matplotlib.pyplot as plt

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
from cerberus.models.gopher import GlobalProfileCNN
from cerberus.metrics import DefaultMetricCollection
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.predict import predict_intervals

# %% [markdown]
# ## 1. Setup Directories and Data
#
# We define the data and checkpoint directories.
# Make sure you have trained the model using `examples/chip_ar_mdapca2b.py` first.

# %%
project_root = get_project_root()
data_dir = project_root / "tests/data"
checkpoint_dir = project_root / "tests/data/models/chip_ar_mdapca2b/peaks-single"

print(f"Data Directory: {data_dir}")
print(f"Checkpoint Directory: {checkpoint_dir}")

if not checkpoint_dir.exists():
    raise FileNotFoundError(
        f"Checkpoint directory {checkpoint_dir} does not exist.\n"
        "Please run 'python examples/chip_ar_mdapca2b.py' to train the model first."
    )

# %%
# Download/Check Data
print("Checking Data...")
genome_files = download_human_reference(data_dir / "genome", name="hg38")
dataset_files = download_dataset(data_dir / "dataset", name="mdapca2b_ar")

# %% [markdown]
# ## 2. Configuration
#
# We setup the configuration. 
# Key differences from training:
# - `DataConfig`: `max_jitter=0` and `reverse_complement=False` for deterministic prediction.
# - `SamplerConfig`: `padded_size=input_len` to get exact input windows.

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

# Data Config
input_len = 2048
output_len = 1024

data_config: DataConfig = {
    "inputs": {}, 
    "targets": {"signal": dataset_files["bigwig"]},
    "input_len": input_len,
    "output_len": output_len, 
    "max_jitter": 0,        # Set to 0 for prediction to match input_len padded_size
    "output_bin_size": 4,
    "encoding": "ACGT",
    "log_transform": True, 
    "reverse_complement": True, # Matches training config (ignored if is_train=False)
    "use_sequence": True,
}

# Sampler Config
# We use the peak intervals from the dataset
sampler_config: SamplerConfig = {
    "sampler_type": "interval",
    "padded_size": input_len, # Exact input length for prediction (no jitter padding needed)
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"]
    }
}

# Model/Train Config (Must match training for correct model instantiation)
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
    "name": "GlobalProfileCNN",
    "model_cls": GlobalProfileCNN,
    "loss_cls": ProfilePoissonNLLLoss,
    "loss_args": {"log_input": True, "full": False},
    "metrics_cls": DefaultMetricCollection,
    "metrics_args": {"num_channels": 1},
    "model_args": {
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
    }
}

# %% [markdown]
# ## 3. Initialize Dataset and Sampler
#
# We initialize the dataset and split it to get the test fold sampler.

# %%
print("Initializing Dataset...")
# We set is_train=False to ensure deterministic behavior (disables jitter/RC)
# even though the config has them enabled.
dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    is_train=False
)

print("Splitting folds (Test Fold = 0)...")
_, _, test_dataset = dataset.split_folds(test_fold=0, val_fold=1)
test_sampler = test_dataset.sampler

print(f"Number of test intervals: {len(test_sampler)}")

# %% [markdown]
# ## 4. Load Model
#
# We use `ModelEnsemble` to load the trained model from the checkpoint.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

print("Loading Model...")
model_ensemble = ModelEnsemble(
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
# Note: predict_intervals merges all provided intervals into one output track.
# Since peaks are disjoint and spread across the genome, predicting them all at once
# would result in a massive sparse array covering the whole range.
# For disjoint peaks, we predict one at a time (or group by cluster).
target_idx = 0
target_interval = test_sampler[target_idx]
intervals_to_predict = [target_interval]

print(f"Predicting on interval: {target_interval}")

predict_config: PredictConfig = {
    "stride": output_len,
    "intervals": [],
    "intervals_paths": [],
    "use_folds": ["test"], # Use the model corresponding to test fold
    "aggregation": "mean"
}

output = predict_intervals(
    intervals=intervals_to_predict,
    dataset=dataset, # Use main dataset (has extractors)
    model_ensemble=model_ensemble,
    predict_config=predict_config,
    device=str(device),
    batch_size=64
)
import dataclasses
outputs = dataclasses.asdict(output)
merged_interval = output.out_interval

print("Prediction Complete!")
print(f"Merged Interval: {merged_interval}")

# %% [markdown]
# ## 6. Analyze Results
#
# The model returns a tuple of outputs. For `GlobalProfileCNN`, it typically returns (logits, log_counts) or similar depending on the forward pass.
# Let's inspect the outputs.

# %%
# Extract Ground Truth for the merged interval
# Note: This ground truth will be raw signal over the huge merged interval.
if dataset.target_signal_extractor is None:
    raise ValueError("Target signal extractor is missing.")

gt_targets = dataset.target_signal_extractor.extract(merged_interval)
print(f"Ground Truth Shape: {gt_targets.shape}")

# Inspect Outputs
for key, track in outputs.items():
    print(f"\nOutput Track '{key}':")
    print(f"  Shape: {track.shape}")
    print(f"  Min: {track.min():.4f}, Max: {track.max():.4f}, Mean: {track.mean():.4f}")

# %% [markdown]
# ### Visualization
#
# Let's visualize the prediction for the first interval in our list.
# We need to find where it sits in the merged output.

# %%
# Visualizing the interval
print(f"Visualizing interval: {target_interval}")

# Calculate offset in merged interval
# Merged interval aligns with bins
output_bin_size = data_config["output_bin_size"]
rel_start_bp = target_interval.center(output_len).start - merged_interval.start
rel_start_bin = rel_start_bp // output_bin_size
n_bins = output_len // output_bin_size

print(f"Relative Start Bin: {rel_start_bin}")
print(f"Number of Bins: {n_bins}")

# Extract prediction for this interval
# Key 'logits' is usually the profile output for GlobalProfileCNN
output_key = "logits" if "logits" in outputs else list(outputs.keys())[0]
pred_profile = outputs[output_key][:, rel_start_bin : rel_start_bin + n_bins]

# Extract Ground Truth for this interval
# gt_targets is in bp resolution (1024 bp). We need to select the region and bin it to match prediction (256 bins).
gt_start_bp = rel_start_bin * output_bin_size
gt_end_bp = gt_start_bp + n_bins * output_bin_size
gt_slice_bp = gt_targets[:, gt_start_bp : gt_end_bp]

# Bin the ground truth
# (1, 1024) -> (1, 256, 4) -> mean -> (1, 256)
gt_slice_binned = gt_slice_bp.view(1, -1, output_bin_size).mean(dim=2)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# 1. Ground Truth
axes[0].plot(gt_slice_binned[0], label="Ground Truth (Signal, Binned)", color="blue")
axes[0].set_title("Ground Truth")
axes[0].legend()

# 2. Transformations
# Based on config:
# - Model trained with ProfilePoissonNLLLoss(log_input=True).
#   This means model output (logits) represents log(rate).
#   So Predicted Rate = exp(logits).
# - DataConfig has log_transform=True.
#   This means Targets passed to loss were log1p(counts).
#   So the model learns to predict rate ~ log1p(counts).
#   Predicted Rate ~= log1p(counts).
#
# To compare on Observed Signal scale (Counts):
# Observed = Counts
# Predicted = exp(Predicted Rate) - 1 = exp(exp(logits)) - 1

logits = torch.from_numpy(pred_profile)
pred_rate = torch.exp(logits) # Corresponds to log1p(counts)
pred_counts = torch.expm1(pred_rate) # Corresponds to counts

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# 1. Ground Truth (Observed Signal)
axes[0].plot(gt_slice_binned[0], label="Ground Truth (Counts)", color="blue")
axes[0].set_title("Observed Signal (Counts)")
axes[0].legend()

# 2. Predicted Counts (Transformed back to Count scale)
axes[1].plot(pred_counts[0], label="Predicted (Counts)", color="green")
axes[1].set_title("Predicted Signal (Counts)")
axes[1].legend()

# 3. Model Space Comparison (Log Scale)
# Compare log1p(GT) with Predicted Rate
gt_log = torch.log1p(gt_slice_binned[0])
axes[2].plot(gt_log, label="Ground Truth (Log1p)", color="blue", alpha=0.5)
axes[2].plot(pred_rate[0], label="Predicted Rate (Log1p)", color="orange", alpha=0.5)
axes[2].set_title("Model Space Comparison (Log Scale)")
axes[2].legend()

plt.xlabel("Bins")
plt.tight_layout()
plt.show()

# %%
