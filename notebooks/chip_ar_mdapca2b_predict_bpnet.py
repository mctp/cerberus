# %% [markdown]
# # Chip-AR MDAPCA2b Prediction Notebook (BPNet)
#
# This notebook demonstrates how to load a pretrained BPNet model and run predictions on genomic intervals.
# The pretrained model is shipped in `pretrained/chip_ar_mdapca2b_bpnet/`.

# %%
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from paths import get_project_root
except ImportError:
    sys.path.append("notebooks")
    from paths import get_project_root

from cerberus.dataset import CerberusDataset
from cerberus.download import download_dataset, download_human_reference
from cerberus.model_ensemble import ModelEnsemble

# %% [markdown]
# ## 1. Setup
#
# We load the pretrained BPNet model from `pretrained/chip_ar_mdapca2b_bpnet/`. All configuration
# (genome, data, model) is recovered automatically from `hparams.yaml`.
# The dataset (MDA-PCA-2b AR, hg38) is downloaded if not already present.
# %%
from cerberus.utils import resolve_device

project_root = get_project_root()
checkpoint_dir = project_root / "pretrained/chip_ar_mdapca2b_bpnet"

if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Pretrained model not found: {checkpoint_dir}")

device = resolve_device()
print(f"Using device: {device}")

print("Loading Model...")
model_ensemble = ModelEnsemble(checkpoint_path=checkpoint_dir, device=device)
config = model_ensemble.cerberus_config

print(f"Model: {config.model_config_.name}")
print(f"Input length: {config.data_config.input_len}")

# %%
# Download/Check Data
data_dir = project_root / "tests/data"
print("Checking Data...")
genome_files = download_human_reference(data_dir / "genome", name="hg38")
dataset_files = download_dataset(data_dir / "dataset", name="mdapca2b_ar")

# %% [markdown]
# ## 2. Initialize Dataset and Sampler
#
# We initialize the dataset from the ensemble config and split it to get the test fold sampler.

# %%
print("Initializing Dataset...")
dataset = CerberusDataset(
    genome_config=config.genome_config,
    data_config=config.data_config,
    sampler_config=config.sampler_config,
    is_train=False,
)

fold_args = config.genome_config.fold_args
print(f"Splitting folds (Test Fold = {fold_args['test_fold']})...")
_, _, test_dataset = dataset.split_folds(
    test_fold=fold_args["test_fold"], val_fold=fold_args["val_fold"]
)
test_sampler = test_dataset.sampler

print(f"Number of test intervals: {len(test_sampler)}")

# %% [markdown]
# ## 3. Run Prediction
#
# We predict on a subset of the test intervals.

# %%
# Select a single interval for demonstration
target_idx = 0
target_interval = test_sampler[target_idx]

# Ensure interval matches input_len (center crop if needed)
# predict_intervals expects intervals of exact input_len
input_len = config.data_config.input_len
if len(target_interval) > input_len:
    target_interval = target_interval.center(input_len)

intervals_to_predict = [target_interval]

print(f"Predicting on interval: {target_interval}")

output = model_ensemble.predict_intervals(
    intervals=intervals_to_predict,
    dataset=dataset,
    use_folds=["test"],
    aggregation="model",
    batch_size=64,
)
# Convert ModelOutput to dict for analysis
import dataclasses

outputs = dataclasses.asdict(output)
merged_interval = output.out_interval

print("Prediction Complete!")
print(f"Merged Interval: {merged_interval}")

# %% [markdown]
# ## 4. Analyze Results
#
# BPNet returns `(profile_logits, log_counts)`.
# `predict_intervals` aggregates these into tracks.
# - Track 0: `profile_logits` (shape 1x1000)
# - Track 1: `log_counts` (scalar broadcasted to 1x1000)

# %%
# Extract Ground Truth
if dataset.target_signal_extractor is None:
    raise ValueError("Target signal extractor is missing.")

if merged_interval is None:
    raise ValueError("Merged interval is None")

gt_targets = dataset.target_signal_extractor.extract(merged_interval)
print(f"Ground Truth Shape: {gt_targets.shape}")

# Inspect Outputs
for key, track in outputs.items():
    if key == "out_interval":
        continue
    print(f"\nOutput Track '{key}':")
    print(f"  Shape: {track.shape}")
    print(f"  Min: {track.min():.4f}, Max: {track.max():.4f}, Mean: {track.mean():.4f}")

# %% [markdown]
# ### Visualization
#
# We compare the predicted profile (converted to counts) with the ground truth.

# %%
# Visualizing the interval
print(f"Visualizing interval: {target_interval}")

output_len = config.data_config.output_len
output_bin_size = config.data_config.output_bin_size
rel_start_bp = target_interval.center(output_len).start - merged_interval.start
rel_start_bin = rel_start_bp // output_bin_size
n_bins = output_len // output_bin_size

print(f"Relative Start Bin: {rel_start_bin}")
print(f"Number of Bins: {n_bins}")

# Extract prediction tracks
# Track 0: Profile Logits
logits = outputs["logits"][:, rel_start_bin : rel_start_bin + n_bins]
logits_tensor = logits

# Track 1: Log Total Counts (scalar per channel)
log_counts = outputs["log_counts"]
log_counts_val = log_counts[0].detach().clone()  # Take first channel
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
gt_slice_bp = gt_targets[:, gt_start_bp:gt_end_bp]

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
plots_dir = project_root / "notebooks/plots"
plots_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(plots_dir / "chip_ar_mdapca2b_predict_bpnet.png")

# %%
