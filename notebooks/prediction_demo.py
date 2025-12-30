# %% [markdown]
# # Cerberus Prediction Demo
#
# This notebook demonstrates how to use `predict_intervals` to run inference on genomic intervals.

# %%
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchmetrics import MetricCollection

try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    PredictConfig,
    SamplerConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.model_manager import ModelManager
from cerberus.predict import predict_intervals

# %% [markdown]
# ## 1. Setup Configuration
#
# We point to the test data available in the repository.

# %%
# Setup paths
project_root = get_project_root()
TEST_DATA = project_root / "tests/data"
FASTA_PATH = TEST_DATA / "genome/hg38/hg38.fa"

# Verify paths
if not FASTA_PATH.exists():
    print(f"Warning: {FASTA_PATH} not found. Please run this from the notebooks directory or ensure paths are correct.")
    
# %%
# Genome Config
genome_config: GenomeConfig = {
    "name": "hg38_test",
    "fasta_path": FASTA_PATH,
    "allowed_chroms": ["chr1", "chr2"], # Assuming test fasta has these
    "exclude_intervals": {},
    "chrom_sizes": {"chr1": 10000, "chr2": 10000}, # Mock sizes
    "fold_type": "chrom_partition",
    "fold_args": {"k": 2}
}

# Data Config
# We define input length and output length
INPUT_LEN = 1000
OUTPUT_LEN = 1000
BIN_SIZE = 1

data_config: DataConfig = {
    "inputs": {},
    "targets": {},
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "output_bin_size": BIN_SIZE,
    "encoding": "ACGT",
    "max_jitter": 0,
    "log_transform": False,
    "reverse_complement": False,
    "use_sequence": True,
}

# Sampler Config (needed for Dataset initialization, though predict uses intervals directly)
sampler_config: SamplerConfig = {
    "sampler_type": "dummy",
    "padded_size": INPUT_LEN,
    "sampler_args": {}
}

# Dataset
dataset = CerberusDataset(genome_config, data_config, sampler_config)

# %% [markdown]
# ## 2. Define Model
#
# We'll use a simple dummy model since we don't have a trained checkpoint.

# %%
class SimpleModel(nn.Module):
    def __init__(self, output_len, bin_size):
        super().__init__()
        self.output_dim = output_len // bin_size
        self.conv = nn.Conv1d(4, 1, kernel_size=1)
        
    def forward(self, x):
        # x: (B, 4, L)
        # We just return a dummy profile and a scalar count
        # Profile: (B, 1, output_dim)
        # Counts: (B, 1)
        
        batch_size = x.shape[0]
        
        # Dummy output: constant profile
        profile = torch.ones((batch_size, 1, self.output_dim), device=x.device)
        
        # Dummy counts
        counts = torch.ones((batch_size, 1), device=x.device) * 10.0
        
        return (profile, counts)

# %% [markdown]
# ## 3. Setup Model Manager

# %%
# Create a fake checkpoint
model = SimpleModel(OUTPUT_LEN, BIN_SIZE)
ckpt_path = Path("dummy_model.ckpt")
torch.save({"state_dict": model.state_dict()}, ckpt_path)

model_config: ModelConfig = {
    "name": "simple",
    "model_cls": SimpleModel,
    "loss_cls": nn.MSELoss,
    "loss_args": {},
    "metrics_cls": MetricCollection,
    "metrics_args": {},
    "model_args": {"output_len": OUTPUT_LEN, "bin_size": BIN_SIZE}
}

train_config: TrainConfig = {
    "batch_size": 32,
    "max_epochs": 1,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "patience": 1,
    "optimizer": "adam",
    "filter_bias_and_bn": False,
    "scheduler_type": "default",
    "scheduler_args": {}
}

# Model Manager
model_manager = ModelManager(
    ckpt_path,
    model_config,
    data_config,
    train_config,
    genome_config,
    device=torch.device("cpu")
)

# Cleanup checkpoint
if ckpt_path.exists():
    ckpt_path.unlink()

# %% [markdown]
# ## 4. Run Prediction
#
# We define some intervals and run `predict_intervals`.

# %%
# Define intervals
# Intervals must match input_len
intervals = [
    Interval("chr1", 0, INPUT_LEN),
    Interval("chr1", 500, 500 + INPUT_LEN),
    Interval("chr1", 1000, 1000 + INPUT_LEN)
]

# Predict Config
predict_config: PredictConfig = {
    "stride": 50,
    "intervals": [],
    "intervals_paths": [],
    "use_folds": ["test"], # Or whatever fold the chroms fall into, ModelManager handles this
    "aggregation": "mean"
}

# Mock ModelManager.get_models to return our model instance directly if loading fails/folds issue
# But for demo, let's assume get_models works if we mock the split or just allow it.
# Simple hack: override get_models
model_manager.get_models = lambda *args, **kwargs: [model]

# Run prediction
# We use a small batch size to demonstrate batching
outputs, merged_interval = predict_intervals(
    intervals,
    dataset,
    model_manager,
    predict_config,
    device="cpu",
    batch_size=2
)

print("Merged Interval:", merged_interval)

# %% [markdown]
# ## 5. Analyze Results

# %%
print("Number of output heads:", len(outputs))

profile_track = outputs[0]
counts_track = outputs[1]

print("Profile Track Shape:", profile_track.shape)
print("Counts Track Shape:", counts_track.shape)

# Since our model returns constant ones, let's check values
print("Mean Profile Value:", np.mean(profile_track))
print("Mean Counts Value:", np.mean(counts_track))

# %%
