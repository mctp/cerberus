# %% [markdown]
# # Cerberus Prediction Demo
#
# This notebook demonstrates how to use `predict_intervals` to run inference on genomic intervals.

# %%
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchmetrics import MetricCollection
import dataclasses

try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    SamplerConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from unittest.mock import patch
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ProfileCountOutput

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

# Dataset
# We don't provide a sampler_config because we're doing inference on specific intervals
dataset = CerberusDataset(genome_config, data_config, sampler_config=None)

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
        
        return ProfileCountOutput(logits=profile, log_counts=counts)

# %% [markdown]
# ## 3. Setup Model Manager

# %%
# Create a fake checkpoint
model = SimpleModel(OUTPUT_LEN, BIN_SIZE)
# ModelEnsemble expects a directory
ckpt_dir = Path("dummy_model_dir")
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / "dummy_model.ckpt"
torch.save({"state_dict": model.state_dict()}, ckpt_path)

model_config: ModelConfig = {
    "name": "simple",
    "model_cls": "__main__.SimpleModel",
    "loss_cls": "torch.nn.MSELoss",
    "loss_args": {},
    "metrics_cls": "torchmetrics.MetricCollection",
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
    "reload_dataloaders_every_n_epochs": 0,
    "scheduler_type": "default",
    "scheduler_args": {}
}

# Model Ensemble
# We mock the internal loader to return our dummy model instead of loading from disk
with patch("cerberus.model_ensemble._ModelManager") as MockLoader:
    loader_instance = MockLoader.return_value
    loader_instance.load_models_and_folds.return_value = ({"single": model}, [])
    
    ensemble = ModelEnsemble(
        ckpt_dir,
        model_config,
        data_config,
        genome_config,
        device=torch.device("cpu")
    )

# Cleanup checkpoint
if ckpt_path.exists():
    ckpt_path.unlink()
if ckpt_dir.exists():
    ckpt_dir.rmdir()

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

# Run prediction
# We use a small batch size to demonstrate batching
output = ensemble.predict_intervals(
    intervals,
    dataset,
    use_folds=["test"], # Or whatever fold the chroms fall into, ModelManager handles this
    aggregation="model",
    batch_size=2
)
outputs = dataclasses.asdict(output)
merged_interval = output.out_interval

print("Merged Interval:", merged_interval)

# %% [markdown]
# ## 5. Analyze Results

# %%
print("Number of output heads:", len(outputs))

# outputs is a dict with keys 'logits' and 'log_counts'
profile_track = outputs["logits"]
counts_track = outputs["log_counts"]

print("Profile Track Shape:", profile_track.shape)
print("Counts Track Shape:", counts_track.shape)

# Since our model returns constant ones, let's check values
print("Mean Profile Value:", profile_track.mean().item())
print("Mean Counts Value:", counts_track.mean().item())

# %%
