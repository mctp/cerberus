# %% [markdown]
# # Training a Simple CNN with Cerberus
# 
# This notebook demonstrates how to train a simple CNN model (Baseline) using the Cerberus framework.
# We will use the MDA-PCA-2b AR ChIP-seq dataset and the hg38 human reference genome.
#
# **Task**: Train a model taking 2048bp DNA input and predicting a BigWig profile (256 bins at 4bp resolution).

# %%
try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

# Cerberus imports
from pprint import pprint
from cerberus.download import download_dataset, download_human_reference
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.datamodule import CerberusDataModule
from cerberus.train import _train

# %% [markdown]
# ## 1. Setup Directories and Download Data
# 
# We'll define a working directory for our data and download the necessary files. We
# will download the human reference genome (hg38) and the MDA-PCA-2b AR dataset (also 
# used in Cerberus tests).

# %%
DATA_DIR = get_project_root() / "tests/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download Human Reference (hg38)
# This includes FASTA, Blacklist, Gaps, Mappability, and ENCODE cCREs
print("Downloading/Checking Human Reference...")
genome_files = download_human_reference(DATA_DIR / "genome", name="hg38")

# Download Dataset (MDA-PCA-2b AR)
# This includes BigWig (signal) and narrowPeak (peaks)
print("Downloading/Checking Dataset...")
dataset_files = download_dataset(DATA_DIR / "dataset", name="mdapca2b_ar")

print("Genome Files:", genome_files)
print("Dataset Files:", dataset_files)

# %% [markdown]
# ## 2. Configuration
# 
# We define the configurations for the Genome, Data, Sampler, and Training.
# 
# **Key Requirements:**
# - Input: 2048bp DNA
# - Output: 256 bins @ 4bp resolution (covering 1024bp)
# - Input tracks: None (just DNA)
# - Target tracks: BigWig signal

# %%
# Genome Config
# We use create_genome_config to automatically parse the FASTA index (.fai)
# and set up the genome configuration.
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
# We specify no input tracks (just DNA) and the BigWig as the target.
data_config: DataConfig = {
    "inputs": {}, # No additional input tracks, just DNA
    "targets": {"signal": dataset_files["bigwig"]},
    "input_len": 2048,
    "output_len": 1024, # 1024bp field of view -> 256 bins @ 4bp
    "max_jitter": 128,  # Augmentation jitter
    "output_bin_size": 4, # Binning resolution
    "encoding": "ACGT", # Standard One-Hot
    "log_transform": True, # Log(x+1) transform targets
    "reverse_complement": True, # Augmentation
        "target_scale": 1.0,
    "count_pseudocount": 1.0,
    "use_sequence": True,
}

# Sampler Config
# We need padded_size >= input_len + 2 * max_jitter = 2048 + 256 = 2304.
sampler_config: SamplerConfig = {
    "sampler_type": "interval",
    "padded_size": 2304,
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"]
    }
}

# Train Config
# Standard training config with AdamW optimizer and cosine scheduler.
train_config: TrainConfig = {
    "batch_size": 16,
    "max_epochs": 3, # Short training for demonstration
    "learning_rate": 0.1,
    "weight_decay": 0.01,
    "patience": 5,
    "optimizer": "adamw",
    "filter_bias_and_bn": True,
    "reload_dataloaders_every_n_epochs": 0,
    "scheduler_type": "cosine",
    "scheduler_args": {
        "num_epochs": 3, # Must match max_epochs
        "warmup_epochs": 0,
        "min_lr": 1e-5
    },
    "adam_eps": 1e-8,
    "gradient_clip_val": None,
}

print("Genome Config:")
pprint(genome_config)
print("Data Config:")
pprint(data_config)
print("Sampler Config:")
pprint(sampler_config)
print("Train Config:")
pprint(train_config)

# %% [markdown]
# ## 3. Initialize DataModule
# 
# We create the `CerberusDataModule` which handles dataset creation and DataLoaders.

# %%
datamodule = CerberusDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    drop_last=True,
)

# Setup datamodule for interactive exploration only.
# _train() will call setup() again with the same parameters before fitting.
# Note: num_workers=0 for compatibility in notebook
datamodule.setup(
    batch_size=train_config["batch_size"],
    num_workers=0,
    in_memory=False
)
if datamodule.train_dataset:
    print("Train set size:", len(datamodule.train_dataset))
if datamodule.val_dataset:
    print("Val set size:", len(datamodule.val_dataset))

# Verify a batch
batch = next(iter(datamodule.train_dataloader()))
print("Batch inputs shape:", batch["inputs"].shape)   # Expected: (B, 4, 2048)
print("Batch targets shape:", batch["targets"].shape) # Expected: (B, 1, 256)

# %% [markdown]
# ## 4. Model Configuration
#
# We define a `ModelConfig` dict specifying the `GlobalProfileCNN` architecture, loss,
# and metrics using dotted class paths. The model dimensions (`input_len`, `output_len`,
# `output_bin_size`) are taken automatically from `data_config` at instantiation time.

# %%
model_config: ModelConfig = {
    "name": "GlobalProfileCNN",
    "model_cls": "cerberus.models.gopher.GlobalProfileCNN",
    "model_args": {},  # input_len/output_len/output_bin_size come from data_config
    "loss_cls": "cerberus.loss.ProfilePoissonNLLLoss",
    "loss_args": {"log_input": True, "full": False, "log1p_targets": False},
    "metrics_cls": "cerberus.metrics.DefaultMetricCollection",
    "metrics_args": {"log1p_targets": False},
}

# %% [markdown]
# ## 5. Training
#
# We run the training loop using `_train`. It will setup the datamodule, instantiate
# the module from `model_config`, and call `trainer.fit()`.

# %%
# Train
# We use limit_train_batches / limit_val_batches to keep this demo short.
trainer = _train(
    model_config=model_config,
    data_config=data_config,
    datamodule=datamodule,
    train_config=train_config,
    num_workers=0,  # Set to 0 for compatibility in notebook
    in_memory=False,
    accelerator="auto",
    devices=1,
    limit_train_batches=10,  # For demo purposes
    limit_val_batches=5,
    enable_checkpointing=True,
    logger=True,  # Enable logging
    enable_progress_bar=True,
    log_every_n_steps=5,
)

print("Training finished.")

# %% [markdown]
# ## Technical Notes: Running as a Script
#
# If you run this file as a standalone script (e.g. `python notebooks/simple_cnn_train.py`) on macOS or Windows,
# you may encounter a `RuntimeError` related to multiprocessing.
#
# **Reason:**
# PyTorch DataLoaders use `multiprocessing` to load data in parallel (`num_workers=4`).
# On macOS and Windows, the default start method is `spawn`, which requires child processes to import the main module.
# Without an `if __name__ == "__main__":` guard, this leads to recursive execux/tion of the script logic.
#
# **Solution:**
# To run as a script, you must wrap the execution logic (specifically the `train()` call and `iter(dataloader)`)
# in an `if __name__ == "__main__":` block.
# Alternatively, set `num_workers=0` in `train_config` to avoid multiprocessing.
#

# %% [markdown]
# ## 6. Analyze Training Results
# 
# We inspect the training and validation loss over epochs.

# %%
import pandas as pd
import matplotlib.pyplot as plt

# The trainer object from the previous cell contains the logger
log_dir = trainer.logger.log_dir
metrics_path = f"{log_dir}/metrics.csv"
print(f"Metrics saved to: {metrics_path}")

try:
    metrics = pd.read_csv(metrics_path)
    
    # Group by epoch to get one row per epoch (taking the last logged value for that epoch)
    epoch_metrics = metrics.groupby("epoch").last()
    
    # Display Loss columns
    loss_cols = [c for c in epoch_metrics.columns if "loss" in c]
    print("\nLoss per Epoch:")
    print(epoch_metrics[loss_cols])
    
    # Plot
    plt.figure(figsize=(10, 6))
    for col in loss_cols:
        # separate train and val
        if "train" in col:
            plt.plot(epoch_metrics.index, epoch_metrics[col], label=col, marker='o')
        elif "val" in col:
            plt.plot(epoch_metrics.index, epoch_metrics[col], label=col, marker='x', linestyle='--')
            
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Could not load or plot metrics: {e}")

# %% [markdown]
# ## 7. Epoch Analysis
# 
# Report the number of samples in one epoch.

# %%
train_samples = len(datamodule.train_dataset)
val_samples = len(datamodule.val_dataset)
batch_size = train_config["batch_size"]

print(f"Total Train Samples (Full Epoch): {train_samples}")
print(f"Total Validation Samples (Full Epoch): {val_samples}")
print(f"Batch Size: {batch_size}")

# Effective samples if limit_train_batches is set
# trainer.limit_train_batches can be int (num batches) or float (fraction) or 1.0 (default)
limit = trainer.limit_train_batches
print(f"Limit Train Batches: {limit}")

if isinstance(limit, int) and limit > 0:
    effective_samples = limit * batch_size
    print(f"Effective Train Samples per Epoch (due to limit): {effective_samples}")
elif isinstance(limit, float) and limit < 1.0:
    effective_samples = int(train_samples * limit)
    print(f"Effective Train Samples per Epoch (due to limit): {effective_samples}")
