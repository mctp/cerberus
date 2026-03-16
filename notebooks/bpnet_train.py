# %% [markdown]
# # Training a BPNet Model with Cerberus
# 
# This notebook demonstrates how to train a BPNet (Base-Resolution Prediction Net) style model using the Cerberus framework.
# We will use the MDA-PCA-2b AR ChIP-seq dataset and the hg38 human reference genome.
#
# **Task**: Train a BPNet model taking 2114bp DNA input and predicting base-resolution BigWig profile (1000bp output).

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
# will download the human reference genome (hg38) and the MDA-PCA-2b AR dataset.

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
# **Key Requirements for BPNet:**
# - Input: 2114bp DNA (Standard BPNet input)
# - Output: 1000bp (Standard BPNet output) at base resolution
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
# BPNet works at base resolution (output_bin_size=1).
data_config: DataConfig = {
    "inputs": {}, # No additional input tracks, just DNA
    "targets": {"signal": dataset_files["bigwig"]},
    "input_len": 2114,
    "output_len": 1000, 
    "max_jitter": 128,  # Augmentation jitter
    "output_bin_size": 1, # Base resolution for BPNet
    "encoding": "ACGT", # Standard One-Hot
    "log_transform": False, # BPNet uses raw counts for multinomial loss
    "reverse_complement": True, # Augmentation
        "target_scale": 1.0,
    "count_pseudocount": 1.0,
    "use_sequence": True,
}

# Sampler Config
# We need padded_size >= input_len + 2 * max_jitter = 2114 + 256 = 2370.
sampler_config: SamplerConfig = {
    "sampler_type": "interval",
    "padded_size": 2370,
    "sampler_args": {
        "intervals_path": dataset_files["narrowPeak"]
    }
}

# Train Config
# Standard training config with AdamW optimizer and cosine scheduler.
train_config: TrainConfig = {
    "batch_size": 16, # BPNet can be memory intensive, adjust if needed
    "max_epochs": 2, # Short training for demonstration
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "patience": 5,
    "optimizer": "adamw",
    "filter_bias_and_bn": True,
    "reload_dataloaders_every_n_epochs": 0,
    "scheduler_type": "cosine",
    "scheduler_args": {
        "num_epochs": 2, # Must match max_epochs
        "warmup_epochs": 0,
        "min_lr": 1e-5
    },
    "adam_eps": 1e-7,
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
)

# Setup datamodule for interactive exploration only.
# _train() will call setup() again with the same parameters before fitting.
# Note: num_workers=0 for compatibility in notebook
datamodule.prepare_data()
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
print("Batch inputs shape:", batch["inputs"].shape)   # Expected: (B, 4, 2114)
print("Batch targets shape:", batch["targets"].shape) # Expected: (B, 1, 1000)

# %% [markdown]
# ## 4. Model Configuration
#
# We define a `ModelConfig` dict specifying the BPNet architecture, loss, and metrics
# using dotted class paths. Setting `alpha="adaptive"` instructs cerberus to compute
# the counts loss weight from the median training-set depth automatically
# (alpha = median_counts / 10), balancing the profile and counts loss terms.

# %%
model_config: ModelConfig = {
    "name": "BPNet_AR",
    "model_cls": "cerberus.models.bpnet.BPNet",
    "model_args": {
        "n_dilated_layers": 8,  # 8 layers (dilations 2..256) matches 2114->1000
        "output_channels": ["signal"],
    },
    "loss_cls": "cerberus.models.bpnet.BPNetLoss",
    "loss_args": {"alpha": "adaptive"},  # computed from training data at fit time
    "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
    "metrics_args": {},
}

# %% [markdown]
# ## 5. Training
#
# We run the training loop using `_train`. It will setup the datamodule, resolve the
# adaptive alpha, instantiate the module, and call `trainer.fit()`.

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
    enable_progress_bar=False,
    log_every_n_steps=5,
)

print("Training finished.")

# %% [markdown]
# ## Technical Notes: Running as a Script
#
# If you run this file as a standalone script (e.g. `python notebooks/bpnet_train.py`) on macOS or Windows,
# you may encounter a `RuntimeError` related to multiprocessing.
#
# **Reason:**
# PyTorch DataLoaders use `multiprocessing` to load data in parallel (`num_workers=4`).
# On macOS and Windows, the default start method is `spawn`, which requires child processes to import the main module.
# Without an `if __name__ == "__main__":` guard, this leads to recursive execution of the script logic.
#
# **Solution:**
# To run as a script, you must wrap the execution logic (specifically the `train()` call and `iter(dataloader)`)
# in an `if __name__ == "__main__":` block.
# Alternatively, set `num_workers=0` in `train_config` to avoid multiprocessing.
#

# %%
