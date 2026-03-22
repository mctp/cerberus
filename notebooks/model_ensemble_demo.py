# %% [markdown]
# # Model Ensemble & Prediction Demo
# 
# This notebook demonstrates how to load a trained model using `ModelEnsemble` and run predictions.
# We use a pre-trained BPNet model (single-fold) located in `tests/data/models`.

# %%
import os
from pathlib import Path
import torch

from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.config import GenomeConfig, DataConfig, ModelConfig
from cerberus.genome import create_genome_config
from cerberus.samplers import IntervalSampler

# %% [markdown]
# ## 1. Setup Configuration
# 
# We need to recreate the configuration used during training to correctly instantiate the model architecture and data pipeline.

# %%
# Define paths (relative to project root)
os.chdir(Path(__file__).parent.parent)
DATA_DIR = Path("tests/data")
MODEL_DIR = Path("tests/data/models/chip_ar_mdapca2b_bpnet/multi-fold")
if not MODEL_DIR.exists():
    print(f"Skipping: checkpoint directory not found: {MODEL_DIR}")
    print("Run 'bash examples/chip_ar_mdapca2b_bpnet.sh' with a multi-fold setup to generate it.")
    import sys; sys.exit(0)

# 1. Genome Config
# We use the same genome config as training
genome_config: GenomeConfig = create_genome_config(
    name="hg38",
    fasta_path=DATA_DIR / "genome/hg38/hg38.fa",
    species="human",
    fold_type="chrom_partition",
    fold_args={"k": 5, "val_fold": 1, "test_fold": 0},
    exclude_intervals={
        "blacklist": DATA_DIR / "genome/hg38/blacklist.bed",
        "gaps": DATA_DIR / "genome/hg38/gaps.bed",
    }
)

# 2. Data Config
# Matches training: 2114bp input -> 1000bp output
data_config = DataConfig(
    inputs={},
    targets={"signal": DATA_DIR / "dataset/mdapca2b_ar/mdapca2b-ar.bigwig"},
    input_len=2114,
    output_len=1000,
    output_bin_size=1,
    encoding="ACGT",
    max_jitter=0,  # Ignored during prediction (is_train=False)
    log_transform=False,
    reverse_complement=False,  # Ignored during prediction
    target_scale=1.0,
    use_sequence=True,
)

# 3. Model Config
# Matches training architecture
model_config = ModelConfig(
    name="BPNet",
    model_cls="cerberus.models.bpnet.BPNet",
    loss_cls="cerberus.models.bpnet.BPNetLoss",
    loss_args={"alpha": 1.0},
    metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
    metrics_args={},
    model_args={
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "filters": 64,
        "n_dilated_layers": 8,
    },
)

# %% [markdown]
# ## 2. Load Model Ensemble
# 
# We initialize `ModelEnsemble` pointing to the directory containing the checkpoint.
# Since this is a single-fold model, it will be loaded under the key "single".

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ensemble = ModelEnsemble(
    checkpoint_path=MODEL_DIR,
    model_config=model_config,
    data_config=data_config,
    genome_config=genome_config,
    device=device
)

print(f"Loaded models: {list(ensemble.keys())}")

# %% [markdown]
# ## 3. Prepare Dataset
# 
# We create a `CerberusDataset` in inference mode (`is_train=False`).
# We also need a list of intervals to predict on. We'll load some from the test set peaks.

# %%
# Load some intervals from the narrowPeak file
peaks_path = DATA_DIR / "dataset/mdapca2b_ar/mdapca2b-ar.narrowPeak.gz"
sampler = IntervalSampler(
    file_path=peaks_path,
    chrom_sizes=genome_config.chrom_sizes,
    padded_size=data_config.input_len,
    exclude_intervals={}, # Don't filter for this demo
    folds=[] # No fold logic for simple interval loading
)

dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler=sampler,
    is_train=False
)

# Get first 5 intervals
test_intervals = []
for i, interval in enumerate(sampler):
    if i >= 5: break
    test_intervals.append(interval)

print(f"Predicting on {len(test_intervals)} intervals:")
# All intervals have been padded to input length centered around peak summit (narrowPeak input)
for iv in test_intervals:
    print(f"  {iv}")

# %% [markdown]

# ## 4. Single Interval Prediction

# %%
pred_iv0 = ensemble.predict_intervals(
    intervals=[test_intervals[0]],
    dataset=dataset,
    use_folds=["test"],
    aggregation="interval+model")

print(f"Predicted output for interval {test_intervals[0]}:")
print(f"pred_iv0.out_interval: {pred_iv0.out_interval}")
print(f"pred_iv0.logits.shape: {pred_iv0.logits.shape}")
# Slicing on the last dimension (length)
# print(f"Sample logits: {pred_iv0.logits[..., [100, 200, 300, 400, 500]]}")
print(f"pred_iv0.log_counts.mean().item(): {pred_iv0.log_counts.mean().item()}")

# %% [markdown]
# ## 5. Predict Over Large Output Region

# %%
iv_block = Interval(chrom="chr8", start=24_000_000, end=24_010_000)
pred_block = ensemble.predict_output_intervals(
    intervals=[iv_block],
    dataset=dataset,
    stride=100,
    use_folds=["test"],
    aggregation="interval+model")

# %% [markdown]
# ## 6. Visualize Predictions
#
# We plot the predicted profile over the large region.

# %%
import matplotlib.pyplot as plt

if pred_block:
    # Assuming pred_block is a list of results, and we want the first one
    prediction = pred_block[0]
    
    # Check if logits exists and is a tensor
    if hasattr(prediction, "logits") and isinstance(prediction.logits, torch.Tensor):
        logits = prediction.logits.squeeze().cpu().numpy()
        
        # If multi-channel, maybe plot first channel or sum
        if logits.ndim > 1:
             logits = logits[0] # Take first channel
             
        plt.figure(figsize=(15, 4))
        plt.plot(logits)
        plt.title(f"Predicted Logits for {prediction.out_interval}")
        plt.xlabel("Position (bins)")
        plt.ylabel("Logits")
        
        # Save figure if running as script
        plots_dir = Path("notebooks/plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plots_dir / "prediction_plot.png")
        print(f"Plot saved to {plots_dir / 'prediction_plot.png'}")
    else:
        print("No logits found in prediction.")
else:
    print("No predictions generated.")

# %%
