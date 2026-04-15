# Usage

This guide provides a basic workflow for setting up and training with Cerberus.

## 1. Setup Configuration

Define your genome, data, and sampling strategy. This example assumes you have run `cerberus.download.download_human_reference("tests/data")` and `cerberus.download.download_dataset("tests/data", "mdapca2b_ar")`.

```python
from pathlib import Path
from cerberus.genome import create_genome_config

# Paths (based on download_human_reference and download_dataset defaults)
data_dir = Path("tests/data")
fasta_path = data_dir / "hg38/hg38.fa"
peaks_path = data_dir / "mdapca2b_ar/mdapca2b-ar.narrowPeak.gz"
signal_path = data_dir / "mdapca2b_ar/mdapca2b-ar.bigwig"
blacklist_path = data_dir / "hg38/blacklist.bed"

# 1. Genome Configuration
genome_config = create_genome_config(
    name="hg38",
    fasta_path=fasta_path,
    species="human",
    # Exclude blacklist regions
    exclude_intervals={"blacklist": blacklist_path},
    fold_type="chrom_partition",
    fold_args={"k": 5, "test_fold": 0, "val_fold": 1}
)

# 2. Data Configuration
from cerberus import DataConfig, SamplerConfig, TrainConfig, ModelConfig

data_config = DataConfig(
    inputs={},  # Only sequence input. Can add {"Track": "path.bw" or "path.bed"}
    targets={"AR": signal_path},  # BigWig signal
    input_len=2114,
    output_len=1000,
    output_bin_size=1,
    encoding="ACGT",
    max_jitter=128,
    log_transform=True,
    reverse_complement=True,
    use_sequence=True,
    target_scale=1.0,  # Multiplicative scale applied to targets before log transform
)

# 3. Sampler Configuration (Peaks + Negatives)
sampler_config = SamplerConfig(
    sampler_type="peak",
    padded_size=2114,
    sampler_args={
        "intervals_path": peaks_path,
        "background_ratio": 1.0,  # 1:1 ratio of peaks to complexity-matched background
    },
)
```

## 2. Instantiate DataModule

Create the `CerberusDataModule`. This handles multiprocessing and data splitting.

```python
from cerberus import CerberusDataModule

data_module = CerberusDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    pin_memory=True,            # Auto-disabled on MPS
    persistent_workers=True,    # Faster worker initialization
    multiprocessing_context=None, # e.g. 'spawn' for MPS stability
    seed=42                     # Optional seed for deterministic sampler initialization
)

# Setup datasets (split into train/val/test)
# You can set the batch size and num_workers here
data_module.setup(batch_size=256, num_workers=8)
```

## 3. Train with PyTorch Lightning

You can use the `cerberus.train` module to instantiate the model and start training easily.

```python
import torch.nn as nn
from cerberus import train_single, train_multi
from cerberus.loss import MSEMultinomialLoss
from torchmetrics import MetricCollection, PearsonCorrCoef, MeanSquaredError

# 4. Train Configuration
train_config = TrainConfig(
    batch_size=256,
    max_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=10,
    optimizer="adamw",
    scheduler_type="cosine",
    scheduler_args={"warmup_epochs": 5},
    filter_bias_and_bn=True,
    reload_dataloaders_every_n_epochs=0,
    adam_eps=1e-8,           # Use 1e-7 for BPNet-style models (matches TF/Keras default)
    gradient_clip_val=None,  # Set to e.g. 1.0 to clip gradients; None = disabled
)

# 5. Model Configuration
# Uses standard models from cerberus.models or your own importable class path.
# "model_cls", "loss_cls", and "metrics_cls" must be fully-qualified class strings.
# input_len, output_len, output_bin_size are automatically passed from DataConfig.

model_config = ModelConfig(
    name="my_bpnet",
    model_cls="cerberus.models.bpnet.BPNet",
    loss_cls="cerberus.models.bpnet.BPNetLoss",
    # Set alpha="adaptive" to compute the counts loss weight from the training set
    # automatically (alpha = median_total_counts / 10). This balances the profile
    # and counts loss terms at the correct scale for the dataset depth.
    loss_args={"alpha": "adaptive"},
    metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
    metrics_args={},
    model_args={
        "n_dilated_layers": 8,
        "output_channels": ["AR"],
    },
    count_pseudocount=1.0,  # Additive offset before log-transforming count targets (scaled units)
)

# Option A: Train a Single Model (Single Split)
# Uses the high-level API to handle instantiation and output structure (creates fold_0)
# (test_fold and val_fold are read from fold_args, or can be overridden here)
trainer = train_single(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,
    train_config=train_config,
    num_workers=8,
    in_memory=False,
    precision="16-mixed",      # Enable Mixed Precision
    matmul_precision="high",   # TensorFloat-32 on Ampere+
    root_dir="logs/single_run",
    run_test=True,             # Evaluate on test fold after training (uses best ckpt)
    accelerator="gpu",
    devices=1
)

# Option B: Cross-Validation Training (Multi-Fold)
# Trains k models (where k is defined in genome_config), each with a different test fold.
trainers = train_multi(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,
    train_config=train_config,
    num_workers=8,
    precision="16-mixed",
    matmul_precision="high",
    root_dir="logs/cross_val", # Models saved in logs/cross_val/fold_0, fold_1, etc.
    run_test=True,             # Evaluate on test fold after each fold's training
    accelerator="gpu",
    devices=1
)
```

### Output Files

After training, each fold directory (`fold_0/`, `fold_1/`, …) contains:

| File | Description |
|------|-------------|
| `model.pt` | Best checkpoint weights as a plain `state_dict` — load without PyTorch Lightning |
| `last.ckpt` | Last epoch checkpoint (useful for inspecting final weights) |
| `checkpoint-epoch=XX-val_loss=Y.ckpt` | Best checkpoint (lowest `val_loss`) |
| `plots/val_count_scatter_epoch_NNN.png` | Per-epoch scatter of predicted vs. true log-counts |
| `lightning_logs/version_0/hparams.yaml` | All configs (model, data, genome, sampler, train) as YAML |
| `lightning_logs/version_0/metrics.csv` | Per-step and per-epoch scalar metrics |

To load `model.pt` for inference without Lightning:
```python
import torch
from cerberus.models.bpnet import BPNet

model = BPNet(...)
model.load_state_dict(torch.load("logs/single_run/fold_0/model.pt", map_location="cpu"))
model.eval()
```

### Logging

Cerberus automatically logs all configurations (`genome_config`, `data_config`, `sampler_config`, `model_config`, `train_config`) to `lightning_logs/version_{#}/hparams.yaml` via Lightning's `save_hyperparameters`. This is the authoritative config file used by `ModelEnsemble` for inference.

## Manual Usage (PyTorch Dataset)

If you aren't using Lightning, you can use `CerberusDataset` directly.

```python
from cerberus import CerberusDataset
from torch.utils.data import DataLoader

dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    seed=42  # Optional seed for deterministic interval selection
)

# Access a single item
item = dataset[0]
print(item['inputs'].shape)   # (4, 2114) -> Sequence (one-hot)
print(item['targets'].shape)  # (1, 1000) -> Signal
print(item['intervals'])      # "chr1:1000-3114(+)" -> genomic interval string
print(item['interval_source']) # e.g. "IntervalSampler" (peak) or "ComplexityMatchedSampler" (background)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Reproducibility and Seeding

Ensuring reproducibility in training runs involves two levels of randomness control:

1.  **Global Randomness (Model & Shuffling)**:
    Use `pl.seed_everything(seed)` at the start of your script. This sets the seed for:
    -   Model weight initialization (`torch`).
    -   Data shuffling order in DataLoaders (`random`, `numpy`).
    -   Transformations in workers (`CerberusDataModule` handles worker seeding).

    ```python
    import pytorch_lightning as pl
    pl.seed_everything(42)
    ```

2.  **Dataset Composition (Samplers)**:
    Some samplers, specifically `RandomSampler` and `ComplexityMatchedSampler`, perform random selection logic (e.g., choosing 10,000 random intervals from the whole genome). To ensure that the **same set of intervals** is selected across different runs, you must provide a seed to the `CerberusDataModule` or `CerberusDataset` at run-time.

    ```python
    # Ensure the sampler picks the same random intervals every time
    data_module = CerberusDataModule(..., seed=42)
    ```

    This seed is passed down to `create_sampler`, which initializes the samplers deterministically. For complex samplers like `MultiSampler`, the seed is propagated to all sub-samplers (e.g., seed, seed+1, seed+2) to ensure the entire sampling tree is deterministic.

## Examples

The `examples/` directory contains ready-to-run shell scripts covering all model × dataset combinations. Each script defines dataset-specific variables at the top and forwards any extra arguments to the underlying tool.

| Script | Model | Dataset |
|---|---|---|
| `examples/chip_ar_mdapca2b_bpnet.sh` | BPNet | MDA-PCA-2b AR (auto-downloaded) |
| `examples/chip_ar_mdapca2b_pomeranian.sh` | Pomeranian | MDA-PCA-2b AR (auto-downloaded) |
| `examples/chip_ar_mdapca2b_gopher.sh` | Gopher | MDA-PCA-2b AR (auto-downloaded) |
| `examples/chip_prox1_tc32_bpnet.sh` | BPNet | TC32 PROX1 (local paths) |
| `examples/chip_prox1_tc32_pomeranian.sh` | Pomeranian | TC32 PROX1 (local paths) |
| `examples/chip_prox1_tc32_gopher.sh` | Gopher | TC32 PROX1 (local paths) |
| `examples/scatac_kidney_pseudobulk.sh` | — | Kidney scATAC-seq pseudobulk BigWigs + peaks |
| `examples/scatac_kidney_dalmatian.sh` | Dalmatian | Kidney scATAC-seq pseudobulk (bulk, all cell types) |

```bash
# Run single-fold training (default)
bash examples/chip_ar_mdapca2b_bpnet.sh

# Run multi-fold cross-validation
bash examples/chip_ar_mdapca2b_pomeranian.sh --multi

# Use PomeranianK5 variant
bash examples/chip_ar_mdapca2b_pomeranian.sh --k5
```

The `notebooks/` directory contains lower-level walkthroughs:

*   `notebooks/cerberus_basics.py`: A step-by-step walkthrough of the library components (Configuration, Samplers, Datasets, Transforms).
*   `notebooks/baseline_cnn_train.py`: A complete training example using the `GlobalProfileCNN` model to predict BigWig tracks from DNA sequence.

## Generic Training Tools

For quick training on custom data, use the model-specific scripts in the `tools/` directory. All scripts accept any BigWig signal and BED/narrowPeak file and share the same CLI structure.

*   `tools/train_bpnet.py`: Train a BPNet model.
    ```bash
    # Standard BPNet (2114bp -> 1000bp, canonical chrombpnet-pytorch settings)
    python tools/train_bpnet.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_bpnet

    # BPNet1024 variant (2112bp -> 1024bp, comparable I/O to Pomeranian)
    python tools/train_bpnet.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_bpnet --1024

    # Multi-fold cross-validation
    python tools/train_bpnet.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_bpnet --multi
    ```

*   `tools/train_pomeranian.py`: Train a Pomeranian model (2112bp → 1024bp).
    ```bash
    # Default Pomeranian (Kernel=9)
    python tools/train_pomeranian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_pomeranian

    # PomeranianK5 variant (Kernel=5)
    python tools/train_pomeranian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_pomeranian --k5

    # Customized training
    python tools/train_pomeranian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_pomeranian \
        --learning-rate 0.001 --patience 15 --background-ratio 2.0
    ```

*   `tools/train_gopher.py`: Train a Gopher (GlobalProfileCNN) model (2048bp → 1024bp at 4bp resolution).
    ```bash
    # Standard Gopher
    python tools/train_gopher.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_gopher

    # Custom bottleneck size
    python tools/train_gopher.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_gopher \
        --bottleneck-channels 16

    # Multi-fold cross-validation
    python tools/train_gopher.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_gopher --multi
    ```

*   `tools/train_dalmatian.py`: Train a Dalmatian model (2112bp → 1024bp, bias-factorized).
    ```bash
    # Default Dalmatian (MSE base loss)
    python tools/train_dalmatian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_dalmatian

    # Poisson base loss
    python tools/train_dalmatian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_dalmatian --base-loss poisson

    # Customized training
    python tools/train_dalmatian.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_dalmatian \
        --signal-filters 128 --bias-weight 2.0
    ```

*   `tools/train_asap.py`: Train an ASAP (ConvNeXtDCNN) model (2048bp → 512 bins at 4bp resolution).
    ```bash
    python tools/train_asap.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_asap

    python tools/train_asap.py --bigwig signal.bw --peaks regions.bed --output-dir models/my_asap --multi
    ```

All tools support `--multi` (cross-validation), `--seed` (sampler seed), `--precision` (`bf16`/`mps`/`full`), `--accelerator`, `--devices`, and `--fasta`/`--blacklist`/`--gaps` for custom genome references. Key default differences reflect each model's canonical training recipe:

| Flag | `train_bpnet.py` | `train_pomeranian.py` | `train_dalmatian.py` | `train_gopher.py` | `train_asap.py` |
|---|---|---|---|---|---|
| `--optimizer` | `adam` | `adamw` | `adamw` | `adamw` | `adamw` |
| `--learning-rate` | `1e-3` | `5e-4` | `1e-3` | `1e-3` | `1e-3` |
| `--weight-decay` | `0.0` | `0.01` | `1e-4` | `0.01` | `0.01` |
| `--scheduler-type` | `default` (constant) | `cosine` | `default` (constant) | `cosine` | `cosine` |
| `--input-len` | `2114` | `2112` | `2112` | `2048` | `2048` |
| `--output-len` | `1000` | `1024` | `1024` | `1024` | `2048` |
| `--output-bin-size` | `1` | `1` | `1` | `4` | `4` |
| Loss | `adaptive` (BPNetLoss) | `adaptive` (BPNetLoss) | `mse` (DalmatianLoss) | — (ProfilePoissonNLLLoss) | — (ProfilePoissonNLLLoss) |

## scATAC-seq Pseudobulk Tools

The `tools/scatac_pseudobulk.py` script generates per-cell-type pseudobulk BigWig coverage tracks and calls peaks from scATAC-seq fragment files using SnapATAC2. The output BigWig and narrowPeak files can be used directly as cerberus training targets and peak-based sampler intervals.

```bash
# Basic: per-cell-type BigWigs
python tools/scatac_pseudobulk.py \
    fragments.tsv.bgz gene_activity.h5ad output_dir/ \
    --genome hg38 --groupby cell_type

# Full pipeline: BigWigs + peaks + merged peak set
python tools/scatac_pseudobulk.py \
    fragments.tsv.bgz gene_activity.h5ad output_dir/ \
    --genome hg38 --groupby cell_type \
    --call-peaks --n-jobs 8
```

Key options:

| Flag | Description |
|---|---|
| `--genome` | Built-in genome (hg38, hg19, mm10, mm39) or use `--chrom-sizes` for custom |
| `--groupby` | obs column to group cells by (default: `cell_type`) |
| `--call-peaks` | Call peaks with MACS3 after BigWig generation |
| `--bulk-peaks` | Also call bulk (all-cells) peaks with MACS3 (slow, off by default) |
| `--no-merge` | Disable merging per-group peaks into `bulk_merge.narrowPeak.bed.gz` |
| `--overwrite` | Re-generate all outputs even if they already exist |
| `--counting-strategy` | `insertion` (Tn5 cut sites), `fragment`, or `paired-insertion` |
| `--normalization` | `raw`, `CPM`, `RPKM`, or `BPM` |
| `--n-jobs` | Total concurrent thread/process budget (default: max(4, cpu_count//2)) |
| `--sequential` | Disable parallel stage overlap |

Output layout (all files in `output_dir/`):

```
cell_type_A.bw                       # per-group BigWig
cell_type_A.narrowPeak.bed.gz        # per-group peaks (with --call-peaks)
cell_type_A.narrowPeak.bed.gz.tbi
...
bulk.bw                              # bulk (all-cells) BigWig (always)
bulk_call.narrowPeak.bed.gz          # with --bulk-peaks --call-peaks
bulk_merge.narrowPeak.bed.gz         # merged per-group peaks (with --call-peaks)
```

See `examples/scatac_kidney_pseudobulk.sh` for a complete example using the kidney scATAC-seq dataset.

## Visualization Tools

### Training Curves

`tools/plot_training_results.py` parses Lightning CSVLogger `metrics.csv` files and generates training curve plots.

```bash
python tools/plot_training_results.py path/to/run_dir
```

Automatically generates plots for all available metrics: loss, profile Pearson, profile MSE, log counts MSE, log counts Pearson, profile loss, count loss, and Dalmatian-specific recon/bias loss. Outputs PNG files (300 dpi) to a `plots/` subdirectory next to each `metrics.csv`.

### BiasNet ISM

`tools/plot_biasnet_ism.py` computes in-silico mutagenesis (ISM) for BiasNet models and generates IC logo + heatmap plots revealing learned sequence motifs.

```bash
python tools/plot_biasnet_ism.py path/to/biasnet_model --n-seqs 1000 --ism-window 31
```

Works with both standalone BiasNet and Dalmatian models (extracts the bias subnetwork automatically). Outputs `{prefix}_biasnet_ism.png` and `{prefix}_biasnet_ism.csv`.

### BiasNet Pairwise ISM

`tools/plot_biasnet_pairwise_ism.py` computes pairwise ISM to measure epistatic interactions between positions.

```bash
python tools/plot_biasnet_pairwise_ism.py path/to/biasnet_model --n-seqs 200 --detail-pair 11 19
```

Outputs a 4-panel plot (IC logo, epistasis heatmap, interaction profile, 4×4 detail) and a long-format CSV. Computationally intensive (~850K forward passes for default settings).

### Model Architecture Inspector

`tools/print_model_dims.py` prints layer-by-layer tensor shapes for any Cerberus model by running a dummy forward pass with hooks.

```bash
python tools/print_model_dims.py Pomeranian
python tools/print_model_dims.py BPNet --input_len 2114 --output_len 1000
python tools/print_model_dims.py BiasNet
```

Supports case-insensitive model name matching and automatically infers default dimensions from the model class.

## Model Interpretation (TF-MoDISco)

Use a two-step workflow:

1. `tools/export_tfmodisco_inputs.py` to export attribution inputs (`ohe.npz` + attribution NPZ).
2. `tools/run_tfmodisco.py` to run TF-MoDISco motifs/report on any compatible NPZ inputs.

What the export script does:

1. Loads model weights from `model.pt` (or best `.ckpt`) and matching `hparams.yaml`.
2. Rebuilds the matching `CerberusDataModule` and split (`train` / `val` / `test`).
3. Exports:
   - `ohe.npz` (DNA one-hot sequence; shape `(N, 4, L)`)
   - `shap.npz` (attribution scores; shape `(N, 4, L)`)

What the TF-MoDISco runner script does:

1. Validates `ohe` and attribution NPZ inputs (expects key `arr_0`, shape `(N, 4, L)`).
2. Runs `modisco motifs`.
3. Optionally runs `modisco report`.

Prerequisites:

```bash
pip install modisco-lite
# Optional (only if export uses --attribution-method integrated_gradients):
pip install captum

# Or install all cerberus extras:
pip install -e ".[extras]"
```

Step 1: export attribution NPZ files:

```bash
python tools/export_tfmodisco_inputs.py \
    --checkpoint-dir models/my_pomeranian/single-fold \
    --fold 0 \
    --split test \
    --n-examples 2000 \
    --seed 42 \
    --output-dir models/my_pomeranian/single-fold/tfmodisco \
    --target-mode log_counts
```

Export with ISM attribution:

```bash
python tools/export_tfmodisco_inputs.py \
    --checkpoint-dir models/my_pomeranian/single-fold \
    --fold 0 \
    --output-dir models/my_pomeranian/single-fold/tfmodisco \
    --attribution-method ism \
    --ism-start 800 \
    --ism-end 1200
```

Step 2: run TF-MoDISco motifs on exported NPZ files:

```bash
python tools/run_tfmodisco.py \
    --ohe-path models/my_pomeranian/single-fold/tfmodisco/ohe.npz \
    --attr-path models/my_pomeranian/single-fold/tfmodisco/shap.npz \
    --modisco-output models/my_pomeranian/single-fold/tfmodisco/modisco_results.h5
```

Generate HTML report after motifs:

```bash
python tools/run_tfmodisco.py \
    --ohe-path models/my_pomeranian/single-fold/tfmodisco/ohe.npz \
    --attr-path models/my_pomeranian/single-fold/tfmodisco/shap.npz \
    --modisco-output models/my_pomeranian/single-fold/tfmodisco/modisco_results.h5 \
    --run-report \
    --meme-db motifs.meme
```

TF-MoDISco recommended motif database for human data:

- Each pattern produced by TF-MoDISco is compared against a motif database using TOMTOM.
- A good default for human analyses is `MotifCompendium-Database-Human.meme.txt` from MotifCompendium.
- Source recommendation: https://github.com/kundajelab/tfmodisco?tab=readme-ov-file#generating-reports

```bash
mkdir -p data/motifs
curl -L \
  -o data/motifs/MotifCompendium-Database-Human.meme.txt \
  https://raw.githubusercontent.com/kundajelab/MotifCompendium/refs/heads/main/pipeline/data/MotifCompendium-Database-Human.meme.txt

python tools/export_tfmodisco_inputs.py \
    --checkpoint-dir models/my_pomeranian/single-fold \
    --fold 0 \
    --output-dir models/my_pomeranian/single-fold/tfmodisco \
    --target-mode log_counts

python tools/run_tfmodisco.py \
    --ohe-path models/my_pomeranian/single-fold/tfmodisco/ohe.npz \
    --attr-path models/my_pomeranian/single-fold/tfmodisco/shap.npz \
    --modisco-output models/my_pomeranian/single-fold/tfmodisco/modisco_results.h5 \
    --run-report \
    --meme-db data/motifs/MotifCompendium-Database-Human.meme.txt
```

Common interpretation target modes (`--target-mode`):

| Mode | Meaning |
|---|---|
| `log_counts` | Attribution to predicted log total counts (default) |
| `profile_bin` | Attribution to one profile-logit bin (`--bin-index`) |
| `profile_window_sum` | Attribution to summed profile logits in `[start:end)` (`--window-start`, `--window-end`) |
| `pred_count_bin` | Attribution to one predicted count bin (softmax(logits) × exp(log_counts)) |
| `pred_count_window_sum` | Attribution to summed predicted counts in a window |

Notes:

1. Use the same `--seed` as training for comparable sampler behavior (`tools/train_*.py` defaults to `42`).
2. For `peak` samplers, the script defaults to positive intervals only (`IntervalSampler` source).
3. Attribution method is configurable via `--attribution-method`:
   - `integrated_gradients` (default, requires `captum`)
   - `ism` (forward-pass mutation deltas; can be expensive over full input length)
4. `tools/run_tfmodisco.py` can consume any compatible attribution NPZ, not just outputs from `tools/export_tfmodisco_inputs.py`.
5. `modisco motifs` expects NPZ key `arr_0`; export scripts should write NPZ in that format.

## Next Steps

Once you have trained a model, you can use the prediction tools to evaluate it or generate genome-wide tracks.
See [Prediction (Inference)](prediction.md) for details on `ModelEnsemble`, `predict_intervals_batched`, and CLI tools like `tools/export_predictions.py`.
