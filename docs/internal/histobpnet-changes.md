# HistoBPNet Architecture and Training Changes Since Fork Point (Commit 4d67716)

## Overview

This document summarizes all significant changes to the histobpnet codebase introduced after commit `4d67716`. The changes represent a major refactor of the data pipeline, the introduction of three new HistoBPNet model families (v1, v2, v3), substantial extensions to the BPNet base architecture, and a unified training entry-point. The overall design philosophy shifts from a single-mode ChromBPNet-style ATAC-seq model toward a family of histone ChIP-seq count-prediction models that share the BPNet dilated-conv trunk but differ in how they consume and represent the histone control signal.

---

## 1. New Model Architectures: HistoBPNet v1, v2, and v3

All three versions wrap a single `BPNet` instance (no ChromBPNet-style bias sub-model) and are **count-only** models: the profile convolution (`fconv`) is disabled by setting `profile_kernel_size=0`, so `fconv=None` and `forward()` returns `pred_profile=None`. The models differ in how the histone control signal is represented, how the count head integrates it, and what data coordinates they use.

**Common model inputs across all three versions:**
- `x`: one-hot encoded DNA sequence, shape `(batch, 4, 2114)` — centered on the ATAC peak summit. This is the only input processed by the convolutional trunk.
- `x_ctl_hist`: pre-computed log-transformed histone control scalar(s), shape `(batch,)` for V2/V3 or `(batch, n_bins)` for V1 — fed directly into the count head, bypassing the trunk entirely. See section 2.2 for details.

**Common model output:** `(None, pred_count)` where `pred_count` has shape `(batch, 1)` for V2/V3 or `(batch, n_bins)` for V1. There is no profile output.

**What "ATAC-seq" means here:** The ATAC-seq peaks BED file is used only as a spatial index — it defines which genomic locations to train on. No ATAC-seq bigwig is ever read. The model never sees ATAC-seq signal. All bigwigs (`--bigwig`, `--bigwig_ctrl`) are histone ChIP-seq and histone input control, respectively.

### 1.1 HistoBPNetV1 (`histobpnet/model/histobpnet_v1.py`)

**Purpose:** Multi-resolution binned count prediction. Instead of predicting a single total-count scalar, the model predicts log-counts at several genomic bin widths simultaneously (default: 1 kb, 2 kb, 4 kb, 8 kb, 16 kb).

**What are output bins?** Output bins are symmetric genomic windows of different widths, all centered on the peak summit. For each peak, base-pair-resolution histone ChIP-seq counts are loaded in a window of `max(output_bins)` bp. `_split_counts_into_bins()` then extracts sub-windows for each bin width (e.g., the central 1000 bp, central 4000 bp, etc.). At training time, the per-bin counts are **summed** to a single number and **log-transformed**, producing one scalar per bin. The model's count head outputs one predicted log-count per bin, and MSE is computed across all bins.

These are **not** profiles and carry **no spatial/positional information**. Each value is one total count (in log space). The profile head is disabled, so there is no base-pair-resolution output.

**Motivation:** Multi-scale learning — narrow bins (1 kb) capture sharp, local enrichment at the peak summit, while broad bins (16 kb) capture diffuse histone signal spread, providing a richer training signal than a single total count.

**Count head tensor flow** (with `use_linear_w_ctrl=True`):
```
global_avg_pool(trunk) → (batch, 64)
    ↓ linear: Linear(64, 1)
(batch, 1)                 ← sequence-derived scalar
    ↓ cat with control vector (batch, 5)
(batch, 6)
    ↓ linear_w_ctrl: Linear(6, 5)
(batch, 5)                 ← one predicted log-count per bin width
```
Note: this intended flow is **broken by a shape bug** — see section 8, bug #3b.

**V1 configuration:**
- `n_count_outputs = len(output_bins)` (default 5), `n_control_tracks = len(output_bins)` (default 5).
- Each bin gets its own histone control scalar, unlike V2/V3 where there is one control scalar total.
- `output_bins` is specified as a comma-separated string, e.g., `"1000, 2000, 4000, 8000, 16000"`.

**V1 loss** (`histobpnet_wrapper_v1.py`):
- Target: `true_binned_logsum` of shape `(batch, 5)`, computed as `torch.log(per_bin_sums + 1e-6)`. Note: uses `log` not `log1p` (inconsistent with V2/V3; flagged as TODO).
- Control: `true_binned_logsum_ctl` of shape `(batch, 5)`, same log transform applied to per-bin histone control sums.
- Loss = `MSE(pred_count, true_binned_logsum)`, averaged over bins and batch.
- Alpha is forced to 1 for all histone models.

**Epoch-end metrics:** Pearson correlation is computed and logged separately for each bin width (e.g., `val_count_pearson_1000bp`).

### 1.2 HistoBPNetV2 (`histobpnet/model/histobpnet_v2.py`)

**Purpose:** Single-resolution count prediction using an ATAC-to-histone peak mapping file (`atac_hgp_map`). The mapping links each ATAC peak to its corresponding histone gapped peak (HGP), which may be at a different genomic location.

**Full data flow, step by step:**

1. **Input files:** `--peaks` = ATAC-seq peaks BED (spatial index only); `--bigwig` = histone ChIP-seq bigwig (target signal); `--bigwig_ctrl` = histone input control bigwig; `--atac_hgp_map` = 6-column TSV mapping `(chrom, start, end)` → `(hist_chrom, hist_start, hist_end)`.

2. **Sequence extraction** (`get_seq`): DNA is fetched from the reference genome centered on the **ATAC peak summit**, width = 2114 bp. Shape: `(n_peaks, 2114, 4)`. This is the model's primary input.

3. **Histone signal extraction** (`get_cts` with `atac_hgp_df` + `get_total_cts=True`): For each ATAC peak, the mapping file is consulted to find the corresponding histone peak coordinates. The histone ChIP-seq bigwig is read at those **histone peak coordinates** (not the ATAC coordinates), and all base-pair values are summed into a **single scalar**. The same is done for the histone input control bigwig. Result: one scalar per peak for ChIP and one scalar per peak for control, each stored as `np.array([sum])` with shape `(1,)`.

4. **Dataset `__getitem__`** returns per-example:
   - `onehot_seq`: `(4, 2114)` — DNA sequence (transposed to channels-first)
   - `profile`: `(1,)` — total histone ChIP-seq counts at the mapped histone peak (a scalar, since `get_total_cts=True`)
   - `profile_ctrl`: `(1,)` — total histone input control counts at the same location
   - `peak_status`: `int` — 1 for peak, 0 for non-peak

5. **Wrapper pre-processing** (`histobpnet_wrapper_v2.py`): Since `profile` is 1D (scalar), `true_logsum = profile.log1p()` and `true_logsum_ctl = profile_ctrl.log1p()`, both `(batch,)`.

6. **Model forward** (`histobpnet_v2.py`): `self.bpnet(x, x_ctl_hist=true_logsum_ctl)`. The trunk processes DNA sequence only; the count head receives the log-control scalar via the `x_ctl_hist` pathway (see section 2.2).

7. **Count head output:** `(batch, 1)`, squeezed to `(batch,)` in the wrapper.

8. **Loss:** `MSE(y_count, true_logsum)` — both `(batch,)`.

**V2 architecture:**
- `n_count_outputs=1`, `n_control_tracks=1`.
- `feed_ctrl` flag (default `True`): when `False`, histone control is not passed into the model; instead the model predicts a log fold-change and control is added post-hoc: `y_count = y_lfc + true_logsum_ctl`.
- `n_cc = config.n_control_tracks if config.feed_ctrl else 0` — controls whether the BPNet is built with a control input at all.

### 1.3 HistoBPNetV3 (`histobpnet/model/histobpnet_v3.py`)

**Purpose:** Identical model architecture and loss to V2, but uses a different data loading path that does not require the `atac_hgp_map` file.

**V2 vs V3 — the only difference is where the histone bigwig is read:**
- **V2** uses two separate coordinate systems: DNA sequence is read centered on the ATAC peak summit, but the histone bigwig is read at **different coordinates** looked up via the `atac_hgp_map` file (the corresponding histone gapped peak, which may be at a completely different genomic location).
- **V3** uses one coordinate system: both DNA sequence and the histone bigwig are read centered on the **ATAC peak summit** (`r['start'] + r['summit'] ± outputlen//2`). There is no mapping file and no coordinate indirection. V3 assumes the histone signal of interest is co-located with the ATAC peak.

**Key constraint:** `outputlen > 0` is required (asserted in `HistoBPNetDatasetV3.__init__`) because `get_cts` uses `outputlen` as the window width for reading the histone bigwig. In V2 this is not needed since the window is defined by the histone peak coordinates in the mapping file.

### Summary Comparison Table

| Feature | V1 | V2 | V3 |
|---|---|---|---|
| Profile head | Disabled (`fconv=None`) | Disabled (`fconv=None`) | Disabled (`fconv=None`) |
| Count head output shape | `(batch, 5)` | `(batch, 1)` | `(batch, 1)` |
| `n_control_tracks` | 5 (one per bin) | 1 | 1 |
| Histone control input shape | `(batch, 5)` | `(batch,)` | `(batch,)` |
| Where histone bigwig is read | At ATAC summit, per-bin windows | At mapped histone peak coords | At ATAC summit |
| Requires `atac_hgp_map` | No | Yes | No |
| Requires `outputlen > 0` | No | No | Yes |
| Log transform | `log(sum + 1e-6)` | `log1p(sum)` | `log1p(sum)` |
| `feed_ctrl` flag | N/A | Yes | Yes |

---

## 2. Changes to BPNet and ChromBPNet Base Architectures

### 2.1 BPNet (`histobpnet/model/bpnet.py`)

The `BPNet` class underwent the most extensive changes of any single file.

**New constructor parameters:**
- `n_count_outputs: int = 1` — number of scalars the count head outputs. 1 for ChromBPNet/V2/V3, `len(output_bins)` for V1.
- `for_histone: str = None` — a string tag (`'histobpnet_v1'`, `'histobpnet_v2'`, `'histobpnet_v3'`) that modifies count head wiring. See below.
- `use_linear_w_ctrl: bool = True` — selects between two count head designs. See below.
- `verbose: bool` default changed from `True` to `False`.

**Profile head (`fconv`) made optional:**
- Previously `fconv` was always constructed. Now, if `profile_kernel_size == 0`, `self.fconv = None` and `forward()` returns `pred_profile=None`.

**New `x_ctl_hist` argument in `forward()`:**
- `forward()` now accepts `x_ctl_hist` in addition to the existing `x_ctl`. "hist" stands for **histone** — this is the control pathway used by all HistoBPNet models. See section 2.2 for the full comparison between the two control pathways.

**Refactored count head — two-linear-layer design:**

The count head has two possible configurations, controlled by `use_linear_w_ctrl`:

When `use_linear_w_ctrl=False` (pre-fork behavior):
```
global_avg_pool(trunk) → (batch, n_filters)
  → cat([pooled, ctl], dim=-1) → (batch, n_filters + n_count_control)
  → Linear(n_filters + n_count_control, n_count_outputs) → (batch, n_count_outputs)
```

When `use_linear_w_ctrl=True` (new default, used by all histone models):
```
global_avg_pool(trunk) → (batch, n_filters)
  → linear: Linear(n_filters, 1) → (batch, 1)        ← compress sequence to one scalar
  → cat([seq_scalar, ctl], dim=-1) → (batch, 1 + n_count_control)
  → linear_w_ctrl: Linear(1 + n_count_control, n_count_outputs) → (batch, n_count_outputs)
```

Where `n_count_control` depends on model type:
- ChromBPNet: `n_count_control=0` (no control tracks) → falls into `use_linear_w_ctrl=False` branch regardless, producing `Linear(n_filters, n_count_outputs)`.
- V2/V3: `n_count_control=1`.
- V1: `n_count_control = n_control_tracks = len(output_bins)` (typically 5).

**`from_keras` class method extended:**
- Now accepts an optional `instance` parameter. If provided, weights are loaded into an existing model instance rather than constructing a new one. This allows initializing HistoBPNet trunk weights from a ChromBPNet-without-bias `.h5` checkpoint.
- Weight shape compatibility check: `if model.linear.weight.shape == incoming.shape` guards the linear weight loading. For histone models with a modified count head, the linear weight shapes typically don't match the ChromBPNet checkpoint, so the count head weights are silently skipped (re-initialized from scratch).

**Assertion hardening:**
- `if crop_len > 0:` guard in `get_embs_after_crop()` replaced with `assert crop_len > 0`.
- `if crop_size > 0:` / `else: F.pad(...)` branch in the control cropping block of `forward()` replaced with `assert crop_size > 0`, removing the padding fallback.

### 2.2 How the Control Signal Is Used (All Model Types)

There are two completely separate control pathways in `BPNet.forward()`. They are never used simultaneously. In all cases, the control is **concatenated** with sequence-derived features and passed through a learned linear layer — the model learns the relationship (it is not hard-coded as addition or subtraction).

#### Pathway 1: `x_ctl` (base-pair-resolution, used by original BPNet for TF binding)

`x_ctl` is a full-resolution control bigwig signal with shape `(batch, n_strands, seq_length)`. It enters the model raw and is processed inside `BPNet`:

**In the profile head** — concatenated as extra channels, then convolved:
```
trunk:  (batch, n_filters, L)           e.g. (batch, 64, L)
x_ctl:  (batch, n_control_tracks, L)    e.g. (batch, 2, L)   [cropped to match trunk length]
  → cat along dim=1 → (batch, n_filters + n_control_tracks, L)
  → fconv: Conv1d(66, n_outputs, kernel_size=75) → (batch, n_outputs, out_len)
```

**In the count head** — collapsed to a scalar, log-transformed, concatenated:
```
x_ctl: (batch, n_strands, L) → sum over strands and positions → (batch, 1) → log1p → (batch, 1)
pooled_seq: (batch, n_filters) from global_avg_pool(trunk)
  → cat → (batch, n_filters + 1) → Linear → (batch, 1)
```

**Used by:** Original BPNet for TF binding prediction (with `n_control_tracks > 0`). Not used by ChromBPNet or any HistoBPNet variant.

#### Pathway 2: `x_ctl_hist` (pre-computed histone control, used by HistoBPNet V1/V2/V3)

`x_ctl_hist` is a log-transformed scalar (or vector for V1) computed **outside the model** in the wrapper. It enters the count head directly — it is never spatially processed and never touches the profile head (which is disabled).

**Pre-computation in the wrapper** (before calling the model):
- V2/V3: `true_logsum_ctl = profile_ctrl.log1p()` → shape `(batch,)`
- V1: `true_binned_logsum_ctl = torch.log(per_bin_ctrl_sums + 1e-6)` → shape `(batch, 5)`

**Inside `count_head()`:** `x_ctl_hist` is used directly (no further log transform). If both `x_ctl` and `x_ctl_hist` are provided, `x_ctl_hist` takes priority (overwrites `ctl`).

For V2/V3 with `use_linear_w_ctrl=True`:
```
pooled_seq: (batch, 64) → linear(64, 1) → (batch, 1)
x_ctl_hist: (batch,) → unsqueeze(-1) → (batch, 1)
  → cat → (batch, 2) → linear_w_ctrl(2, 1) → (batch, 1)
```

**`feed_ctrl=False` mode** (V2/V3 only): the only case where control handling is hard-coded. The histone control is not passed into the model; the model predicts a log fold-change, and control is added post-hoc in the wrapper:
```python
y_count = y_lfc + true_logsum_ctl   # addition in log space = multiplication in linear space
```

#### ChromBPNet — no control pathway

ChromBPNet has `n_control_tracks=0`. Neither `x_ctl` nor `x_ctl_hist` is used. Instead, the bias sub-model itself acts as the control — it sees only sequence and learns sequence-intrinsic bias. The two sub-models are combined: profile via addition (`acc_profile + bias_profile`); counts via log-sum-exp (`log(exp(acc_counts) + exp(bias_counts))`).

### 2.3 ChromBPNet (`histobpnet/model/chrombpnet.py`)

- Constructor now requires a `BPNetModelConfig` object.
- The bias sub-model is hardcoded to `n_filters=128, n_layers=4` rather than mirroring the accessibility model.
- `tf_style_reinit()` method added: applies Xavier uniform initialization to all `Conv1d` and `Linear` weights, zeros all biases. Only applied to `self.model` (the accessibility component), not `self.bias` (assumed to be loaded from a pre-trained checkpoint).
- Log-sum-exp count combination: `y_counts = log(exp(acc_counts) + exp(bias_counts))` implemented via `_Log` and `_Exp` wrapper modules for DeepLIFT/SHAP compatibility.

---

## 3. New Data Loading Infrastructure

### 3.1 DataModule (`histobpnet/data_loader/datamodule.py`) — Newly Added

The monolithic `dataset.py` file (which contained both `DataModule` and `ChromBPNetDataset`) was split into separate files. `datamodule.py` is a pure Lightning `LightningDataModule` responsible for:

- Dispatching to the correct `Dataset` class based on `config.model_type`.
- Computing per-device batch size as `config.batch_size // gpu_count` for correct DDP behavior.
- Loading and splitting peak/negative region DataFrames at construction time (not lazily).
- Providing `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `negative_dataloader()`, and `chrom_dataloader()` / `chrom_dataset()` for prediction.
- `median_count` property (lazy, cached): computes the median total count over the training+validation set, used as the ChromBPNet alpha scaling factor.

**Key behavioral changes vs. old `DataModule` in `dataset.py`:**
- Old `DataModule.__init__` took `(config, args)`; new one takes `(config: DataConfig, gpu_count: int)`. The coupling to the argparse namespace is removed.
- `reload_dataloaders_every_n_epochs=1` in the trainer (see `main.py`) replaces the old `train_dataloader()` manually calling `self.train_dataset.crop_revcomp_data()`.
- The `setup()` method passes `config` directly to `Dataset.__init__` rather than passing individual kwargs like `genome_fasta`, `cts_bw_file`, `add_revcomp`.
- Training validation split: train uses `rc_frac=config.rc_frac` (0 for histone, 0.5 for ChromBPNet); validation and all prediction dataloaders use `rc_frac=0`.
- Negative sampling: train uses `negative_sampling_ratio=config.negative_sampling_ratio`; test and prediction use `negative_sampling_ratio=-1` (all regions included).

### 3.2 ChromBPNetDataset (`histobpnet/data_loader/chrombpnet_dataset.py`) — Newly Added (Extracted)

Extracted from the old `dataset.py` into its own file. Interface changes:
- Now takes `config: DataConfig` as a named argument rather than individual path strings.
- `add_revcomp` boolean replaced by `rc_frac: float` (fraction of examples to augment with reverse complement).
- `return_coords` removed; coordinates are always tracked internally.
- `__getitem__` now returns a `peak_status` field (integer: 1 for peak, 0 for non-peak) in addition to `onehot_seq` and `profile`. This enables per-epoch Pearson scatter plots split by peak/non-peak status.
- `validate_mode(mode)` utility function guards valid mode strings: `['train', 'val', 'test', 'chrom', 'negative']`.

### 3.3 HistoBPNetDatasetV1 (`histobpnet/data_loader/histobpnet_dataset_v1.py`) — Newly Added

Extends `ChromBPNetDataset` for multi-bin histone prediction.

**Key features:**
- Requires `negative_sampling_ratio == -1` (no subsampling) and `rc_frac == 0` (no reverse-complement augmentation).
- Calls `load_data()` with `output_bins=config.output_bins` and `pass_zero_mode=config.pass_zero_mode`.
- Post-loads: runs `_split_counts_into_bins()` to partition the counts array into per-bin windows centered on the peak summit. For each bin width `w`, a symmetric window of `w` bp around the summit is extracted, plus `max_jitter` bp of padding on each side.
- Stores `per_bin_peak_cts_dict` and `per_bin_peak_cts_ctrl_dict`, both `dict[int -> np.ndarray]` keyed by bin width, where each value has shape `(n_peaks, bin_width + 2*max_jitter)`.
- `__getitem__` returns `per_bin_profile` and `per_bin_profile_ctrl` as dictionaries mapping bin width → `np.ndarray` of shape `(bin_width,)`.

### 3.4 HistoBPNetDatasetV2 (`histobpnet/data_loader/histobpnet_dataset_v2.py`) — Newly Added

Extends `ChromBPNetDataset` for single-resolution histone prediction with ATAC-to-histone peak mapping.

**Key features:**
- Requires `max_jitter == 0` and `rc_frac == 0`.
- Requires `config.atac_hgp_map` to be provided (raises assertion error otherwise).
- Loads the ATAC-HGP map as a TSV with columns `[chrom, start, end, hist_chrom, hist_start, hist_end]` and calls `add_peak_id()`.
- Passes `atac_hgp_df`, `get_total_cts=True`, `skip_missing_hist`, `ctrl_scaling_factor`, and `outputlen_neg` to `load_data()`.
- Because `get_total_cts=True`, `__getitem__` returns `profile` and `profile_ctrl` each as a scalar `(1,)` — the total summed histone signal at the mapped histone peak. `peak_status` is also returned.

### 3.5 HistoBPNetDatasetV3 (`histobpnet/data_loader/histobpnet_dataset_v3.py`) — Newly Added

Nearly identical to V2 but without the ATAC-to-histone map requirement.

**Key differences from V2:**
- Does not load or use `atac_hgp_map`. Histone bigwig is read at the ATAC peak summit coordinates (same as DNA sequence), not at separate histone peak coordinates.
- Does not pass `atac_hgp_df` or `skip_missing_hist` to `load_data()`.
- Asserts `outputlen > 0` (needed as the window width for reading the histone bigwig).
- Otherwise identical interface and output format to V2.

### 3.6 Genome Registry (`histobpnet/data_loader/genome.py`) — Newly Added

A new genome data management module using `pooch` for file caching and downloading:

- `EnhancedDataset` class wraps a `pooch.Pooch` dataset with lazy initialization, safer fetch with error handling, and cache introspection.
- `hg38_datasets()`, `mm10_datasets()`, `hg19_datasets()`, `motifs_datasets()` factory functions register checksummed files on Zenodo (records 12193595 for hg38, 12193429 for mm10).
- Fold split JSON files (`fold_0.json` through `fold_4.json`) are fetched from Zenodo and parsed by `DataConfig` to determine train/validation/test chromosomes. This replaces the old `fold_path` argument that required the user to provide a local JSON path.
- `Genome` class provides lazy-loaded `fasta` and `annotation` properties.
- Module-level objects `hg38 = GRCh38` are pre-built at import time for convenience; `mm10` and `hg19` are `None` (commented out pending need).

---

## 4. Training Pipeline Changes (`scripts/train/main.py`)

### 4.1 Model Factory

The `create_model_wrapper()` function replaces ad-hoc model instantiation. It dispatches on `args.model_type`:
- `'bpnet'` → `BPNetWrapper`
- `'chrombpnet'` → `ChromBPNetWrapper`
- `'histobpnet_v1'` → `HistoBPNetWrapperV1`
- `'histobpnet_v2'` → `HistoBPNetWrapperV2`
- `'histobpnet_v3'` → `HistoBPNetWrapperV3`

Each wrapper's constructor calls `.load_pretrained_chrombpnet()` to optionally warm-start from a ChromBPNet-without-bias checkpoint (`.h5` file loaded via `BPNet.from_keras(instance=...)`).

### 4.2 Alpha Scaling

- For all histone models: `args.alpha = 1` (forced in `train()`).
- For ChromBPNet: `args.alpha = datamodule.median_count / 10`, matching the original ChromBPNet training paper logic.

### 4.3 Trainer Configuration

Key Lightning trainer settings:
- `reload_dataloaders_every_n_epochs=1`: re-samples negatives and applies jitter/augmentation each epoch.
- `check_val_every_n_epoch=1`: validates every epoch.
- `val_check_interval=None`: run full validation at epoch end, not mid-epoch.
- `EarlyStopping(monitor='val_loss', patience=5)`.
- `ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True)`.
- `gradient_clip_val=args.gradient_clip` (CLI arg, default `None`).
- `precision=args.precision` (CLI arg, default 32).

### 4.4 Post-Training Prediction

After training completes, `predict()` is immediately called on the test chromosomes (`chrom="test"`) using the best checkpoint. The prediction loop:
1. Calls `dm.chrom_dataloader(chr)` to get a dataloader over all regions on the target chromosomes.
2. Runs `trainer.predict(model, dataloader)`.
3. Validates batch count consistency.
4. Calls `compare_with_observed()` and, for ChromBPNet, `save_predictions()`.
5. Profile prediction is skipped (`skip_profile=True`) for all histone model types.

### 4.5 Dual Artifact Saving

After training, the model saves **both** a Lightning `.ckpt` (includes optimizer state, for resumption) and a raw PyTorch state dict `.pt` file (for clean inference loading). The `.pt` file saves only `model.bpnet.state_dict()` for histone models (the inner BPNet trunk), or `model.model.state_dict()` for ChromBPNet (the accessibility sub-network).

### 4.6 WandB Integration

- WandB run is initialized in `setup()` with the full argparse namespace as the run config, plus the process ID.
- Run name = `args.name + "|" + instance_id`.
- Scatter plots (predicted vs. true log-counts, split by All / Peaks / Non-peaks) are logged as WandB images at the end of each epoch via `_log_plot()`.
- `--skip_wandb` flag disables WandB initialization.

### 4.7 New CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--lr` | 0.001 | Learning rate |
| `--optimizer_eps` | 1e-7 | Adam epsilon |
| `--max_epochs` | 100 | Max training epochs |
| `--precision` | 32 | Torch training precision |
| `--gradient_clip` | None | Gradient clipping value |
| `--skip_wandb` | False | Disable WandB |
| `--adjust_bias` | False | Adjust bias model logcounts |
| `--cvd` | None | CUDA_VISIBLE_DEVICES override |
| `--shap` | 'counts' | SHAP target for interpretation |
| `--chrom` | 'test' | Chromosome set for prediction |

---

## 5. Configuration Changes

### 5.1 `BPNetModelConfig` (`histobpnet/model/model_config.py`)

**New parameters:**
- `use_linear_w_ctrl: bool = True` — selects the two-linear-layer count head (new default; pre-fork behavior corresponds to `False`).
- `feed_ctrl: bool = True` — for V2/V3, whether to feed the histone control scalar into the model (vs. adding it as a post-hoc offset).
- `n_count_outputs: int = None` — number of count output scalars; auto-set from `model_type`.
- `output_bins: str = None` — comma-separated bin widths; parsed to a list of ints; for V1 defaults to `"1000, 2000, 4000, 8000, 16000"`.

**Auto-configuration logic by `model_type`:**
- `out_dim`: 1000 for ChromBPNet, **0** for all histone models (disables profile head).
- `profile_kernel_size`: 75 for ChromBPNet, **0** for all histone models.
- `n_control_tracks`: 0 for ChromBPNet; `len(output_bins)` for V1; 1 for V2/V3.
- `n_count_outputs`: `len(output_bins)` for V1; 1 for V2/V3.
- `output_bins`: only applicable to V1; ignored for other types.

**Validation:** `is_histone()` utility checks whether the model type starts with `"histobpnet"`.

### 5.2 `DataConfig` (`histobpnet/data_loader/data_config.py`)

**Removed parameters:**
- `data_dir` (previously used for `{data_dir}/peaks.bed` path templates)
- `data_type` (replaced by `model_type`)
- `training_chroms`, `validation_chroms`, `test_chroms` (now derived from fetched fold JSON)
- `fold_path` (replaced by auto-download via `genome.py`)
- `plus`, `minus`, `ctl_plus`, `ctl_minus`, `background` (strand-specific and background bigwig paths)
- `rc` (replaced by `rc_frac`)

**New parameters:**
- `extra_kwargs: dict` — catch-all for `model_type` and `output_bins`.
- `bigwig_ctrl: str = None` — histone input control bigwig path (replaces `ctl_plus`/`ctl_minus`).
- `genome: str = 'hg38'` — selects which Zenodo genome registry to use.
- `atac_hgp_map: str = None` — path to ATAC-to-histone peak mapping file (required for V2).
- `skip_missing_hist: str = "N/A"` — controls behavior when a histone peak has no corresponding ATAC peak; valid values: `"Yes"`, `"No"`, `"N/A"`.
- `pass_zero_mode: str = "N/A"` — controls how zero-signal regions are handled; valid values: `"zero_seq"`, `"zero_ctl"`, `"zero_cts"`, `"N/A"`.
- `rc_frac: float = None` — replaces the boolean `rc`; auto-set to 0.5 for ChromBPNet, 0 for histone.
- `ctrl_scaling_factor: float = None` — multiplicative scaling factor applied to the histone input control bigwig signal; defaults to 1.0.
- `outputlen_neg: int = None` — output window length for negative/non-peak regions; defaults to 1000 for histone models.

**Auto-configuration logic by `model_type`:**
- `shift`: 500 for ChromBPNet (jitter), **0** for histone (no jitter).
- `rc_frac`: 0.5 for ChromBPNet, **0** for histone.
- `out_window`: 1000 for ChromBPNet, **0** for histone (count-only).
- Chromosome splits: loaded from `fold_{fold}.json` via the `genome.py` registry rather than from a user-supplied path.

**Validation additions:**
- `_validate_model_type()` replaces `_validate_data_type()`.
- `_validate_output_bins()` checks all bins are positive integers.
- `_validate_skip_missing_hist()` and `_validate_pass_zero_mode()` check valid enum values.
- `out_window` constraint relaxed: previously `out_window <= 0` raised an error; now `out_window < 0` raises an error (allowing `out_window=0` for count-only histone models).
- `in_window < out_window` constraint commented out (pending fix for histone case).

---

## 6. Losses, Optimizer, and Training Details

### 6.1 Loss Functions

**ChromBPNet/BPNet (unchanged):** Combined multinomial negative log-likelihood (profile) + MSE (counts):
```
loss = beta * multinomial_nll(y_profile, true_profile) + alpha * mse_loss(y_count, true_counts)
```
where `beta=1` and `alpha = median_count / 10`.

**HistoBPNet V1, V2, V3 (new):** Pure MSE in log space, no profile loss:
```
loss = mse_loss(y_count, true_logsum)
```
For V1 `y_count` and `true_logsum` are both `(batch, 5)`; MSE is averaged over bins and batch. For V2/V3 they are both `(batch,)`.

**Log transformation inconsistency:** V1 uses `log(sum + 1e-6)`. V2/V3 use `log1p(sum)`. Flagged with TODO comments in the code.

### 6.2 Optimizer

Adam with configurable `lr` (default 0.001) and `eps` (default 1e-7). `eps=1e-7` matches TensorFlow's Adam default. Configured in `ModelWrapper.configure_optimizers()`.

### 6.3 Weight Initialization

`tf_style_reinit()` is called in the constructors of `ChromBPNet`, `HistoBPNetV1`, `HistoBPNetV2`, and `HistoBPNetV3`. It applies:
- Xavier uniform (`nn.init.xavier_uniform_`) to all `Conv1d` and `Linear` weight matrices.
- Zero initialization to all biases.

This matches TensorFlow/Keras default initialization. Note: in V1 this method has a bug (see section 8, bug #3a).

### 6.4 Pretrained Weight Loading

All three HistoBPNet wrappers implement `load_pretrained_chrombpnet()`, which:
1. Calls `BPNet.from_keras(chrombpnet_wo_bias_path, instance=self.model.bpnet)` to load a ChromBPNet-without-bias Keras `.h5` checkpoint into the BPNet trunk.
2. Does **not** freeze the loaded weights — the trunk is fine-tuned on histone data.
3. Count head linear weights are loaded only if the shape matches. For histone models with a modified count head, the shapes typically don't match, so the count head is re-initialized from scratch.

### 6.5 Bias Model Adjustment

`adjust_bias_model_logcounts()` in `model_wrappers.py`: given a frozen bias model and a negative-region dataloader, computes the mean delta between observed and predicted log-counts over negative regions and adds it to the Dense layer bias. This corrects the bias model's count scale before training the accessibility model, matching the original ChromBPNet training protocol.

### 6.6 Epoch-End Scatter Plots

After each training and validation epoch, `_epoch_end()` in `ModelWrapper`:
- For V2/V3: computes Pearson correlation and logs a density scatter plot of predicted vs. true log-counts, stratified into three panels: All, Peaks only, Non-peaks only.
- For V1: logs per-bin Pearson correlations (no scatter plot).
- Plots are pushed to WandB as images.

---

## 7. Deleted and Refactored Files

- `histobpnet/data_loader/dataset.py` — deleted. Its `DataModule` class moved to `datamodule.py`; its `ChromBPNetDataset` class moved to `chrombpnet_dataset.py`.
- `histobpnet/model/bpnet_wrapper.py` — new file containing `BPNetWrapper` and `ChromBPNetWrapper`, extracted from what was previously inline in the training script or a monolithic wrapper file.
- `histobpnet/model/model_wrappers.py` — new base class `ModelWrapper(LightningModule)` that provides the shared training loop, optimizer, metric accumulation, and WandB logging. All model-specific wrappers inherit from this.

---

## 8. Known Bugs and Open Questions

1. **log vs log1p inconsistency:** V1 wrapper uses `torch.log(sum + 1e-6)` while V2/V3 use `log1p`. Flagged with TODOs in the code.
2. **Count head bias placement:** When `use_linear_w_ctrl=True`, two linear layers are used. A code comment notes uncertainty about which layer's bias term should be controlled by `count_output_bias`.
3a. **V1 `tf_style_reinit` bug:** The method iterates over `self.model.modules()` but V1 stores the BPNet as `self.bpnet`, not `self.model`. The initialization silently has no effect for V1. (V2 and V3 correctly use `self.modules()`.)
3b. **V1 `count_head` shape bug:** `bpnet.py:227` does `ctl = x_ctl_hist.unsqueeze(-1)`. For V2/V3, `x_ctl_hist` is `(batch,)` → `(batch, 1)` (correct). For V1, `x_ctl_hist` is `(batch, 5)` → `(batch, 5, 1)` (3D), which cannot be concatenated with the 2D `pred_count` of shape `(batch, 1)` at line 237. The `unsqueeze` was written for the V2/V3 scalar case and is incorrect for V1's multi-bin vector.
4. **ChromBPNet `tf_style_reinit` scope:** Only `self.model` (accessibility) is re-initialized, not `self.bias`. This assumes the bias model will be loaded from a pre-trained checkpoint.
5. **DDP strategy:** A `DDPStrategy(find_unused_parameters=False)` line is commented out in the trainer; the current default DDP is used instead.
6. **`in_window < out_window` validation:** This constraint is commented out in `DataConfig` as it does not hold for histone models where `out_window=0`. Needs a conditional fix.
