# Repository Analysis Dump

This document contains a detailed analysis of the repositories in `repos/` to inform the development of `cerberus`.

## Objectives
- Re-implement `ASAP` models.
- Use `bpnet-lite` infrastructure.
- Consult `chrombpnet-pytorch` training algorithms details (background, loss, etc.).
- Use `GenVarLoader` for IO.
- Ensure compatibility with `tangermeme`.

---

## 1. ASAP (Target Models)

**Location**: `repos/ASAP`

### File Analysis

#### `src/asap/`
- **`dataset.py`**:
    - **Concept**: High-level dataset creation functions. Orchestrates `WGDataset` and `PeakDataset`.
    - **Functions**: `training_datasets`, `peak_dataset`, `robustness_peak_dataset`, `wg_dataset`, `robustness_wg_dataset`.
- **`task.py`**:
    - **Concept**: Main entry points for training, evaluation, and prediction tasks.
    - **Functions**: `_get_model`, `train_model`, `eval_model`, `eval_robustness`, `predict_snv_atac`, `export_predictions`.

#### `src/asap/models/`
- **`convnext_cnn.py`**:
    - **Concept**: The primary ConvNeXt-based model architecture.
    - **Functions**: `ConvNeXtCNN.__init__`, `ConvNeXtCNN.forward`.
- **`convnext_dcnn.py`**, **`convnext_lstm.py`**, **`convnext_transformer.py`**:
    - **Concept**: Variations of ConvNeXt combined with DCNN, LSTM, or Transformer blocks.
- **`dilated_cnn.py`**:
    - **Concept**: A dilated CNN architecture (Basenji-like).
- **`vanilla_cnn.py`**:
    - **Concept**: A simple baseline CNN.
- **`layers/convnext.py`**:
    - **Concept**: Implementation of `ConvNeXtV2Block` using `Conv1d`, `LayerNorm`, `GRN`.
- **`layers/unmap_predictor.py`**:
    - **Concept**: Predicts mappability/unmappable regions.

#### `src/asap/dataloader/`
- **`base.py`**:
    - **Concept**: Abstract base class `BaseDataset` for ASAP datasets. Handles common logic like chromosome setting and BigWig writing.
    - **Functions**: `idx_to_ohe`, `idx_to_seq`, `write_predictions_to_bigwig`.
- **`bounded.py`**:
    - **Concept**: `BoundedDataset`. Inherits from Base. Handles datasets defined by bounds (regions).
    - **Functions**: `_generate_chrom_data`.
- **`wg.py`**:
    - **Concept**: `WGDataset`. Represents Whole Genome dataset using sliding windows.
- **`peak.py`**:
    - **Concept**: `PeakDataset`. Represents dataset centered on peaks.
- **`bw_to_data.py`**:
    - **Concept**: Core logic to extract data from BigWigs and Fasta based on indices. Uses caching (`npy`).
    - **Functions**: `get_wg_filtered_data`, `idx_to_filtered_data`, `get_data_by_idx`.
- **`utils/seq.py`**:
    - **Concept**: Fasta reading utilities. **Crucial**: Uses **AGCT** order (0,1,2,3).
    - **Functions**: `get_chr_seq`, `seq_to_idx`, `seq_to_onehot`.
- **`utils/data_bw.py`**:
    - **Concept**: BigWig reading utilities.
    - **Functions**: `get_binned_signal` (supports max/mean binning).

### Input/Output Dimensions
- **Input**: 2048bp.
- **Output**: 1024bp (256 bins of 4bp).
- **Logic**: Input window includes margins. Model predicts full 2048bp coverage (512 bins), but evaluation/loss trims to center 1024bp.

### Negative Regions
- **Strategy**: ASAP does not explicitly mention "GC-matched negatives" in the provided code.
- **WGDataset**: Trains on the whole genome (sliding window), so it implicitly includes non-peak "negative" regions naturally.
- **PeakDataset**: Trains on peaks only.

### Mappability Handling
- **Objective**: Predict mappability of the input sequence as an auxiliary task to improve robustness or learn bias.
- **Architecture**:
    - Uses `UnmapPredictor` module (`Linear(channels_in, 1) -> Sigmoid`).
    - Integrated into models (e.g., `ConvNeXtCNN`) at an intermediate layer (typically after the stem).
    - Output `u` represents the predicted mappability probability.
- **Data Loading**:
    - Accepts an `unmappable_bed_file` (BED file of unmappable regions).
    - **Threshold Filtering**:
        - Iterates over candidate windows (`start`, `end`).
        - Calculates the total overlap length with unmappable regions.
        - If `overlap_length > threshold * window_size` (default `threshold=0.35`), the window is discarded.
        - This ensures training data has sufficient mappability (at least 65% mappable).
    - **Mask Generation**:
        - For kept windows, generates a binary mappability mask `m` (1=mappable, 0=unmappable) of length `window_size`.
        - Unmappable regions are set to 0.
    - This mask `m` is passed to the trainer.
- **Training**:
    - **Loss**: `MSELoss` between predicted mappability `u` and ground truth mask `m`.
    - **Total Loss**: `base_loss + unmap_loss`.
    - **Alignment Note**: The code in `trainer.py` slices the ground truth mask `m_i[:, :m_len]` to match the model output length. Since the model output `u` is often downsampled (e.g., by `MaxPool1d(2)` in the stem), this slicing might imply a spatial mismatch (first half of ground truth vs strided prediction) unless `m_i` is pre-processed or the downsampling is handled otherwise. This is a specific implementation detail to be aware of.

---

## 2. BPNet-lite (Infrastructure)

**Location**: `repos/bpnet-lite`

### File Analysis

#### `bpnetlite/`
- **`bpnet.py`**:
    - **Concept**: Defines the `BPNet` architecture and Wrappers.
    - **Functions**: `BPNet`, `ControlWrapper`, `ProfileWrapper`, `CountWrapper`.
- **`losses.py`**:
    - **Concept**: Loss functions.
    - **Functions**: `MNLLLoss` (Multinomial NLL), `log1pMSELoss`, `_mixture_loss` (weighted sum).
- **`io.py`**:
    - **Concept**: Data loading utilities.
    - **Functions**: `PeakNegativeSampler` (samples peaks/negatives from tensors), `PeakGenerator` (orchestrates loading).
- **`chrombpnet.py`**:
    - **Concept**: Defines `ChromBPNet` class (Bias + Accessibility models).
- **`performance.py`**:
    - **Concept**: Metrics calculation.
    - **Functions**: `pearson_corr`, `calculate_performance_measures`.

### Input/Output Dimensions
- **Standard**: Input 2114bp, Output 1000bp. (Configurable).
- **Compatibility**: Can be configured to match ASAP (2048/1024).

### Negative Regions
- **Strategy**: Expects `negatives` (coordinates) to be provided as input to `PeakGenerator` or `PeakNegativeSampler`.
- **Usage**: Samples from these negatives at a rate defined by `negative_ratio` (default 0.1) to mix with peaks during training.
- **Origin**: Assumes these negatives are pre-generated (e.g., GC-matched). Does not generate them itself.

---

## 3. ChromBPNet-pytorch (Training Algorithms)

**Location**: `repos/chrombpnet-pytorch`

### File Analysis

#### `chrombpnet/`
- **`chrombpnet.py`**:
    - **Concept**: Defines `ChromBPNet` module.
    - **Functions**: `ChromBPNet` class.
- **`model_wrappers.py`**:
    - **Concept**: PyTorch Lightning wrappers for models. Handles training step and optimization.
    - **Functions**: `ChromBPNetWrapper`, `BPNetWrapper`, `multinomial_nll`.
- **`dataset.py`**:
    - **Concept**: PyTorch Lightning DataModule. Manages dataloaders.
    - **Functions**: `DataModule`, `ChromBPNetDataset`, `negative_dataloader`.
- **`data_utils.py`**:
    - **Concept**: Low-level data utilities.
    - **Functions**: `dna_to_one_hot` (**ACGT** order), `get_seq`, `get_cts`, `load_data` (reads peaks/nonpeaks).
- **`main.py`**:
    - **Concept**: CLI entry point.
    - **Functions**: `train`, `predict`, `interpret`.

### Input/Output Dimensions
- **Standard**: Input 2114bp, Output 1000bp.
- **Structure**: Two-tower (Bias + Accessibility).

### Negative Regions
- **Strategy**: Explicitly handles "negatives" (non-peaks).
- **Configuration**: `negative_sampling_ratio` in `DataConfig`.
- **Bias Training**: The bias model is often trained on these negatives (or adjusted using them via `adjust_bias_model_logcounts`).
- **Generation**: Expects a `negatives.bed` file. It assumes these are GC-matched.

---

## 4. GenVarLoader (Alternative IO)

**Location**: `repos/GenVarLoader`

### File Analysis

#### `python/genvarloader/`
- **`__init__.py`**: Exports main classes (`BigWigs`, `RefDataset`).
- **`_bigwig.py`**:
    - **Concept**: Efficient BigWig reader.
    - **Functions**: `BigWigs.read` (returns numpy arrays), `intervals`.
- **`_dataset/_reference.py`**:
    - **Concept**: Reference genome dataset.
    - **Functions**: `RefDataset` (extracts sequences from BED), `Reference` (holds genome in memory/mmap).
- **`_dataset/_impl.py`**:
    - **Concept**: Core `Dataset` implementation.
    - **Functions**: `Dataset.open`, `Dataset.to_dataloader`.

### Input/Output Dimensions
- **Flexible**: Can handle any input/output length specified by user.

### Negative Regions
- **Strategy**: Agnostic. It just loads regions provided in a BED file/DataFrame. If you provide a BED file with negatives, it loads them.

### Usage in Cerberus
- Considered as a primary backend but found too heavy for initial requirements.
- Retained as an alternative/future backend, especially if VCF support is needed.

### One-Hot Encoding
- **Native Output**: `GenVarLoader` (specifically `RefDataset`) returns sequences as **S1 numpy arrays** (bytes), e.g., `[b'A', b'C', b'G', b'T']`.
- **Expectation**: It does **not** perform one-hot encoding internally.
- **Solution**: Use **SeqPro** (see below). `seqpro.ohe` can convert these bytes to one-hot encoded arrays.

---

## 5. Tangermeme (Compatibility)

**Location**: `repos/tangermeme`

### File Analysis

#### `tangermeme/`
- **`io.py`**:
    - **Concept**: Input/Output utilities.
    - **Functions**: `extract_loci` (loads seq/signal to tensors), `one_hot_to_fasta`.
- **`predict.py`**:
    - **Concept**: Prediction utility.
    - **Functions**: `predict` (batched prediction for models).
- **`deep_lift_shap.py`**:
    - **Concept**: DeepLIFT/SHAP implementation.
- **`utils.py`**:
    - **Concept**: General utilities.
    - **Functions**: `one_hot_encode`.

### Input/Output Dimensions
- **Flexible**: Works with any model signature `model(X, *args)`.

---

## 6. SeqPro (Utility)

**Location**: `repos/SeqPro`

### Overview
`SeqPro` provides optimized sequence processing utilities that complement `GenVarLoader`. It handles encoding, padding, and augmentation.

### File Analysis

#### `python/seqpro/`
- **`_encoders.py`**:
    - **Concept**: Core encoding functions.
    - **Functions**:
        - `ohe(seqs, alphabet)`: One-hot encodes sequences.
        - `pad_seqs`: Pads or truncates sequences.
        - `tokenize`: Converts chars to integer tokens.
- **`_modifiers.py`**:
    - **Concept**: Sequence modifiers/augmentations.
    - **Functions**:
        - `reverse_complement`: Fast RC.
        - `jitter`: Randomly shifts sequence window.
        - `k_shuffle`: k-mer shuffling (for dinucleotide shuffle).
- **`alphabets/_alphabets.py`**:
    - **Concept**: Alphabet definitions.
    - **Functions**: `NucleotideAlphabet`.
        - Default `DNA` is **ACGT**.
        - Can instantiate custom alphabet `NucleotideAlphabet("AGCT", "TCGA")` to match ASAP.

### Synergy with GenVarLoader
- **One-Hot Encoding**: Pass the S1 byte array from `GenVarLoader` to `seqpro.ohe`.
- **Augmentation**: Use `seqpro.jitter` or `seqpro.reverse_complement` within the dataloader transform.

---

## Summary of Key Differences

| Feature | ASAP | ChromBPNet / BPNet-lite | GenVarLoader |
| :--- | :--- | :--- | :--- |
| **Input Size** | 2048 bp | 2114 bp | Flexible |
| **Output Size** | 1024 bp (256 bins) | 1000 bp | Flexible |
| **One-hot Order** | **AGCT** (0,1,2,3) | **ACGT** (0,1,2,3) | **S1 Bytes** (Needs conversion) |
| **Binning** | Max (default) or Mean | No binning (bp resolution) | Raw values |
| **Negatives** | Implicit (Whole Genome) | Explicit (GC-matched file) | Agnostic (User provided) |
| **IO Strategy** | Disk-based (pyBigWig) | In-memory tensors | Disk-based (pyBigWig optimized) |

## Recommendation for Cerberus

1.  **Architecture**: Replicate `ConvNeXtCNN` from ASAP.
2.  **Dimensions**: Use **2048bp Input / 1024bp Output** to match ASAP.
3.  **Encoding**: **Crucial Decision**: ASAP uses AGCT. Standard is ACGT. If loading ASAP weights, must use AGCT. If training from scratch, prefer ACGT (standard).
    - *Decision*: Since goal is "re-implement", if we want to reproduce results exactly or use pre-trained weights, use AGCT. Otherwise, stick to ACGT for compatibility with ChromBPNet/Tangermeme defaults.
4.  **IO**: Use `pyfaidx` and `pybigtools`.
    - Create a `SequenceExtractor` using `pyfaidx` that outputs One-hot (ACGT or AGCT).
    - Create a `SignalExtractor` using `pybigtools` that bins the BigWig signal (4bp max/mean) to match ASAP target.
5.  **Negatives**:
    - Implement `PeakNegativeSampler` logic (from bpnet-lite) backed by lazy loading via `pyfaidx`/`pybigtools`.
    - Allow passing a `negatives.bed` (GC-matched) for bias training/regularization.
