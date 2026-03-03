# BPReveal Model Interpretation Review

Review of interpretation functionality in [BPReveal](https://github.com/mmtrebuchet/bpreveal) (v5.2.2) and its compatibility with cerberus models.

## 1. Overview of Interpretation Functionality

BPReveal implements a complete model interpretation pipeline consisting of three main capabilities:

1. **Attribution scoring** -- DeepSHAP and In-Silico Mutagenesis (ISM)
2. **Motif discovery and scanning** -- TF-MoDISco integration and CWM-based motif scanning
3. **Visualization** -- sequence logos, PISA heatmaps, profile tracks

All interpretation code operates on TensorFlow/Keras models. There is **no use of captum, shap (the pip package), or any other third-party interpretation library**. BPReveal maintains its own forked implementation of DeepSHAP (derived from the original `shap` package, MIT-licensed) in `src/internal/shap.py`, modified for TensorFlow 2.16+ compatibility.

## 2. Attribution Methods

### 2.1 DeepSHAP (`TFDeepExplainer`)

**File:** `src/internal/shap.py` -- class `TFDeepExplainer`

A custom gradient-based SHAP implementation that overrides TensorFlow's gradient registry to implement DeepLIFT-style attribution through the computational graph. Key details:

- Operates by splitting each forward pass into (sample, reference) pairs, then computing modified gradients where nonlinear operations are handled with the DeepLIFT rescale rule: `attribution = grad * (x_out - r_out) / (x_in - r_in)`.
- Registers custom gradient handlers for ~40 TensorFlow ops (see `op_handlers` dict):
  - **Linear ops** (passthrough): `Identity`, `Add`, `Conv2D`, `BiasAdd`, `Reshape`, `Pad`, `ConcatV2`, `BatchToSpaceND`, `SpaceToBatchND`, etc.
  - **Nonlinear 1D ops**: `Relu`, `Sigmoid`, `Tanh`, `Softplus`, `Exp`, `Log`, `Elu`, `Selu`, `ClipByValue`, `Square`, `Rsqrt`.
  - **Nonlinear 2D ops**: `Mul`, `RealDiv`, `MatMul`, `SquaredDifference`, `Minimum`, `Maximum`.
  - **Special handlers**: `Softmax` (decomposed into exp/sum/div), `MaxPool`, `GatherV2`/`ResourceGather`.
  - **Dependence breakers**: `Shape`, `RandomUniform`, `ZerosLike`.
- Reference sequences are generated via k-mer-preserving dinucleotide shuffles (`ushuffle.py`).
- Supports a configurable number of shuffled references (recommended: 20).

**Hypothetical contribution scores** are supported via a custom `combine_mult_and_diffref` function that computes scores for all 4 possible bases at each position, not just the observed base. These are required for feeding into TF-MoDISco.

**Standard contribution scores** use the normal SHAP formula: `mult * (original - background)`, producing scores only for the observed base.

### 2.2 In-Silico Mutagenesis (`ISMDeepExplainer`)

**File:** `src/internal/shap.py` -- class `ISMDeepExplainer`

A scanning mutagenesis approach:

- Slides a window of configurable width across the input sequence.
- At each position, a user-provided `shuffler` function generates mutated subsequences.
- The model is run on each mutated sequence and the output difference from the reference is recorded.
- Importance is: `(reference_output - mean_mutated_output) * one_hot_input`.
- Batched internally for GPU efficiency (configurable batch size).
- Does not require gradient computation -- purely forward-pass based.
- No hypothetical scores (zeros where input is zero).

### 2.3 Interpretation Orchestration (`InterpRunner`)

**File:** `src/internal/interpretUtils.py` -- class `InterpRunner`

A multi-process pipeline with three stages:

```
Generator --> Batcher(s) --> Saver(s)
   (reads        (runs SHAP      (writes
    sequences)    or ISM)          HDF5)
```

- **Generators**: `ListGenerator` (in-memory strings), `FastaGenerator` (FASTA file), `FlatBedGenerator` (BED + genome FASTA, one query per region), `PisaBedGenerator` (BED + genome, one query per base).
- **Batchers**: `_ShapBatcher` (DeepSHAP backend) and `_IsmBatcher` (ISM backend). Each loads a separate copy of the model.
- **Savers**: `FlatListSaver` (in-memory via shared memory), `FlatH5Saver` (chunked gzip HDF5, full-width), `PisaH5Saver` (HDF5, receptive-field width).
- Supports parallel batchers (GPU memory fraction is configurable).
- Data flow uses `multiprocessing.Process` with `CrashQueue` (queue with error propagation).

### 2.4 Metrics (What Gets Explained)

Attribution is always computed with respect to a scalar metric. BPReveal defines:

- **Profile metric** (`interpretFlat.profileMetric`): Weighted sum of softmaxed, mean-normalized logits over selected tasks. Measures profile "spikiness". The mean-normalization uses `stop_gradient` to prevent SHAP from backpropagating through the softmax weights.
- **Counts metric** (`interpretFlat.countsMetric`): Raw log-counts scalar from the counts head.
- **PISA metric** (`interpretPisa.pisaMetric`): Logit value at the leftmost output position for a single task. Used for per-base pairwise interaction scoring.

### 2.5 Entry Points

| Script | Description |
|--------|-------------|
| `interpretFlat` | Standard BPNet-style attribution (profile + counts). Outputs `hyp_scores`, `input_seqs`, `input_predictions` to HDF5. |
| `interpretPisa` | PISA (Pairwise Interaction Score Analysis). Per-base SHAP scores showing how each input base contributes to a single output position. Outputs `shap`, `sequence`, `shuffle_predictions`. |

## 3. Motif Discovery and Scanning

### 3.1 TF-MoDISco Integration

**Dependency:** `modisco >= 2.4.0` (modiscolite)

BPReveal produces the inputs required by TF-MoDISco:

- `shapToNumpy` converts interpretation HDF5 to NumPy arrays in the format expected by modiscolite: transposed from `(N, L, 4)` to `(N, 4, L)`.
- Hypothetical contribution scores (from `interpretFlat` with `useHypotheticalContribs=True`) are the recommended input.

### 3.2 CWM-Based Motif Scanning

**File:** `src/motifUtils.py` (66 KB)

A complete motif scanning pipeline operating on contribution score HDF5 files:

- **Pattern representation**: `Pattern` and `MiniPattern` classes wrapping CWM (Contribution Weight Matrix), PPM, PSSM, and trimming metadata from MoDISco output.
- **Scoring**: `slidingDotproduct` -- convolution-based scanning using `scipy.signal.correlate`.
- **Quantile thresholding**: `arrayQuantileMap` maps scores to quantiles of the original MoDISco seqlet distribution.
- **Three-cutoff filtering**: sequence match (PSSM), contribution match (CWM Jaccard), and contribution magnitude (L1 norm).
- **Multi-threaded scanning**: `scanPatterns` uses producer-consumer architecture with configurable thread count (scales to 70+ cores).
- **Jaccard similarity**: C-compiled module (`jaccard.py`) for fast CWM similarity computation.
- **Output**: TSV file with hit coordinates, strand, scores, and pattern assignments.

### 3.3 Motif Pipeline Scripts

| Script | Description |
|--------|-------------|
| `motifSeqletCutoffs` | Determines score thresholds from MoDISco seqlets |
| `motifAddQuantiles` | Adds quantile thresholds to pattern definitions |
| `motifScan` | Scans contribution scores for motif instances |

## 4. Visualization

### 4.1 Sequence Logos

**File:** `src/internal/plotUtils.py`

- `plotLogo()`: Standard DNA sequence logo rendering using matplotlib, with letter heights proportional to importance scores.

### 4.2 PISA Plots

**File:** `src/plotting.py` (44 KB)

- PISA heatmap generation showing pairwise base interactions.
- Profile track plotting alongside PISA heatmaps.
- Annotation overlays (motif positions, genomic features).
- Custom colormaps for clipped/unclipped PISA values.

### 4.3 Score Export

| Script | Description |
|--------|-------------|
| `shapToBigwig` | Converts HDF5 attribution scores to BigWig format for genome browser viewing. Projects hypothetical scores to actual contributions by multiplying by one-hot sequence. |
| `shapToNumpy` | Converts HDF5 to NumPy arrays for MoDISco input. |
| `predictToBigwig` | Converts model predictions to BigWig format. |

## 5. External Library Dependencies

### 5.1 BPReveal Depends On

| Library | Purpose | Used In |
|---------|---------|---------|
| **TensorFlow >= 2.20** | Model execution, gradient computation, DeepSHAP implementation | `shap.py`, all interpretation |
| **tf-keras >= 2.20** | Legacy Keras model loading | `shap.py`, `interpretUtils.py` |
| **modisco >= 2.4.0** | Motif discovery from attribution scores | post-interpretation pipeline |
| **h5py** | Attribution score I/O | all savers, `shapToBigwig`, `shapToNumpy` |
| **pyBigWig** | BigWig export | `shapToBigwig`, `predictToBigwig` |
| **pysam** | FASTA genome access | generators, savers |
| **pybedtools** | BED file parsing | generators |
| **scipy** | Convolution for motif scanning | `motifUtils.py` |
| **matplotlib** | Visualization | `plotUtils.py`, `plotting.py` |
| **numpy** | Array operations throughout | everywhere |
| **tqdm** | Progress bars | interpretation pipeline |

### 5.2 NOT Used

- **captum** -- not present anywhere in the codebase
- **shap** (pip package) -- not imported; BPReveal has its own forked/modified implementation
- **LIME**, **Grad-CAM**, or any other third-party interpretation library

## 6. Compatibility with Cerberus Models

### 6.1 Fundamental Incompatibilities

The BPReveal interpretation code **cannot be used directly** with cerberus models due to several fundamental differences:

| Aspect | BPReveal | Cerberus |
|--------|----------|----------|
| **Framework** | TensorFlow 2.x / Keras | PyTorch |
| **Input layout** | `(B, L, 4)` -- length-first | `(B, 4, L)` -- channels-first |
| **Model type** | `keras.Model` with `.input` / `.outputs` properties | `nn.Module` with `forward()` |
| **Gradient mechanism** | Custom TF gradient registry override | Standard `torch.autograd` |
| **Output structure** | List of tensors `[profile_head0, ..., counts_head0, ...]` | Dataclasses: `ProfileCountOutput`, `ProfileLogRates` |
| **Model loading** | `utils.loadModel()` -> Keras SavedModel | `CerberusModule` -> PyTorch Lightning checkpoint |

### 6.2 What Would Need Reimplementation

#### Attribution Methods

**DeepSHAP**: The entire `TFDeepExplainer` is TensorFlow-specific (gradient registry hacking, `tf.GradientTape`, TF op handlers). For PyTorch cerberus models, options are:

1. **captum** (`pip install captum`): Provides `DeepLift`, `DeepLiftShap`, `IntegratedGradients`, `GradientShap`, and other methods. These work natively with `nn.Module` and `torch.autograd`. This is the recommended path.
2. **shap** (pip package): The `DeepExplainer` in the `shap` package supports PyTorch models directly, though it has known compatibility issues with newer PyTorch versions and complex architectures.
3. **Custom implementation**: Port the BPReveal SHAP logic to PyTorch. Feasible but substantial effort -- the op handler registry approach does not translate to PyTorch's dynamic graph.

**ISM**: The `ISMDeepExplainer` is simpler and more framework-agnostic. The core algorithm (slide window, mutate, measure output change) only requires forward passes. A PyTorch port would be straightforward:
- Replace `self.model(np.array([sample]))` with `model(torch.tensor(sample).unsqueeze(0))`.
- Handle the `(B, 4, L)` vs `(B, L, 4)` transpose.
- The shuffler interface is already framework-independent.

**Gradient x Input (simplest baseline)**: Cerberus models already support input gradient computation (verified in `tests/test_bpnet_gradient_training.py`). A simple `input.grad * input` saliency map requires no external library.

#### Metrics

The BPReveal metric functions (`profileMetric`, `countsMetric`, `pisaMetric`) are defined using Keras/TF ops but encode simple mathematical operations. Equivalent PyTorch functions for cerberus:

```python
# Profile metric (cerberus equivalent)
def profile_metric(output: ProfileCountOutput) -> torch.Tensor:
    logits = output.logits  # (B, C, L)
    flat = logits.reshape(logits.shape[0], -1)  # (B, C*L)
    centered = flat - flat.mean(dim=1, keepdim=True)
    weights = torch.softmax(centered.detach(), dim=1)
    return (weights * centered).sum(dim=1)  # (B,)

# Counts metric
def counts_metric(output: ProfileCountOutput) -> torch.Tensor:
    return output.log_counts.sum(dim=1)  # (B,)

# PISA metric
def pisa_metric(output: ProfileCountOutput, task_idx: int = 0) -> torch.Tensor:
    return output.logits[:, task_idx, 0]  # leftmost position
```

#### Hypothetical Contribution Scores

The `combineMultAndDiffref` function for hypothetical contributions is pure numpy and is conceptually framework-independent. With captum's `DeepLiftShap`, custom attribution rules can be specified, or the hypothetical score computation can be applied as a post-processing step on per-base attributions.

### 6.3 Reusable Components (Framework-Independent)

The following BPReveal components operate on numpy arrays and HDF5 files, making them **directly reusable** with cerberus output:

| Component | File | Notes |
|-----------|------|-------|
| Motif scanning | `motifUtils.py` | Operates on `(N, L, 4)` numpy arrays of contribution scores. Cerberus output would need transposing from `(N, 4, L)`. |
| Motif cutoff analysis | `motifSeqletCutoffs` | Pure numpy/HDF5 |
| SHAP-to-BigWig export | `shapToBigwig.py` | Reads HDF5, writes BigWig. Reusable if cerberus produces HDF5 in the same schema. |
| SHAP-to-NumPy export | `shapToNumpy.py` | Trivial format conversion |
| Sequence logo plotting | `plotUtils.py` | Matplotlib-based, accepts numpy arrays |
| PISA plotting | `plotting.py` | Matplotlib-based |
| Dinucleotide shuffling | `ushuffle.py` | Pure C/numpy, framework-independent |
| Jaccard similarity | `jaccard.py` | Compiled C module |
| TF-MoDISco | `modisco` package | Accepts numpy arrays of attribution scores |

### 6.4 Cerberus Model-Specific Considerations

#### Multi-Architecture Support

Cerberus has 6 model architectures. Compatibility with different interpretation methods:

| Model | Architecture | captum DeepLiftShap | captum IntegratedGradients | ISM | Gradient x Input |
|-------|-------------|--------------------|-----------------------------|-----|------------------|
| **BPNet** | Dilated residual CNN | Yes | Yes | Yes | Yes |
| **Pomeranian** | ConvNeXtV2 + PGC | Likely (needs testing with GRN) | Yes | Yes | Yes |
| **GlobalProfileCNN** | Conv + MaxPool + Dense | Yes | Yes | Yes | Yes |
| **ConvNeXtDCNN** | ConvNeXtV2 blocks | Likely | Yes | Yes | Yes |

Notes:
- **IntegratedGradients** and **Gradient x Input** are the most universally compatible since they only require `torch.autograd.grad` on the model's forward pass.
- **DeepLiftShap** via captum may have issues with non-standard layers (GRN in Pomeranian/ConvNeXtDCNN) because DeepLIFT requires specific propagation rules for each layer type.
- **ISM** is architecture-agnostic (only uses forward passes).

#### Output Format Differences

BPReveal models output a flat list: `[profile_0, profile_1, ..., counts_0, counts_1, ...]`.
Cerberus models output structured dataclasses: `ProfileCountOutput(logits, log_counts)` or `ProfileLogRates(log_rates)`.

For interpretation, a wrapper function is needed to extract the relevant scalar from cerberus output and make it differentiable:

```python
def make_profile_metric(model, task_indices=None):
    """Wraps a cerberus model to return a scalar for attribution."""
    def wrapped(x):
        output = model(x)
        logits = output.logits  # (B, C, L)
        if task_indices is not None:
            logits = logits[:, task_indices, :]
        flat = logits.reshape(logits.shape[0], -1)
        centered = flat - flat.mean(dim=1, keepdim=True)
        weights = torch.softmax(centered.detach(), dim=1)
        return (weights * centered).sum(dim=1)
    return wrapped
```

#### Encoding Axis Transposition

BPReveal uses `(B, L, 4)` layout; cerberus uses `(B, 4, L)`. All attribution scores from cerberus would need transposing before passing to BPReveal's downstream tools (MoDISco, motif scanning, BigWig export).

### 6.5 `ProfileLogRates` vs `ProfileCountOutput` Models

Two cerberus model families produce different output types:

- **`ProfileCountOutput`** (BPNet, Pomeranian): Have separate profile logits and log-counts. The BPReveal interpretation metrics (profile spikiness, counts) map directly.
- **`ProfileLogRates`** (GlobalProfileCNN, ConvNeXtDCNN): Only have log-rates, no separate counts head. The profile metric would need adaptation (e.g., sum of log-rates as the scalar, or per-position log-rate for PISA-like analysis).

## 7. Recommended Integration Strategy

### Phase 1: Basic Attribution (captum)

Use captum's `IntegratedGradients` as the primary attribution method:

- Works with all 6 cerberus architectures without modification.
- Provides both per-base and hypothetical contribution scores.
- Well-maintained, PyTorch-native, extensive documentation.

Add `GradientShap` (captum's implementation) as an alternative that more closely matches BPReveal's DeepSHAP approach.

### Phase 2: ISM

Port the ISM algorithm from BPReveal. This is a ~100-line implementation once the shuffler interface is adapted. The core loop is framework-independent; only the model forward call changes.

### Phase 3: Downstream Tools

Produce HDF5 output in BPReveal-compatible format (or a cerberus-native format with a conversion script). This enables reuse of:
- TF-MoDISco for motif discovery
- `motifScan` for CWM-based motif scanning
- `shapToBigwig` for genome browser visualization
- PISA plotting utilities

### Phase 4: PISA

Implement PISA for cerberus models. The per-base interpretation approach is a direct adaptation of `interpretPisa` using captum attribution on a single-position output metric.
