# Variant Effect Prediction Support for Cerberus

## 1. Executive Summary

This document analyzes how to extend cerberus to support prediction of variant effects on model outputs, with a focus on screening GWAS variants. It covers two distinct use cases: (A) **prediction-time variant scoring** (comparing model outputs on reference vs. alternative allele sequences) and (B) **training-time haplotype augmentation** (training models on personalized genomes with variants incorporated). The analysis is based on a thorough review of the cerberus codebase and seven external repositories: **kipoiseq**, **GenVarLoader**, **tangermeme**, **bilby**, **gReLU**, **SeqPro**, and **bpreveal**.

**Recommendation:** Implement variant scoring as a self-contained prediction-time tool (`tools/score_variants.py`) with a small supporting module (`src/cerberus/variants.py`). Use **kipoiseq** patterns for VCF parsing and ref/alt sequence construction (either as a lightweight dependency or by lifting the ~200 lines of core logic from its `VariantSeqExtractor`). For advanced analysis functions (batched ISM, DeepLIFT/SHAP), lift the specific needed code from tangermeme rather than taking it as a dependency. For training on haplotypes (Phase 3), **GenVarLoader** is the only viable option and integrates behind the existing `BaseSequenceExtractor` protocol.

---

## 2. Current State of Cerberus

### 2.1 Existing ISM capability

The only existing variant-related code is `tools/plot_biasnet_ism.py`, which performs naive in-silico mutagenesis on BiasNet:

- **Scope:** BiasNet only (or BiasNet extracted from Dalmatian). Not usable with Pomeranian, BPNet, or full Dalmatian.
- **Algorithm:** Triple-nested loop: `for seq -> for pos -> for nuc`. Each mutation is a separate forward pass on a single sequence. For 1000 sequences with a 31bp window, this is ~93,000 individual forward passes with no batching.
- **Output:** Average delta-logits at the center output position only. No profile-level or count-level effect sizes.
- **Variants:** No VCF support. Only systematic single-nucleotide scanning at fixed window positions.

### 2.2 Prediction infrastructure

Cerberus has a mature prediction pipeline that variant scoring can build on:

| Component | Location | Relevance |
|-----------|----------|-----------|
| `ModelEnsemble` | `src/cerberus/model_ensemble.py` | Loads fold models, routes intervals to correct folds, ensembles predictions. Direct reuse for variant scoring. |
| `predict_intervals_batched()` | `model_ensemble.py:287-351` | Batched inference generator. Could be adapted for ref/alt paired prediction. |
| `create_eval_dataset()` | `predict_misc.py:25-46` | Creates inference dataset with no augmentation. Provides extractors for ref sequences. |
| `SequenceExtractor` / `InMemorySequenceExtractor` | `sequence.py:100-218` | Extract one-hot reference sequences from FASTA. The `BaseSequenceExtractor` protocol (`sequence.py:89-97`) is the extension point for variant-aware extractors. |
| `_reconstruct_linear_signal()` | `predict_bigwig.py` | Converts logits + log_counts to linear signal. Needed for meaningful variant effect sizes. |
| Output types | `output.py` | `ProfileCountOutput`, `FactorizedProfileCountOutput`, `ProfileLogRates` -- all structured dataclasses. |

### 2.3 Model output structure

All cerberus models return structured `ModelOutput` dataclasses, not raw tensors. This is relevant when considering external tool compatibility:

- **ProfileCountOutput:** `logits (B, C, L)` + `log_counts (B, C)` -- profile shape and total count predicted separately.
- **FactorizedProfileCountOutput:** Adds decomposed bias/signal components from Dalmatian.
- **ProfileLogRates:** `log_rates (B, C, L)` -- Poisson-style direct rate prediction.

Biologically meaningful signal requires reconstruction: `softmax(logits) * exp(log_counts)` for ProfileCountOutput, or `exp(log_rates)` for ProfileLogRates. Variant effect scores must be computed on reconstructed signal, not raw logits, to capture both shape and magnitude changes.

---

## 3. External Repository Analysis

### 3.1 kipoiseq -- Lightweight variant sequence extraction

**Repository:** `github.com/kipoi/kipoiseq` (now at `../s2f-models/repos/kipoiseq`)
**Publication:** Kipoi ecosystem, Nature Biotechnology 2019. MIT license.
**Status:** Last commit September 2024. Last release v0.7.1 (October 2021). Sporadic but responsive maintenance.

**Core capability:** VCF-driven ref/alt sequence construction from FASTA + VCF. This is exactly what prediction-time variant scoring needs.

**Key classes:**

- **`VariantSeqExtractor`** (`kipoiseq/extractors/vcf_seq.py:60-303`): The workhorse. Given a genomic interval, a list of variants, and an anchor position, returns the alternative sequence as a string. Key features:
  - `fixed_len=True` mode: output sequence has the same length as the input interval, handling indels by extending the FASTA fetch and trimming from the distal flank. This solves the constant-input-length problem for cerberus models.
  - Splits variants at the anchor point into upstream/downstream, processes them outward from the anchor.
  - Handles overlapping variants by splitting at boundaries.
  - Strand-aware: reverse complements if `interval.strand == "-"`.
  - Padding with 'N' when sequence cannot extend to fixed length (`is_padding=True`).

- **`SingleVariantVCFSeqExtractor`** (`vcf_seq.py:328-341`): Convenience class. Given FASTA + VCF, yields one alt sequence per variant overlapping an interval. Wraps `VariantSeqExtractor`.

- **`SingleSeqVCFSeqExtractor`** (`vcf_seq.py:344-355`): Yields a single sequence with all overlapping variants applied (haplotype-like).

- **`MultiSampleVCF`** (`kipoiseq/extractors/vcf.py`): VCF parser backed by **cyvcf2**. `fetch_variants(interval, sample_id)` yields `Variant` objects for a genomic region.

- **`Variant` dataclass** (`kipoiseq/dataclasses.py:17-149`): `chrom`, `pos` (1-based), `ref`, `alt`, plus `id`, `qual`, `filter`, `info`. Factory methods: `from_cyvcf()`, `from_str()`. Immutable core fields.

- **`Interval` dataclass** (`kipoiseq/dataclasses.py:152-375`): `chrom`, `start` (0-based), `end`, `strand`. Methods: `shift()`, `resize()`, `trim()`, `center()`, `slop()`, `truncate()`.

**Dependencies:** cyvcf2, pyfaidx, pyranges, numpy, pandas. Moderate weight -- cyvcf2 is a C extension but widely available via conda/pip. The core `VariantSeqExtractor` logic itself only depends on pyfaidx.

**Suitability for prediction:** Excellent. The `VariantSeqExtractor` with `fixed_len=True` is purpose-built for exactly this use case -- construct constant-length alt sequences from VCF variants for model scoring. The anchor-based bidirectional extension handles indels cleanly. Returns strings that cerberus's `encode_dna()` can one-hot encode directly.

**Suitability for training:** Limited. kipoiseq extracts one variant (or one haplotype) at a time per interval. It does not support:
- Batch/parallel haplotype reconstruction across many samples.
- Personalized diploid genome training (iterating over sample x haplotype x region).
- Track (BigWig) realignment for indels.
- Efficient sparse genotype storage.

kipoiseq is a **prediction-time tool**, not a training-time data loader.

### 3.2 GenVarLoader -- Training-time haplotype data loading

**Repository:** `github.com/mcvickerlab/GenVarLoader` (at `../s2f-models/repos/GenVarLoader`)
**Publication:** bioRxiv preprint, January 2025 (DOI: 10.1101/2025.01.15.633240). **Not yet peer-reviewed.** MIT license.
**Authors:** David Laub (UCSD, primary developer), Aaron Ho (Salk), Graham McVicker (Salk), Hannah Carter (UCSD).
**Status:** Actively maintained. v0.21.4 released March 26, 2026. 90+ releases. Primarily solo development by David Laub.

**Core capability:** On-the-fly reconstruction of personalized haplotype sequences from reference genome + sparse variant indices. Designed for training sequence models on genetic variation at scale.

**Key architecture:**
- **Numba JIT-compiled** `reconstruct_haplotype_from_sparse()` -- applies variants to reference sequence at query time. Parallelized over regions and ploidy.
- **Sparse genotype storage:** Only ALT allele indices stored per sample. Reference is implicit. 2,032x storage reduction vs. bcftools consensus (6.3 TB vs. 3.1 GB for 1000 Genomes).
- **Track realignment:** BigWig signal tracks re-aligned to haplotype coordinates when indels change sequence length via `shift_and_realign_tracks_sparse()`.
- **Performance:** 300-1,000x faster than reading pre-generated FASTA; 190-450x faster than pyBigWig for tracks. Throughput exceeds maximum GPU input bandwidth.

**Key API:**
- `Dataset.open(path, reference, jitter, min_af, max_af)` -- lazy dataset with AF filtering.
- `dataset.with_seqs("haplotypes" | "annotated" | "reference" | "variants")` -- select sequence type.
- `dataset.with_len(512)` -- fixed-length output.
- Indexing: `dataset[region_idx, sample_idx]` -- returns reconstructed haplotype bytes.
- `DatasetWithSites` -- post-hoc application of site-only variants (e.g., ClinVar) to existing haplotypes.

**Adoption:** Used by Nucleotide Transformer (Nature Methods 2024), Variformer (Genome Biology 2025), CLIPNET, and several other preprints. Still primarily within the McVicker/Carter lab ecosystem (~28 GitHub stars).

**Dependencies:** Heavy -- numba, genoray (VCF/PGEN parsing), seqpro, awkward arrays, polars, hirola, pyBigWig, pysam. All available via pip/conda but a significant dependency tree.

**Suitability for training:** This is what it was built for. The only tool that solves personalized genome training at scale. Key advantages:
- Batch parallel haplotype reconstruction across (regions x samples x ploidy).
- Memory-mapped sparse genotype format -- no need to materialize haplotype FASTAs.
- Track realignment for indels (the hardest part of haplotype training).
- Allele frequency filtering for population-stratified training.
- PyTorch `DataLoader` compatible via `to_dataloader()`.

**Suitability for prediction (variant scoring):** Possible but overkill. GenVarLoader is designed for iterating over (region, sample) pairs during training, not for "score this list of VCF variants against a model." Using it for variant scoring would require:
- Pre-building a GVL dataset from the VCF (disk I/O, preprocessing step).
- Extracting both reference and haplotype sequences for comparison.
- The `DatasetWithSites` class handles post-hoc single-variant application, but it's SNP-only and requires an existing GVL dataset.

For prediction-time variant scoring, kipoiseq's `VariantSeqExtractor` is simpler, faster to set up, and more appropriate.

### 3.3 kipoiseq vs. GenVarLoader: Summary

| Aspect | kipoiseq | GenVarLoader |
|--------|----------|-------------|
| **Primary use case** | Prediction-time: extract ref/alt sequences for scoring | Training-time: iterate over personalized haplotypes |
| **Variant types** | SNPs + indels (unified treatment) | SNPs + indels (with track realignment) |
| **Fixed-length output** | Yes (`fixed_len=True` with anchor-based trimming) | Yes (`with_len(N)` with jitter) |
| **VCF parsing** | cyvcf2 | genoray (VCF, PGEN, SparseVar) |
| **Batch/parallel** | Sequential (one variant/interval at a time) | Numba-parallel across regions x samples x ploidy |
| **Multi-sample** | Yes (via `sample_id` parameter) | Yes (core design, sparse genotype storage) |
| **Track realignment** | No | Yes (BigWig signals realigned to haplotype coords) |
| **Storage format** | Direct VCF/FASTA access (no preprocessing) | Preprocessed `.gvl` dataset (requires write step) |
| **Dependencies** | Light (cyvcf2, pyfaidx, pyranges) | Heavy (numba, genoray, seqpro, polars, awkward) |
| **PyTorch integration** | None (returns strings) | `to_dataloader()`, nested tensors |
| **Publication** | Nature Biotechnology 2019 (Kipoi) | bioRxiv preprint 2025 (not peer-reviewed) |
| **Maintenance** | Sporadic (last commit Sep 2024) | Active (release 11 days ago) |
| **Best for cerberus** | Phase 1: variant scoring tool | Phase 3: haplotype-augmented training |

### 3.4 tangermeme -- Analysis patterns to lift (not depend on)

**Repository:** `../s2f-models/repos/tangermeme`

tangermeme provides well-designed implementations of ISM, DeepLIFT/SHAP, marginalization, ablation, and variant effect scoring for PyTorch models. However, taking it as a dependency is undesirable because:
- It expects raw tensor outputs from `model(X)`, while cerberus returns structured dataclasses. A wrapper is needed regardless.
- Its `predict()` function duplicates cerberus's own batched inference with `ModelEnsemble`.
- The core algorithms are simple enough to lift into cerberus directly.

**Code worth lifting:**

| tangermeme module | Core algorithm | Lines of actual logic | Why lift it |
|-------------------|---------------|----------------------|-------------|
| `variant_effect.py` | `substitution_effect`, `deletion_effect`, `insertion_effect` | ~150 | Clean ref/alt tensor construction with edge trimming for indels. But kipoiseq's `VariantSeqExtractor` handles this at the string level, so this is redundant if we use kipoiseq. |
| `saturation_mutagenesis.py` | `_edit_distance_one` (Numba), `_attribution_score` | ~100 | Batched ISM generation. Replaces the naive triple loop in `plot_biasnet_ism.py`. Worth lifting for a future generalized ISM tool. |
| `deep_lift_shap.py` | Hook-based DeepLIFT/SHAP | ~300 | Per-nucleotide attribution. Complex but self-contained. Lift when attribution is needed. |
| `ersatz.py` | `substitute()`, `shuffle()`, `dinucleotide_shuffle()` | ~100 | Sequence perturbation primitives. Simple, useful for controls. |

**Verdict:** Do not add tangermeme as a dependency. Lift specific algorithms when needed, adapting them to work with cerberus's `ModelOutput` dataclasses directly.

### 3.5 bilby -- Reference architecture for VCF-to-score pipeline

**Repository:** `../s2f-models/repos/bilby` (JAX/Flax -- not directly reusable, architecturally instructive).

**`score_snps()` function** (`bilby/snps.py`): The most complete VCF-to-variant-effect-score pipeline in the reviewed repos:
1. Parse VCF with pysam, cluster proximal SNPs.
2. For each variant: extract ref sequence, apply alt allele at center.
3. Run model on ref and alt sequences.
4. Compute effect statistics: SAD, log fold change.
5. Strand-paired target handling (forward + reverse predictions averaged).
6. **Shift ensembling:** Average predictions across multiple sequence shifts (e.g., -2, -1, 0, +1, +2 bp) for CNN artifact reduction.
7. Output to HDF5 per-variant, per-target.

**Key design decisions to adopt:**
- **Shift ensembling** is important for cerberus profile models where the output grid is fixed. Small input shifts reduce position-dependent CNN artifacts.
- **SNP clustering:** Group variants within `input_len/2` to share reference sequence extraction.
- **HDF5 output** for millions of variants.

### 3.6 gReLU -- ISM dataset pattern

**Repository:** `../s2f-models/repos/gReLU` (Nature Methods 2025, Genentech).

`ISMDataset` (`grelu/data/dataset.py:869+`) is a PyTorch `Dataset` that generates all single-nucleotide mutations on-the-fly in `__getitem__`. Clean pattern worth considering for a generalized cerberus ISM tool, but tangermeme's Numba-based `_edit_distance_one` is more efficient for batched generation.

### 3.7 Other tools in the landscape

| Tool | Year | Venue | Variant support | Notes |
|------|------|-------|----------------|-------|
| **Selene** | 2019 | Nature Methods | Reference only | General DL training on sequences |
| **Janggu** | 2020 | Nature Comms | Reference only | Multi-modal data loading (FASTA/BAM/BigWig) |
| **EUGENe** | 2023 | Nature Comp Sci | Limited | FAIR toolkit, same ecosystem as GVL (Adam Klie is co-author of both) |
| **AlphaGenome** | 2025 | Nature | API-based scoring | DeepMind's 1Mb context model, closed source |
| **Enformer** | 2021 | Nature Methods | Prediction-time only | Uses custom data loaders, no VCF integration |

No other tool combines training-time haplotype augmentation with track realignment the way GenVarLoader does. For prediction-time variant scoring, kipoiseq is the most established and lightest-weight option.

---

## 4. Training vs. Prediction: Distinct Facilities

### 4.1 Prediction-time variant scoring (Phase 1)

**Goal:** Given a trained model and a VCF file, compute the predicted effect of each variant on the model's output (profile shape and total count).

**What changes in the data path:**
- No changes to `CerberusDataset`, `DataConfig`, extractors, transforms, or samplers.
- A new tool (`tools/score_variants.py`) constructs one-hot ref/alt sequences directly from FASTA + VCF, bypassing the training data pipeline entirely.
- `ModelEnsemble` is used as-is for batched prediction.

**What changes in the model path:**
- No changes to model architectures, loss functions, or training.
- Signal reconstruction from `ModelOutput` dataclasses is handled in the new `variants.py` module.

**Key design:**
```
VCF + FASTA
    |
Parse variants (cyvcf2 via kipoiseq or directly)
    |
For each variant (or cluster of proximal variants):
  - Compute variant_center = pos + len(ref) // 2
  - Extract ref sequence centered on variant_center, padded to input_len
  - Construct alt sequence from same window (apply variant, trim symmetrically)
  - One-hot encode both (cerberus encode_dna)
    |
Batch ref/alt pairs
    |
ModelEnsemble.forward() on ref batch -> ref_output
ModelEnsemble.forward() on alt batch -> alt_output
    |
Reconstruct linear signal from both outputs
    |
Compute effect sizes (SAD, logFC, JSD, Pearson delta)
    |
Output: TSV/HDF5 with per-variant effect scores
```

**Infrastructure needed (all additive):**
1. `tools/score_variants.py` -- CLI tool (analogous to existing `tools/export_bigwig.py`).
2. `src/cerberus/variants.py` -- VCF parsing, ref/alt sequence construction, effect size computation. Small module (~200-300 lines). Core logic either lifted from kipoiseq or using kipoiseq as a dependency.
3. No new model code. No changes to existing modules.

### 4.2 Training-time haplotype augmentation (Phase 3)

**Goal:** Train models on personalized haplotype sequences instead of (or in addition to) the reference genome. Enables learning allele-specific effects and improves generalization to diverse populations.

**What changes in the data path:**
- `BaseSequenceExtractor` protocol (`sequence.py:89-97`) is the clean extension point. A new `HaplotypeSequenceExtractor` wraps GenVarLoader.
- `DataConfig` needs an optional `variants_path` field (VCF/PGEN/GVL path) and `samples` list.
- `CerberusDataset.__getitem__()` calls the extractor which returns a haplotype instead of reference sequence. The rest of the pipeline (transforms, loss, model) is unchanged.
- **Critical complication -- indels and track alignment:**
  - **SNP-only mode:** Restrict to SNPs. Sequence length preserved. BigWig targets unaffected. Simple and sufficient for most GWAS applications (~85% of common variants).
  - **Indel-aware mode:** Use GenVarLoader's `shift_and_realign_tracks_sparse()` to shift target BigWig signals. Requires changes to signal extraction and target alignment. Significantly more complex.

**What changes in the model path:**
- No changes to model architectures. Models accept `(B, 4, input_len)` regardless of whether the sequence is reference or haplotype.
- Loss functions and metrics unchanged.
- Training data now includes genetic diversity, which may require larger datasets or longer training.

**Infrastructure needed:**
1. `src/cerberus/haplotype.py` -- `HaplotypeSequenceExtractor` implementing `BaseSequenceExtractor`.
2. `DataConfig` extension with optional variant/sample fields.
3. GenVarLoader as optional dependency (not required for basic cerberus usage).

### 4.3 Comparison table

| Aspect | Prediction-time scoring (Phase 1) | Training-time haplotypes (Phase 3) |
|--------|-----------------------------------|-------------------------------------|
| **Goal** | Score variant effects on trained model | Train model on personalized genomes |
| **Core library changes** | None -- tool + small module | `DataConfig`, new extractor class |
| **External dependency** | kipoiseq or lifted code (~200 lines) | GenVarLoader (heavy, optional) |
| **Model changes** | None | None |
| **Indel handling** | Anchor-based trimming (kipoiseq) | Complex (track realignment or SNP-only) |
| **Effort** | Small (1-2 days) | Medium-large (3-5 days) |
| **Value** | Immediate (GWAS screening) | Strategic (allele-specific models) |
| **Risk** | Low (additive, no core changes) | Medium (data pipeline changes, unreviewed dependency) |

---

## 5. Detailed Design: Variant Scoring Tool (Phase 1)

### 5.1 VCF parsing and variant representation

Two options for VCF handling:

**Option A -- kipoiseq as dependency:** Use `MultiSampleVCF` for VCF parsing and `VariantSeqExtractor` for ref/alt construction directly. Pros: battle-tested, handles edge cases. Cons: pulls in kipoiseq dependency tree (cyvcf2, pyranges, pandas).

**Option B -- Lift core logic:** The essential code in kipoiseq's `VariantSeqExtractor` is ~200 lines (`vcf_seq.py:60-303`). The VCF parsing via cyvcf2 is straightforward (~50 lines in `vcf.py`). Lift these into `src/cerberus/variants.py` with only cyvcf2 + pyfaidx as dependencies (both already available or trivially installable). This avoids the broader kipoiseq/kipoi dependency tree.

**Recommendation:** Option B (lift core logic). cyvcf2 is the only new dependency, and we avoid coupling to the Kipoi ecosystem. The `Variant` representation can be a simple frozen dataclass:

```python
@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int          # 0-based
    ref: str
    alt: str
    id: str = "."
    @property
    def is_snp(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1
```

### 5.2 Ref/alt sequence construction -- window centering

**Two separate concerns must not be conflated:**

1. **Window placement** -- where to center the `input_len` window in the genome.
2. **Sequence construction** -- how to build the alt sequence so that alignment with the ref is preserved where possible.

#### Window placement: center on variant midpoint

Both ref and alt windows are centered on the **midpoint of the variant footprint** in reference coordinates, not on the first base:

```
variant_center = variant.pos + len(variant.ref) // 2
window_start   = variant_center - input_len // 2
```

**Why:** A 20bp deletion at position 1000 spans [1000, 1020) in reference. Centering on `pos=1000` pushes the entire deleted region to the right of center. Centering on `pos + 10 = 1010` places the deletion symmetrically within the model's receptive field, giving equal flanking context on both sides.

For SNPs (`len(ref) == 1`), `variant_center = pos` -- no difference from the naive approach.

#### Sequence construction: symmetric trimming from both flanks

For indels, the alt sequence is a different length than the ref. To restore `input_len`, we trim (insertion) or extend (deletion) **symmetrically from both flanks**, keeping the variant centered.

```
Example: 3bp deletion at pos 1000, ref=ACGT, alt=A, input_len=2112, centered at 1002

ref window: [1002 - 1056 ... 1000=ACGT ... 1002 + 1056)    = 2112 bp
alt window: [1002 - 1056 - 1 ... 1000=A ... 1002 + 1056 + 2) = 2112 bp
             ^^^ shifted 1bp ^^^         ^^^ shifted 2bp ^^^
```

No bases in the flanking regions are positionally aligned between ref and alt -- every position is shifted by `floor(indel_len/2)` or `ceil(indel_len/2)` depending on the flank. This is **intentional** and preferable to one-sided anchoring for several reasons:

1. **Honest comparison.** Per-position alignment is fundamentally impossible for indels -- the genome itself is different. One-sided anchoring creates a false asymmetry: one flank appears "aligned" while the other is shifted, which can inflate per-position metrics (a peak that shifted by 3bp looks like a peak disappearing + appearing). Symmetric trimming treats both flanks equally.

2. **Centered in the output window.** The variant sits at the center of the model's output profile, where the receptive field has full context and edge artifacts are minimized. With one-sided anchoring, the effect bleeds toward one edge.

3. **Aggregate metrics don't need alignment.** The primary outputs (SAD, log FC, JSD, Pearson) compare total signal or distributional properties across the entire profile. They answer "does this variant change the prediction in this region" -- an inherently aggregate question. No per-position correspondence is needed.

4. **SNPs are unaffected.** For SNPs (`len(ref) == len(alt) == 1`), no trimming is needed. All positions are perfectly aligned. The symmetric approach only diverges from naive per-position comparison for indels.

**Construction steps:**
1. Compute `variant_center = pos + len(ref) // 2`.
2. Extract **ref** sequence from FASTA: `[variant_center - input_len//2, variant_center + input_len//2)`.
3. Construct **alt** sequence:
   - **SNP:** Direct substitution at the variant offset within the window. No length change.
   - **Deletion:** Remove deleted bases from the ref string. Extend the FASTA fetch by `ceil(del_len/2)` upstream and `floor(del_len/2)` downstream to restore `input_len`.
   - **Insertion:** Splice in the alt allele at the variant offset. Trim `floor(ins_len/2)` from the upstream end and `ceil(ins_len/2)` from the downstream end to maintain `input_len`.
4. One-hot encode both via `encode_dna()`.

Both tensors are `(4, input_len)`. The `Interval` for fold routing corresponds to the ref window coordinates.

### 5.3 Effect size computation

Multiple complementary metrics (inspired by bilby `score_snps` and Enformer scoring):

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **SAD** (Sum of Absolute Differences) | `sum(abs(pred_alt - pred_ref))` | Total magnitude of change across profile |
| **Log Fold Change (count)** | `log_counts_alt - log_counts_ref` | Change in total predicted signal |
| **Profile JSD** | `JSD(softmax(logits_ref), softmax(logits_alt))` | Divergence in profile shape |
| **Profile Pearson r** | `corr(pred_ref, pred_alt)` | Correlation between ref and alt profiles |
| **Max absolute delta** | `max(abs(pred_alt - pred_ref))` | Peak local effect |

For Dalmatian models, also compute:
- **Signal-only effect:** Compare `signal_logits` (TF binding component, bias detached).
- **Bias-only effect:** Compare `bias_logits` (Tn5 bias). Should be near zero for real regulatory variants.

### 5.4 Output format

Two formats:
1. **TSV** (default): One row per variant with columns for variant ID, chrom, pos, ref, alt, and each effect metric per channel. Human-readable, compatible with downstream GWAS tools.
2. **HDF5** (optional, for large-scale): Stores full ref/alt profiles per variant for post-hoc re-analysis.

### 5.5 Shift ensembling

Following bilby's approach: for each variant, run predictions at multiple small offsets (e.g., [-3, -1, 0, +1, +3] bp) and average. This reduces CNN positional artifacts and improves score stability. The offset shifts the input window relative to the variant; the variant always falls within the window.

Cost: 5x more forward passes. Benefit: more robust scores, especially near output bin boundaries. Implement as optional `--n-shifts` CLI flag.

### 5.6 Batching strategy

**Sparse variants (typical GWAS):** Each variant produces one ref + one alt sequence. Batch independently across variants.

**Dense variants (saturation mutagenesis):** Group variants by region (bilby's SNP clustering approach: group within `input_len/2`). Extract reference once per cluster, share across variants.

### 5.7 Integration with ModelEnsemble

The scoring tool uses `ModelEnsemble` for prediction, gaining:
- Automatic fold routing (variants scored by held-out models).
- Multi-fold ensembling (averaged predictions).
- Device management.

The tool constructs one-hot tensors directly (bypassing `CerberusDataset`) and calls `ModelEnsemble._forward_models()` with `intervals` for fold routing.

---

## 6. Detailed Design: Haplotype Training (Phase 3)

### 6.1 HaplotypeSequenceExtractor

A new class implementing `BaseSequenceExtractor` that wraps GenVarLoader:

```python
class HaplotypeSequenceExtractor(BaseSequenceExtractor):
    """Extracts haplotype sequences from a GenVarLoader dataset."""
    
    def __init__(self, gvl_dataset, sample_idx: int, haplotype: int = 0,
                 encoding: str = "ACGT"):
        self.gvl = gvl_dataset.with_seqs("haplotypes").with_len("variable")
        self.sample_idx = sample_idx
        self.haplotype = haplotype
        self.encoding = encoding
    
    def extract(self, interval: Interval) -> torch.Tensor:
        # Map cerberus Interval to GVL region index
        hap_bytes = self.gvl[region_idx, self.sample_idx]
        seq = hap_bytes[self.haplotype].decode()
        return encode_dna(seq, self.encoding)
```

**Challenge:** GenVarLoader indexes by pre-registered region index, not by arbitrary genomic coordinates. Integration requires either pre-registering all training intervals at dataset creation, or a coordinate-based lookup layer.

### 6.2 SNP-only mode (recommended initial approach)

For fixed-length profile models:
- Filter VCF to SNPs only (`len(ref) == 1 and len(alt) == 1`).
- Sequence length guaranteed to match `input_len`.
- No track realignment needed -- BigWig signals are position-invariant for SNPs.
- Covers ~85% of common GWAS variants.

### 6.3 Indel-aware mode (future)

For full variant support:
- GenVarLoader handles variable-length haplotype reconstruction correctly.
- Target signal tracks must be realigned via `shift_and_realign_tracks_sparse()`.
- Cerberus's `SignalExtractor` must be extended to produce realigned targets.
- The model's `input_len` constraint means either: (a) pad/trim haplotypes to fixed length, or (b) modify the training pipeline for variable-length inputs (major change, not recommended).

### 6.4 GenVarLoader maturity considerations

- **Not yet peer-reviewed** (bioRxiv preprint only as of April 2026).
- Primarily solo-developed by one PhD student (David Laub).
- Actively maintained with frequent releases.
- Heavy dependency chain (numba, genoray, seqpro, polars, awkward arrays).
- Adopted by several published projects (Nucleotide Transformer, Variformer).
- No viable alternative exists for this use case.

**Mitigation:** Pin to a specific version. Isolate behind the `BaseSequenceExtractor` protocol so the rest of cerberus is unaffected if GVL API changes.

---

## 7. Recommended Implementation Roadmap

### Phase 1: Variant scoring tool (immediate value)

**Priority: HIGH | Effort: Small | Risk: Low**

1. Create `src/cerberus/variants.py`:
   - `Variant` dataclass.
   - `load_vcf(path, region=None) -> list[Variant]` -- parse VCF with cyvcf2.
   - `variant_to_ref_alt(variant, fasta, input_len) -> tuple[Tensor, Tensor]` -- construct one-hot ref/alt pairs. Core logic lifted from kipoiseq's `VariantSeqExtractor`.
   - `compute_variant_effects(ref_output, alt_output) -> dict[str, float]` -- effect size metrics.
   - `reconstruct_linear_signal(output) -> Tensor` -- shared utility to convert `ModelOutput` to comparable tensors.

2. Create `tools/score_variants.py`:
   - CLI: model checkpoint, VCF, FASTA, output path, optional BED for region filter.
   - Load `ModelEnsemble`, parse VCF, batch score variants, write TSV/HDF5.
   - `--channels` to select output channels; `--n-shifts` for shift ensembling.

### Phase 2: Generalized ISM tool

**Priority: MEDIUM | Effort: Small-Medium | Risk: Low**

3. Create `tools/ism.py` replacing the naive `plot_biasnet_ism.py`:
   - Works with any cerberus model (Pomeranian, Dalmatian, BPNet).
   - Batched ISM generation (lift `_edit_distance_one` from tangermeme or implement simple batched mutation).
   - Profile-level and count-level effect sizes.
   - Optional DeepLIFT/SHAP attribution (lift from tangermeme when needed).

### Phase 3: Haplotype training (strategic)

**Priority: LOW (unless allele-specific modeling is a near-term goal) | Effort: Large | Risk: Medium**

4. Add GenVarLoader as optional dependency.
5. Implement `HaplotypeSequenceExtractor` (SNP-only mode first).
6. Extend `DataConfig` with optional variant configuration.
7. Create example training script.
8. (Future) Indel-aware mode with track realignment.

---

## 8. Dependency Analysis

| Dependency | Phase | New? | Weight | Notes |
|------------|-------|------|--------|-------|
| cyvcf2 | 1 | Yes | Light (C ext, conda/pip) | Only new dependency for Phase 1 |
| pyfaidx | 1 | No | Already used | Reference sequence extraction |
| h5py | 1 (HDF5 output) | Likely available | Standard | Optional output format |
| GenVarLoader | 3 | Yes | Heavy (numba, genoray, seqpro, polars) | Optional, isolated behind protocol |

Phase 1 adds only **one new dependency** (cyvcf2). Phase 2 adds none. Phase 3 adds GenVarLoader as an optional dependency.

---

## 9. Comparison of Approaches Across Repositories

| Feature | kipoiseq | GenVarLoader | tangermeme | bilby | gReLU | cerberus (current) |
|---------|----------|-------------|-----------|-------|-------|-------------------|
| **VCF parsing** | cyvcf2 | genoray | `io.read_vcf()` | pysam | No | No |
| **SNP scoring** | `VariantSeqExtractor` | `DatasetWithSites` | `substitution_effect()` | `score_snps()` | `ISM_predict()` | Naive ISM (BiasNet only) |
| **Indel handling** | Anchor-based trimming | Full haplotype reconstruction | Edge trimming | Shift compensation | No | No |
| **Batch prediction** | Sequential | Numba parallel | GPU-managed | JAX vmap | DataLoader | `ModelEnsemble` |
| **Shift ensembling** | No | Jitter param | No | Yes | No | No |
| **Haplotype training** | No | Core feature | No | No | No | No |
| **Track realignment** | No | Yes (BigWig) | No | No | No | No |
| **Attribution** | No | No | DeepLIFT/SHAP, PISA | No | DeepSHAP | No |
| **Framework** | Framework-agnostic | NumPy/Numba | PyTorch | JAX/Flax | PyTorch | PyTorch |
| **Best cerberus role** | Phase 1: lift VCF/seq logic | Phase 3: haplotype training | Lift ISM/attribution code | Architecture reference | ISMDataset pattern | -- |

---

## 10. Risk Assessment

### Phase 1 risks (low)

- **VCF edge cases:** Multi-allelic sites, overlapping variants, structural variants. Mitigate by filtering to biallelic SNPs/small indels initially.
- **Output interpretation:** Profile model outputs (logits + counts) are not directly comparable to standard Enformer-style variant effect scores. Need clear documentation of what metrics mean for multinomial profile models.
- **Memory for large VCFs:** Millions of variants x input_len one-hot tensors. Mitigate by streaming (process in chunks, write incrementally).

### Phase 2 risks (low)

- **DeepLIFT/SHAP hook compatibility:** If lifted from tangermeme, hooks may interact with cerberus's ConvNeXtV2Block or PGCBlock layers. Needs testing per model architecture.

### Phase 3 risks (medium)

- **GenVarLoader API stability:** Under active development, not yet peer-reviewed. Pin to specific version.
- **Coordinate alignment:** Ensuring GVL haplotype positions correctly map to cerberus target BigWig signals. Off-by-one errors would silently corrupt training data.
- **Performance:** On-the-fly haplotype reconstruction adds data loading latency. May need GVL's in-memory mode or pre-computation.

---

## 11. Conclusion

Variant effect prediction requires two distinct facilities that should not be conflated:

**For prediction (scoring variants against a trained model):** A self-contained tool with minimal dependencies. The core logic is straightforward -- construct ref/alt one-hot sequences from VCF + FASTA, run both through `ModelEnsemble`, compare outputs. kipoiseq's `VariantSeqExtractor` provides a well-tested reference implementation for the sequence construction step (~200 lines worth lifting). The only new dependency is cyvcf2 for VCF parsing. This delivers immediate value for GWAS screening with no changes to cerberus's core library.

**For training (learning from genetic variation):** GenVarLoader is the only tool that solves this at scale. It integrates cleanly behind cerberus's `BaseSequenceExtractor` protocol. However, it is a heavier commitment -- unreviewed preprint, solo developer, heavy dependency chain. The SNP-only mode is a pragmatic starting point that avoids the complexity of indel-aware track realignment. This is a Phase 3 effort, worth pursuing when allele-specific modeling becomes a near-term goal.

Analysis methods (ISM, attribution) can be built incrementally by lifting specific algorithms from tangermeme as needed, rather than taking it as a dependency.
