# Variant Effect Prediction

Cerberus supports scoring the predicted effect of genomic variants (SNPs, indels, delins) on model outputs. This is useful for prioritizing GWAS variants, interpreting non-coding mutations, and validating that a trained model captures known regulatory variants.

## Overview

The variant scoring workflow compares model predictions on a **reference** sequence versus an **alternative** sequence containing the variant. The pipeline is:

```
VCF / TSV + FASTA
    |
    v
load_vcf() / load_variants()   Parse variants, convert to 0-based coordinates
    |
    v
variant_to_ref_alt()  Construct paired one-hot tensors centered on the variant
    |
    v
ModelEnsemble        Run model on ref and alt tensors (batched)
    |
    v
compute_variant_effects()  Compare outputs: SAD, log fold change, JSD, etc.
    |
    v
TSV / HDF5 output
```

No changes to the model, training pipeline, or core library are required. Variant scoring is purely a prediction-time workflow.

## Installation

Variant effect prediction requires the `cyvcf2` package for VCF parsing:

```bash
pip install cerberus[variants]
```

Or install cyvcf2 directly:

```bash
pip install cyvcf2
```

## VCF Preparation

The input VCF **must be biallelic and left-aligned**. Multi-allelic records are skipped with a warning. Normalize your VCF with:

```bash
bcftools norm -m- -f reference.fa input.vcf.gz \
  | bcftools view -v snps,indels \
  | bgzip -c > normalized.vcf.gz
tabix -p vcf normalized.vcf.gz
```

Region-based queries (the `region` parameter) require a bgzipped and indexed VCF (`.vcf.gz` + `.tbi`).

## Core API

### Variant

The `Variant` dataclass represents a single genomic variant using **0-based** coordinates, consistent with cerberus's `Interval` class. VCF files use 1-based POS; `load_vcf()` converts automatically.

```python
from cerberus.variants import Variant

# Manual construction (0-based pos)
snp = Variant("chr1", 99, "A", "G", id="rs12345")
deletion = Variant("chr1", 99, "ACGT", "A")
insertion = Variant("chr1", 99, "A", "ACGT")

# Properties
snp.end          # 100 — half-open: [99, 100)
snp.ref_center   # 99  — midpoint of ref footprint
snp.is_snp       # True
deletion.size_change  # -3

# Convert to Interval (for fold routing)
snp.to_interval()  # Interval("chr1", 99, 100)

# Parse from string
v = Variant.from_str("chr1:99:A>G")
```

The `info` field stores VCF INFO values as a dict:

```python
v = Variant("chr1", 99, "A", "G", info={"AF": 0.05, "DP": 100})
v.info["AF"]  # 0.05
```

### load_variants

Parses a simple tab-separated file with `chrom`, `pos`, `ref`, `alt` columns:

```python
from cerberus.variants import load_variants

# Load from a TSV (1-based pos by default, like VCF/dbSNP/ClinVar)
variants = list(load_variants("my_variants.tsv"))

# If positions are 0-based (BED-derived data)
variants = list(load_variants("my_variants.tsv", zero_based=True))
```

The file format is tab-delimited with a header row. Required columns: `chrom`, `pos`, `ref`, `alt` (in any order). An optional `id` column populates `Variant.id`. Extra columns are ignored. Lines starting with `#` (other than the header) are skipped.

Example TSV file:

```
#chrom	pos	ref	alt	id
chr1	100	A	G	rs001
chr1	200	C	T	rs002
chr1	300	ACGT	A	rs003
```

**Coordinate convention**: By default, `pos` is interpreted as **1-based** — the convention used by VCF, dbSNP, ClinVar, HGVS, and most variant databases. The loader subtracts 1 to convert to cerberus's 0-based system. Set `zero_based=True` for 0-based input.

### load_vcf

Parses a VCF/BCF file and yields `Variant` objects:

```python
from cerberus.variants import load_vcf
from cerberus.interval import Interval

# Load all PASS variants
variants = list(load_vcf("variants.vcf.gz"))

# Filter to a region (requires indexed VCF)
variants = list(load_vcf(
    "variants.vcf.gz",
    region=Interval("chr1", 0, 1_000_000),  # 0-based half-open
))

# Or with a tabix-style string (1-based inclusive)
variants = list(load_vcf("variants.vcf.gz", region="chr1:1-1000000"))

# Include non-PASS variants
variants = list(load_vcf("variants.vcf.gz", pass_only=False))

# Capture specific INFO fields
variants = list(load_vcf("variants.vcf.gz", info_fields=["AF", "DP"]))
```

`load_vcf` returns a **generator** for memory efficiency with large VCFs. Call `list()` to materialize.

### variant_to_ref_alt

Constructs paired one-hot encoded ref and alt tensors:

```python
import pyfaidx
from cerberus.variants import variant_to_ref_alt

fasta = pyfaidx.Fasta("reference.fa")
variant = Variant("chr1", 99, "A", "G")

ref_tensor, alt_tensor, interval = variant_to_ref_alt(
    variant, fasta, input_len=2112
)
# ref_tensor: (4, 2112) — one-hot reference sequence
# alt_tensor: (4, 2112) — one-hot alternative sequence
# interval:   Interval("chr1", ...) — window for fold routing
```

#### Window centering

The input window is centered on `variant.ref_center` — the **midpoint of the reference allele footprint**, not the first base of the variant. This ensures the variant is symmetrically placed within the model's receptive field.

For SNPs, `ref_center == pos`. For a 20bp deletion at position 1000, `ref_center = 1010`.

#### Indel handling

For indels, the alt sequence is constructed by splicing in the alternative allele and trimming **symmetrically from both flanks** to restore `input_len`. This means no bases are positionally aligned between ref and alt — the comparison is intentionally aggregate (SAD, JSD) rather than per-position.

For SNPs and MNPs, all positions are perfectly aligned.

#### Validation

The function raises `ValueError` if:

- The window extends beyond chromosome boundaries.
- The FASTA sequence at the variant position does not match `variant.ref` (wrong genome build).
- The chromosome is not found in the FASTA.

### compute_variant_effects

Compares ref and alt model outputs to compute effect metrics:

```python
from cerberus.variants import compute_variant_effects
from cerberus.output import get_log_count_params

# Get pseudocount settings from model config
log_counts_include_pseudocount, pseudocount = get_log_count_params(
    ensemble.cerberus_config.model_config_
)

# Run model on ref and alt
ref_output = ensemble(ref_batch)
alt_output = ensemble(alt_batch)

# Compute effects
effects = compute_variant_effects(
    ref_output, alt_output,
    log_counts_include_pseudocount=log_counts_include_pseudocount,
    pseudocount=pseudocount,
)
```

Returns a `dict[str, torch.Tensor]` with shape `(B, C)` per metric:

| Metric | Key | Description |
|--------|-----|-------------|
| Sum of Absolute Differences | `"sad"` | Total signal change across the profile |
| Max Absolute Difference | `"max_abs_diff"` | Largest local effect at any position |
| Pearson Correlation | `"pearson"` | Similarity between ref and alt profile shapes |
| Log Fold Change | `"log_fc"` | Change in predicted total counts (log space) |
| Jensen-Shannon Divergence | `"jsd"` | Profile shape divergence (0 = identical, ln2 = maximal) |

`log_fc` and `jsd` are only available for `ProfileCountOutput` models (BPNet, Pomeranian, Dalmatian).

#### Dalmatian models

For `FactorizedProfileCountOutput` (Dalmatian), additional signal sub-model metrics are computed automatically:

- `"signal_sad"`, `"signal_log_fc"`, `"signal_jsd"`

These isolate the regulatory signal component from the Tn5 bias. A variant that changes `signal_*` metrics but not the combined metrics may be masked by bias; one that changes combined but not signal may be a Tn5 artifact.

## Output Functions

The variant scoring workflow uses the `compute_*` functions from `cerberus.output` to convert raw model outputs into comparable quantities:

| Function | Output shape | Space | Description |
|----------|-------------|-------|-------------|
| `compute_signal` | `(B, C, L)` | linear | Full predicted signal (shape * magnitude combined) |
| `compute_profile_probs` | `(B, C, L)` | probability | Normalized profile shape (sums to 1 along L) |
| `compute_channel_log_counts` | `(B, C)` | log | Per-channel total predicted counts |
| `compute_total_log_counts` | `(B,)` | log | Total predicted counts across all channels |

All accept `ModelOutput` and the `(log_counts_include_pseudocount, pseudocount)` parameter pair from `get_log_count_params()`.

## Batched Scoring

### score_variants_from_ensemble

The recommended way to score variants. Handles batching, config extraction, error handling (skips bad variants with warnings), and fold routing automatically:

```python
from cerberus.model_ensemble import ModelEnsemble
from cerberus.predict_variants import score_variants_from_ensemble
from cerberus.variants import load_vcf

ensemble = ModelEnsemble("logs/my_dalmatian", device="cuda")
variants = load_vcf("gwas_hits.vcf.gz")

for result in score_variants_from_ensemble(ensemble, variants, batch_size=64):
    v = result.variant
    e = result.effects
    print(f"{v}  SAD={e['sad'].item():.2f}  logFC={e['log_fc'].item():.4f}")
```

Each `VariantResult` contains:

- `variant` — the scored `Variant`
- `effects` — `dict[str, torch.Tensor]` with per-channel metrics (shape `(C,)`)
- `interval` — the reference-coordinate input window used for scoring

Parameters like `input_len`, `fasta_path`, and pseudocount settings are extracted from `ensemble.cerberus_config` automatically. Pass a `fasta` argument to override the FASTA path.

### score_variants

Lower-level function for use with a plain `nn.Module` or when you need full control over parameters:

```python
from cerberus.predict_variants import score_variants

for result in score_variants(
    model=my_model,
    variants=variants,
    fasta=fasta,
    input_len=2112,
    log_counts_include_pseudocount=True,
    pseudocount=1.0,
    device=torch.device("cuda"),
    batch_size=64,
):
    ...
```

Both functions are **generators** — results are yielded as they are computed, so memory usage is bounded by `batch_size`, not the total number of variants.

## CLI Tool

### Score Variants (`tools/score_variants.py`)

Scores variant effects from the command line, writing results to a TSV file:

```bash
# From a VCF
python tools/score_variants.py path/to/model_dir \
    --vcf variants.vcf.gz --output effects.tsv

# From a tab-delimited variant file
python tools/score_variants.py path/to/model_dir \
    --variants variants.tsv --output effects.tsv

# With region filter and custom folds
python tools/score_variants.py path/to/model_dir \
    --vcf variants.vcf.gz --output effects.tsv \
    --region chr1:1000000-2000000 --use-folds test+val
```

Output columns: `chrom`, `pos_0based`, `ref`, `alt`, `id`, and per-channel effect metrics (e.g. `sad_ch0`, `log_fc_ch0`, `jsd_ch0`).

| Argument | Description |
|---|---|
| `--vcf` / `--variants` | Variant source (mutually exclusive, required). |
| `--output` | Output TSV path (default: `variant_effects.tsv`). Supports `.gz`. |
| `--fasta` | Override FASTA path (default: from model config). |
| `--region` | Restrict to variants in a region (e.g. `chr1:1000000-2000000`). |
| `--batch-size` | Variants per inference batch (default: 64). |
| `--use-folds` | Folds for ensemble prediction (e.g. `test`, `test+val`, `all`). |
| `--device` | Device override (`cuda`, `mps`, `cpu`). |
| `--zero-based` | Interpret `--variants` positions as 0-based (default: 1-based). |

## Manual Workflow (Low-Level)

For custom scoring logic where `score_variants` doesn't fit:

```python
import pyfaidx
import torch
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import get_log_count_params
from cerberus.variants import (
    compute_variant_effects,
    load_vcf,
    variant_to_ref_alt,
)

# 1. Load model
ensemble = ModelEnsemble("logs/my_dalmatian", device="cuda")
config = ensemble.cerberus_config
input_len = config.data_config.input_len

# 2. Resolve pseudocount settings
lc_include, pseudocount = get_log_count_params(config.model_config_)

# 3. Load variants
fasta = pyfaidx.Fasta(str(config.genome_config.fasta_path))
variants = list(load_vcf("gwas_hits.vcf.gz", pass_only=True))

# 4. Score each variant
for variant in variants:
    try:
        ref_t, alt_t, interval = variant_to_ref_alt(variant, fasta, input_len)
    except ValueError:
        continue  # skip edge-of-chromosome variants

    ref_batch = ref_t.unsqueeze(0).to(ensemble.device)
    alt_batch = alt_t.unsqueeze(0).to(ensemble.device)

    with torch.no_grad():
        ref_out = ensemble(ref_batch, intervals=[interval])
        alt_out = ensemble(alt_batch, intervals=[interval])

    effects = compute_variant_effects(
        ref_out, alt_out,
        log_counts_include_pseudocount=lc_include,
        pseudocount=pseudocount,
    )

    print(f"{variant}  SAD={effects['sad'].item():.2f}  logFC={effects['log_fc'].item():.4f}")
```
