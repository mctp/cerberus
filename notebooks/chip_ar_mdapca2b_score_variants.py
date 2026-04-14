# %% [markdown]
# # Variant Effect Scoring Demo (BPNet)
#
# This notebook demonstrates how to score the predicted effect of genomic
# variants using a trained BPNet model and cerberus's variant scoring pipeline.
#
# It loads the same pre-trained BPNet model as `chip_ar_mdapca2b_predict_bpnet.py`,
# creates a few synthetic variants within known peak regions, and scores them
# using `score_variants_from_ensemble` (the high-level API) and `score_variants`
# (the lower-level API).
#
# **Prerequisites:** Train the BPNet model first:
# ```bash
# bash examples/chip_ar_mdapca2b_bpnet.sh
# ```

# %%
import sys
from pathlib import Path

import torch

try:
    from paths import get_project_root
except ImportError:
    sys.path.append("notebooks")
    from paths import get_project_root

from cerberus.model_ensemble import ModelEnsemble
from cerberus.predict_variants import score_variants, score_variants_from_ensemble
from cerberus.utils import resolve_device
from cerberus.variants import Variant, load_vcf

# %% [markdown]
# ## 1. Setup

# %%
project_root = get_project_root()
data_dir = project_root / "tests/data"
checkpoint_dir = data_dir / "models/chip_ar_mdapca2b_bpnet/single-fold"

if not checkpoint_dir.exists():
    print(f"Checkpoint directory not found: {checkpoint_dir}")
    print("Run 'bash examples/chip_ar_mdapca2b_bpnet.sh' to train the model first.")
    sys.exit(0)

device = resolve_device()
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Load Model
#
# We use `ModelEnsemble` which automatically discovers the config and weights.

# %%
from cerberus.config import DataConfig, GenomeConfig, ModelConfig
from cerberus.download import download_dataset, download_human_reference
from cerberus.genome import create_genome_config

# Download/check data
genome_files = download_human_reference(data_dir / "genome", name="hg38")
dataset_files = download_dataset(data_dir / "dataset", name="mdapca2b_ar")

# Recreate configs (same as the training script)
genome_config: GenomeConfig = create_genome_config(
    name="hg38",
    fasta_path=genome_files["fasta"],
    species="human",
    fold_type="chrom_partition",
    fold_args={"k": 5, "val_fold": 1, "test_fold": 0},
    exclude_intervals={
        "blacklist": genome_files["blacklist"],
        "gaps": genome_files["gaps"],
    },
)

data_config = DataConfig(
    inputs={},
    targets={"signal": dataset_files["bigwig"]},
    input_len=2114,
    output_len=1000,
    output_bin_size=1,
    encoding="ACGT",
    max_jitter=0,
    log_transform=False,
    reverse_complement=False,
    target_scale=1.0,
    use_sequence=True,
)

model_config = ModelConfig(
    name="BPNet",
    model_cls="cerberus.models.bpnet.BPNet",
    loss_cls="cerberus.loss.PoissonMultinomialLoss",
    loss_args={},
    metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
    metrics_args={},
    model_args={
        "input_channels": ["A", "C", "G", "T"],
        "output_channels": ["signal"],
        "filters": 64,
        "n_dilated_layers": 8,
    },
)

ensemble = ModelEnsemble(
    checkpoint_path=checkpoint_dir,
    model_config=model_config,
    data_config=data_config,
    genome_config=genome_config,
    device=device,
)
print(f"Loaded models: {list(ensemble.keys())}")
print(f"Input length: {ensemble.cerberus_config.data_config.input_len}")

# %% [markdown]
# ## 3. Create Synthetic Variants
#
# We create a few variants near known peak summits from the training data.
# These are synthetic — in a real analysis you would load from a VCF file.

# %%
# Peak 1: chr6:93940029-93941032, summit offset 538 → summit at 93940567
# Peak 3: chr11:114178889-114180869, summit offset 586 → summit at 114179475

variants = [
    # SNP at peak 1 summit
    Variant("chr6", 93940567, "A", "G", id="peak1_summit_snp"),
    # SNP 100bp upstream of peak 1 summit
    Variant("chr6", 93940467, "C", "T", id="peak1_upstream_snp"),
    # SNP at peak 3 summit
    Variant("chr11", 114179475, "G", "A", id="peak3_summit_snp"),
    # SNP far from any peak (control — should have minimal effect)
    Variant("chr1", 10000000, "A", "C", id="intergenic_control"),
]

print(f"Scoring {len(variants)} variants:")
for v in variants:
    print(f"  {v} ({v.id})")

# %% [markdown]
# ## 4. Score Variants (High-Level API)
#
# `score_variants_from_ensemble` is the simplest way to score variants.
# It extracts `input_len`, FASTA path, and pseudocount settings from the
# ensemble config automatically.

# %%
results = list(score_variants_from_ensemble(
    ensemble=ensemble,
    variants=variants,
    batch_size=64,
))

print(f"\nScored {len(results)} variants:\n")
print(f"{'ID':<25} {'SAD':>10} {'log_fc':>10} {'JSD':>10} {'pearson':>10}")
print("-" * 70)
for r in results:
    e = r.effects
    print(
        f"{r.variant.id:<25} "
        f"{e['sad'].item():>10.2f} "
        f"{e['log_fc'].item():>10.4f} "
        f"{e['jsd'].item():>10.6f} "
        f"{e['pearson'].item():>10.4f}"
    )

# %% [markdown]
# ## 5. Score Variants (Low-Level API)
#
# `score_variants` gives you more control — you pass the model, FASTA,
# input length, and pseudocount parameters explicitly. This is useful when
# working with a plain `nn.Module` (not an ensemble) or when you need
# custom parameters.

# %%
import pyfaidx

from cerberus.output import get_log_count_params

fasta = pyfaidx.Fasta(str(genome_config.fasta_path))
lc_include, pseudocount = get_log_count_params(ensemble.cerberus_config.model_config_)

# Score just one variant to show the full VariantResult
single_variant = [Variant("chr6", 93940567, "A", "G", id="peak1_summit")]
result = next(score_variants(
    model=ensemble,
    variants=single_variant,
    fasta=fasta,
    input_len=data_config.input_len,
    log_counts_include_pseudocount=lc_include,
    pseudocount=pseudocount,
    device=device,
))

print(f"Variant: {result.variant}")
print(f"Interval: {result.interval}")
print(f"\nEffect metrics:")
for name, tensor in result.effects.items():
    print(f"  {name}: {tensor.item():.6f}")

fasta.close()

# %% [markdown]
# ## 6. Score from a VCF File
#
# In a real analysis, you would load variants from a VCF file.
# Here we demonstrate the pattern using the test VCF fixture (which has
# a tiny FASTA — the variants won't match the real genome, so they'll
# be skipped with warnings).

# %%
vcf_path = project_root / "tests/data/fixtures/test_variants.vcf"
vcf_variants = list(load_vcf(vcf_path))
print(f"Loaded {len(vcf_variants)} variants from VCF:")
for v in vcf_variants:
    print(f"  {v} ({v.id})")

# These will be skipped because the test VCF uses a tiny FASTA (104bp chr1)
# that doesn't match the real genome. In a real workflow, the VCF chromosomes
# and positions would match the model's reference genome.
results = list(score_variants_from_ensemble(
    ensemble=ensemble,
    variants=vcf_variants,
    batch_size=64,
))
print(f"\nScored {len(results)} / {len(vcf_variants)} variants "
      f"({len(vcf_variants) - len(results)} skipped due to FASTA mismatch)")

# %% [markdown]
# ## Summary
#
# - **`score_variants_from_ensemble()`** — High-level: pass ensemble + variants,
#   everything else is auto-detected from config.
# - **`score_variants()`** — Low-level: explicit model, FASTA, input_len, pseudocount.
# - **`VariantResult`** — Contains `.variant`, `.effects` (dict of per-channel tensors),
#   and `.interval` (the genomic window used for scoring).
# - **Effect metrics**: `sad`, `log_fc`, `jsd`, `pearson`, `max_abs_diff`
#   (plus `signal_*` variants for Dalmatian models).
# - **Error handling**: Variants that fail (boundary, ref mismatch) are skipped
#   with warnings — they don't crash the run.
#
# For CLI usage:
# ```bash
# python tools/score_variants.py path/to/model_dir --vcf variants.vcf.gz --output effects.tsv
# ```

# %%
