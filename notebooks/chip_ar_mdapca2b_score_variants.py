# %% [markdown]
# # Variant Effect Scoring Demo (BPNet)
#
# This notebook demonstrates how to score the predicted effect of genomic
# variants using a trained BPNet model and cerberus's variant scoring pipeline.
#
# **Prerequisites:** Train the BPNet model first:
# ```bash
# bash examples/chip_ar_mdapca2b_bpnet.sh
# ```

# %%
import sys
from pathlib import Path

try:
    from paths import get_project_root
except ImportError:
    sys.path.append("notebooks")
    from paths import get_project_root

import pyfaidx

from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import get_log_count_params
from cerberus.predict_variants import score_variants, score_variants_from_ensemble
from cerberus.utils import resolve_device
from cerberus.variants import Variant

# %% [markdown]
# ## 1. Load Model
#
# `ModelEnsemble` discovers `hparams.yaml` in the checkpoint directory and
# reconstructs the full config (genome, data, model) automatically.  No
# manual config reconstruction is needed.
#
# If the model was trained on a different machine and the stored paths
# (FASTA, BigWig) don't resolve, you can override individual configs:
# ```python
# ensemble = ModelEnsemble(
#     checkpoint_dir,
#     genome_config=my_genome_config,  # override FASTA path
#     data_config=my_data_config,      # override BigWig paths
#     device=device,
# )
# ```

# %%
project_root = get_project_root()
checkpoint_dir = project_root / "tests/data/models/chip_ar_mdapca2b_bpnet/single-fold"

if not checkpoint_dir.exists():
    print(f"Checkpoint directory not found: {checkpoint_dir}")
    print("Run 'bash examples/chip_ar_mdapca2b_bpnet.sh' to train the model first.")
    sys.exit(0)

device = resolve_device()
print(f"Using device: {device}")

ensemble = ModelEnsemble(checkpoint_dir, device=device)
config = ensemble.cerberus_config

print(f"Model: {config.model_config_.name}")
print(f"Input length: {config.data_config.input_len}")
print(f"FASTA: {config.genome_config.fasta_path}")
print(f"Loaded folds: {list(ensemble.keys())}")

# %% [markdown]
# ## 2. Define Variants
#
# We score a mix of variants: synthetic SNPs at known AR peak summits
# (from the training narrowPeak file), a known KLK3 missense variant,
# and an intergenic control.  In a real analysis you would load from
# a VCF file (see Section 5).

# %%
# Peak 1: chr6:93940029-93941032, summit offset 538 -> summit at 93940567
# Peak 3: chr11:114178889-114180869, summit offset 586 -> summit at 114179475
# Ref alleles verified against hg38.fa.

variants = [
    # SNP at peak 1 summit
    Variant("chr6", 93940567, "C", "T", id="peak1_summit_snp"),
    # SNP 100bp upstream of peak 1 summit
    Variant("chr6", 93940467, "A", "G", id="peak1_upstream_snp"),
    # SNP at peak 3 summit
    Variant("chr11", 114179475, "C", "A", id="peak3_summit_snp"),
    # KLK3 missense variant (19:50858501 T>C, 0-based pos=50858500)
    Variant("chr19", 50858500, "T", "C", id="KLK3_missense"),
    # MSMB intergenic variant (10:46046326 A>G, 0-based pos=46046325)
    Variant("chr10", 46046325, "A", "G", id="MSMB_intergenic"),
    # SNP far from any peak (control -- should have minimal effect)
    Variant("chr1", 10000000, "A", "C", id="intergenic_control"),
]

print(f"Scoring {len(variants)} variants:")
for v in variants:
    print(f"  {v} ({v.id})")

# %% [markdown]
# ## 3. Score Variants (High-Level API)
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
# ## 4. Score Variants (Low-Level API)
#
# `score_variants` gives you more control -- you pass the model, FASTA,
# input length, and pseudocount parameters explicitly. This is useful when
# working with a plain `nn.Module` (not an ensemble) or when you need
# custom parameters.

# %%
fasta = pyfaidx.Fasta(str(config.genome_config.fasta_path))
lc_include, pseudocount = get_log_count_params(config.model_config_)

# Score just one variant to show the full VariantResult
single_variant = [Variant("chr6", 93940567, "C", "T", id="peak1_summit")]
result = next(score_variants(
    model=ensemble,
    variants=single_variant,
    fasta=fasta,
    input_len=config.data_config.input_len,
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
# ## 5. Score from a VCF File
#
# In a real analysis, load variants from a VCF:
# ```python
# from cerberus.variants import load_vcf
#
# variants = load_vcf("gwas_hits.vcf.gz", region="chr6:93000000-94000000")
# results = list(score_variants_from_ensemble(ensemble, variants))
# ```
#
# Or from the CLI:
# ```bash
# python tools/score_variants.py path/to/model_dir --vcf gwas_hits.vcf.gz --output effects.tsv
# ```

# %% [markdown]
# ## 6. Saturation Mutagenesis
#
# `generate_variants` produces every possible SNV within a genomic interval.
# Combined with `score_variants_from_ensemble`, this gives a complete variant
# effect map — the model's predicted impact of every single-nucleotide change
# across a region.
#
# We scan a 356bp region on chr21 (chr21:41560339-41560695), generating
# 356 x 3 = 1068 SNVs and scoring them all in one call.

# %%
from cerberus.interval import Interval
from cerberus.variants import generate_variants

# region = Interval("chr21", 41560339, 41560695)
# region = Interval("chr21", 41501137, 41501151)
# chr21:41499923-41499938
region = Interval("chr21", 41499923, 41499938)
sat_variants = generate_variants(region, pyfaidx.Fasta(str(config.genome_config.fasta_path)))

sat_results = list(score_variants_from_ensemble(
    ensemble=ensemble,
    variants=sat_variants,
    batch_size=128,
))
print(f"Scored {len(sat_results)} SNVs across {region}")

# %%
# Build a summary table: position, ref, alt, SAD, log_fc
import csv
import io

buf = io.StringIO()
writer = csv.writer(buf, delimiter="\t")
writer.writerow(["chrom", "pos", "ref", "alt", "sad", "log_fc", "jsd"])
for r in sat_results:
    e = r.effects
    writer.writerow([
        r.variant.chrom, r.variant.pos, r.variant.ref, r.variant.alt,
        f"{e['sad'].item():.4f}",
        f"{e['log_fc'].item():.6f}",
        f"{e['jsd'].item():.8f}",
    ])

print(buf.getvalue()[:500])
print(f"... ({len(sat_results)} rows total)")

# %%
# Find the top 10 variants by SAD (largest predicted effect)
sorted_results = sorted(sat_results, key=lambda r: r.effects["sad"].item(), reverse=True)

print(f"\nTop 10 variants by SAD in {region}:\n")
print(f"{'pos':>12} {'ref':>4} {'alt':>4} {'SAD':>10} {'log_fc':>10} {'JSD':>12}")
print("-" * 58)
for r in sorted_results[:10]:
    e = r.effects
    print(
        f"{r.variant.pos:>12} "
        f"{r.variant.ref:>4} "
        f"{r.variant.alt:>4} "
        f"{e['sad'].item():>10.2f} "
        f"{e['log_fc'].item():>10.4f} "
        f"{e['jsd'].item():>12.8f}"
    )

# %%
# Per-position max effect: for each reference base (in sequence order),
# show the alt allele that causes the largest SAD.
max_by_pos: dict[int, tuple[float, float, str, str]] = {}  # pos → (sad, log_fc, ref, alt)
for r in sat_results:
    sad = r.effects["sad"].item()
    log_fc = r.effects["log_fc"].item()
    pos = r.variant.pos
    if pos not in max_by_pos or sad > max_by_pos[pos][0]:
        max_by_pos[pos] = (sad, log_fc, r.variant.ref, r.variant.alt)

print(f"\nMax effect per position in {region}:\n")
print(f"{'pos':>12} {'ref':>4} {'worst_alt':>10} {'max_SAD':>10} {'log_fc':>10}")
print("-" * 52)
for pos in sorted(max_by_pos):
    sad, log_fc, ref, alt = max_by_pos[pos]
    print(f"{pos:>12} {ref:>4} {alt:>10} {sad:>10.2f} {log_fc:>10.4f}")

# %% [markdown]
# ## Summary
#
# | Function | Use case |
# |---|---|
# | `score_variants_from_ensemble()` | High-level: pass ensemble + variants, config auto-detected |
# | `score_variants()` | Low-level: explicit model, FASTA, input_len, pseudocount |
# | `ModelEnsemble(checkpoint_dir)` | Loads model + full config from `hparams.yaml` |
#
# **`VariantResult`** contains `.variant`, `.effects` (dict of per-channel tensors),
# and `.interval` (the genomic window used for scoring).
#
# **Effect metrics**: `sad`, `log_fc`, `jsd`, `pearson`, `max_abs_diff`
# (plus `signal_*` for Dalmatian models).
#
# **CLI usage**:
# ```bash
# python tools/score_variants.py path/to/model_dir --vcf variants.vcf.gz --output effects.tsv
# ```

# %%
