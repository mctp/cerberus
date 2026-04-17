# %% [markdown]
# # ISM vs Taylor-ISM on a random test peak (BPNet, AR ChIP-seq)
#
# This notebook loads the pretrained BPNet model for AR ChIP-seq on MDA-PCA-2b
# cells, picks one test-fold peak at random, and computes per-nucleotide
# attributions using:
#
# 1. **Exact ISM** — brute-force `3 * L_span` forward passes.
# 2. **Taylor-ISM (TISM)** — first-order approximation, one forward + one
#    backward pass (Sasse et al. 2024, *iScience*).
#
# We then visualise both with the `cerberus.plots` sequence-logo helpers and
# check that TISM is a faithful approximation by computing the Pearson
# correlation between the two attribution maps over the same span.
#
# **Attribution target**: the model's predicted **log total counts** —
# i.e. which bases contribute to the overall AR binding strength at the peak.

# %%
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from paths import get_project_root
except ImportError:
    sys.path.append("notebooks")
    from paths import get_project_root

from cerberus.attribution import (
    AttributionTarget,
    compute_ism_attributions,
    compute_taylor_ism_attributions,
)
from cerberus.interval import load_intervals_bed
from cerberus.model_ensemble import ModelEnsemble
from cerberus.plots import plot_attribution_panel, plot_seqlogo
from cerberus.sequence import SequenceExtractor
from cerberus.utils import resolve_device

# %% [markdown]
# ## 1. Load the pretrained BPNet ensemble

# %%
project_root = get_project_root()
checkpoint_dir = project_root / "pretrained/chip_ar_mdapca2b_bpnet"
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Pretrained model not found: {checkpoint_dir}")

device = resolve_device()
print(f"Using device: {device}")

model_ensemble = ModelEnsemble(checkpoint_path=checkpoint_dir, device=device)
config = model_ensemble.cerberus_config
input_len = config.data_config.input_len
print(f"Model: {config.model_config_.name}  (input_len = {input_len})")

# Extract the single-fold model from the ensemble for attribution. ISM / TISM
# need a plain ``nn.Module`` taking ``(B, C, L)`` → ``ProfileCountOutput``,
# which every fold model satisfies.
fold_keys = list(model_ensemble.keys())
print(f"Ensemble folds: {fold_keys}")
model = model_ensemble[fold_keys[0]].to(device)
model.eval();  # noqa: E702 — suppress repr in Jupyter

# %% [markdown]
# ## 2. Pick a random test peak from the shipped fixture
#
# The BED fixture lists the exact intervals held out of training. No dataset
# / sampler / fold-split setup is needed — we only need the interval and its
# reference sequence for ISM, so we go through two library primitives:
#
# * `cerberus.interval.load_intervals_bed` (gzip-aware) reads the
#   fixture directly.
# * `cerberus.sequence.SequenceExtractor` gives us the one-hot the
#   model saw in training, straight from the FASTA referenced in
#   `config.genome_config.fasta_path`.

# %%
fixture_path = (
    project_root / "tests/data/fixtures/chip_ar_mdapca2b_intervals_test.bed.gz"
)
test_intervals, _ = load_intervals_bed(fixture_path)
print(f"Fixture intervals: {len(test_intervals)}")

# Pick one peak at random — seeded for reproducibility. Re-run with a
# different ``PEAK_SEED`` to study a different locus.
PEAK_SEED = 42
rng = np.random.default_rng(PEAK_SEED)
peak_idx = int(rng.integers(0, len(test_intervals)))
peak_interval = test_intervals[peak_idx].center(input_len)
print(f"Random peak (seed={PEAK_SEED}, idx={peak_idx}): {peak_interval}")

# %% [markdown]
# ## 3. Build the scalar attribution target
#
# We wrap the model in ``AttributionTarget(reduction="log_counts")`` so that
# both ISM and TISM differentiate a single scalar per batch element — the
# model's predicted log total counts. Channel 0 is the only BPNet output head
# in this checkpoint.

# %%
target_model = AttributionTarget(
    model=model,
    reduction="log_counts",
    channel=0,
    bin_index=None,
    window_start=None,
    window_end=None,
).to(device)
target_model.eval();  # noqa: E702 — suppress repr in Jupyter

# %% [markdown]
# ## 4. Extract the input one-hot and choose an ISM window
#
# `SequenceExtractor` reads the FASTA referenced in the model's config
# and returns ``(4, L)``; we add a batch dim. Exact ISM is O(L_span) forward
# passes, so we restrict the window to the central 200 bp of the peak to keep
# the notebook interactive. TISM runs on the same window for a fair
# comparison — though in practice it can cover the full 2114 bp in the same
# wall-clock time.

# %%
extractor = SequenceExtractor(config.genome_config.fasta_path)
inputs = extractor.extract(peak_interval).unsqueeze(0).to(device)  # (1, 4, L)
print(f"Input shape: {tuple(inputs.shape)}  dtype: {inputs.dtype}")

SPAN_WIDTH = 200
center = input_len // 2
span = (center - SPAN_WIDTH // 2, center + SPAN_WIDTH // 2)
print(f"ISM span: {span}  ({SPAN_WIDTH} bp around center)")

# %% [markdown]
# ## 5. Run exact ISM

# %%
t0 = time.perf_counter()
ism_attrs = compute_ism_attributions(target_model, inputs, span=span)
ism_elapsed = time.perf_counter() - t0
print(f"Exact ISM:     {ism_elapsed:7.2f} s  →  shape {tuple(ism_attrs.shape)}")

# %% [markdown]
# ## 6. Run Taylor-ISM

# %%
t0 = time.perf_counter()
tism_attrs = compute_taylor_ism_attributions(target_model, inputs, span=span)
tism_elapsed = time.perf_counter() - t0
print(f"Taylor-ISM:    {tism_elapsed:7.2f} s  →  shape {tuple(tism_attrs.shape)}")
print(f"Speedup: {ism_elapsed / max(tism_elapsed, 1e-6):.1f}×")

# %% [markdown]
# ## 7. How well does TISM match exact ISM?
#
# Compute the Pearson correlation over all in-span elements of the two
# ``(4, L)`` attribution matrices. Sasse et al. 2024 report a mean Pearson of
# ~0.7 across thousands of peaks; individual peaks vary from ~0.5 to >0.9
# depending on how non-linear the model is at that locus.

# %%
span_start, span_end = span
ism_flat = ism_attrs[0, :, span_start:span_end].detach().cpu().numpy().ravel()
tism_flat = tism_attrs[0, :, span_start:span_end].detach().cpu().numpy().ravel()
pearson_r = float(np.corrcoef(ism_flat, tism_flat)[0, 1])
print(f"Pearson R (ISM vs TISM, in-span elements): {pearson_r:.4f}")

# %% [markdown]
# ## 8. Visualise both attribution maps

# %%
plots_dir = project_root / "notebooks/plots"
plots_dir.mkdir(exist_ok=True, parents=True)

ism_np = ism_attrs[0, :, span_start:span_end].detach().cpu().numpy()
tism_np = tism_attrs[0, :, span_start:span_end].detach().cpu().numpy()

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 1, hspace=0.35)

subfig_ism = fig.add_subfigure(gs[0])
ism_logo_ax, ism_heat_ax = plot_attribution_panel(subfig_ism, ism_np)
ism_logo_ax.set_title(
    f"Exact ISM — {peak_interval}  ({SPAN_WIDTH} bp)   {ism_elapsed:.2f} s"
)

subfig_tism = fig.add_subfigure(gs[1])
tism_logo_ax, tism_heat_ax = plot_attribution_panel(subfig_tism, tism_np)
tism_logo_ax.set_title(
    f"Taylor-ISM (first-order)   {tism_elapsed:.2f} s   "
    f"(Pearson R vs ISM = {pearson_r:.3f})"
)

out_path = plots_dir / "chip_ar_mdapca2b_attribution_ism_vs_taylor.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path.relative_to(project_root)}")

# %% [markdown]
# ## 9. Element-wise ISM vs TISM scatter
#
# Every point is one (base, position) attribution element. Perfect agreement
# lies on the dashed identity line; first-order Taylor error is the deviation
# from it. The coloured contours would be denser toward the origin where both
# methods agree on "nothing interesting is happening here".

# %%
fig2, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(ism_flat, tism_flat, alpha=0.25, s=6, color="#1f77b4")
lim = float(np.max(np.abs(np.concatenate([ism_flat, tism_flat])))) * 1.05
ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y = x")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Exact ISM attribution")
ax.set_ylabel("Taylor-ISM attribution")
ax.set_title(f"ISM vs Taylor-ISM  (Pearson R = {pearson_r:.3f})")
ax.legend(loc="upper left")

scatter_path = plots_dir / "chip_ar_mdapca2b_attribution_ism_vs_taylor_scatter.png"
fig2.savefig(scatter_path, dpi=120, bbox_inches="tight")
print(f"Saved: {scatter_path.relative_to(project_root)}")

# %% [markdown]
# ## 10. IC-mode logo of Taylor-ISM at full sequence length
#
# Because TISM needs only one forward + one backward, we can cheaply run it on
# the entire 2114 bp input — not just the 200 bp window. Visualised as an
# information-content logo, peaks indicate positions where any base carries
# high-magnitude attribution.

# %%
t0 = time.perf_counter()
tism_full = compute_taylor_ism_attributions(
    target_model, inputs, span=(None, None)
)
full_elapsed = time.perf_counter() - t0
print(
    f"Taylor-ISM full-length ({input_len} bp):  {full_elapsed:.2f} s  "
    f"vs estimated exact ISM ~{ism_elapsed * input_len / SPAN_WIDTH:.0f} s"
)

tism_full_np = tism_full[0].detach().cpu().numpy()

fig3, ax3 = plt.subplots(figsize=(18, 2.2))
plot_seqlogo(ax3, tism_full_np, mode="ic")
ax3.set_title(
    f"Taylor-ISM full-length IC logo — {peak_interval}  ({input_len} bp)"
)
ax3.set_ylabel("IC (bits)")

full_path = plots_dir / "chip_ar_mdapca2b_attribution_tism_full_length.png"
fig3.savefig(full_path, dpi=120, bbox_inches="tight")
print(f"Saved: {full_path.relative_to(project_root)}")

# %% [markdown]
# ## 11. Cross-architecture: BPNet TISM vs Pomeranian TISM on the same peak
#
# Sasse et al. 2024 report ISM↔ISM correlation between two re-initialised
# instances of the *same* architecture is ~0.58 — model-to-model variation
# in attributions is real and substantial. Comparing TISM across two
# *different* architectures (BPNet vs Pomeranian, both trained on the same
# AR-ChIP data) is a stronger test of which features are intrinsic to the
# data and which are model-specific.
#
# Pomeranian uses a slightly shorter input window (2112 bp vs BPNet's 2114),
# so we re-center the same fixture peak to its native input length. Both
# spans cover the same 200 bp of genomic sequence so the in-span TISM tensors
# can be compared element-wise.

# %%
pomeranian_dir = project_root / "pretrained/chip_ar_mdapca2b_pomeranian"
if not pomeranian_dir.exists():
    raise FileNotFoundError(f"Pretrained Pomeranian not found: {pomeranian_dir}")

pomeranian_ensemble = ModelEnsemble(checkpoint_path=pomeranian_dir, device=device)
pomeranian_config = pomeranian_ensemble.cerberus_config
pomeranian_input_len = pomeranian_config.data_config.input_len
print(
    f"Model: {pomeranian_config.model_config_.name}  "
    f"(input_len = {pomeranian_input_len})"
)

pom_fold_keys = list(pomeranian_ensemble.keys())
pomeranian_model = pomeranian_ensemble[pom_fold_keys[0]].to(device)
pomeranian_model.eval();  # noqa: E702 — suppress repr in Jupyter

pomeranian_target = AttributionTarget(
    model=pomeranian_model,
    reduction="log_counts",
    channel=0,
    bin_index=None,
    window_start=None,
    window_end=None,
).to(device)
pomeranian_target.eval();  # noqa: E702 — suppress repr in Jupyter

# %%
# Re-extract the SAME genomic peak at Pomeranian's input length, then run
# TISM over the central 200 bp (same genomic window as the BPNet run above).
pom_peak_interval = test_intervals[peak_idx].center(pomeranian_input_len)
pom_extractor = SequenceExtractor(pomeranian_config.genome_config.fasta_path)
pom_inputs = pom_extractor.extract(pom_peak_interval).unsqueeze(0).to(device)

pom_center = pomeranian_input_len // 2
pom_span = (pom_center - SPAN_WIDTH // 2, pom_center + SPAN_WIDTH // 2)

t0 = time.perf_counter()
pom_tism = compute_taylor_ism_attributions(
    pomeranian_target, pom_inputs, span=pom_span
)
pom_elapsed = time.perf_counter() - t0
print(
    f"Pomeranian TISM ({SPAN_WIDTH} bp):  {pom_elapsed:.2f} s   "
    f"BPNet TISM ({SPAN_WIDTH} bp): {tism_elapsed:.2f} s"
)

# %% [markdown]
# ### Element-wise correlation across architectures
#
# Both arrays are ``(4, 200)`` and aligned to the same 200 bp of genome.

# %%
bpnet_tism_np = tism_attrs[0, :, span_start:span_end].detach().cpu().numpy()
pom_tism_np = pom_tism[0, :, pom_span[0] : pom_span[1]].detach().cpu().numpy()
assert bpnet_tism_np.shape == pom_tism_np.shape, (
    f"shape mismatch: bpnet {bpnet_tism_np.shape} vs pom {pom_tism_np.shape}"
)

cross_pearson = float(
    np.corrcoef(bpnet_tism_np.ravel(), pom_tism_np.ravel())[0, 1]
)
print(f"Pearson R (BPNet TISM vs Pomeranian TISM): {cross_pearson:.4f}")
print(f"Pearson R (BPNet TISM vs BPNet ISM)      : {pearson_r:.4f}  (reference)")

# %% [markdown]
# ### Side-by-side cross-architecture logo + heatmap

# %%
fig4 = plt.figure(figsize=(14, 7))
gs4 = fig4.add_gridspec(2, 1, hspace=0.35)

sub_bp = fig4.add_subfigure(gs4[0])
ax_bp_logo, _ = plot_attribution_panel(sub_bp, bpnet_tism_np)
ax_bp_logo.set_title(
    f"BPNet  TISM   {tism_elapsed:.2f} s   "
    f"({peak_interval})"
)

sub_pom = fig4.add_subfigure(gs4[1])
ax_pom_logo, _ = plot_attribution_panel(sub_pom, pom_tism_np)
ax_pom_logo.set_title(
    f"Pomeranian  TISM   {pom_elapsed:.2f} s   "
    f"(Pearson R vs BPNet TISM = {cross_pearson:.3f})"
)

cross_path = plots_dir / "chip_ar_mdapca2b_attribution_tism_cross_architecture.png"
fig4.savefig(cross_path, dpi=120, bbox_inches="tight")
print(f"Saved: {cross_path.relative_to(project_root)}")

# %% [markdown]
# ### Cross-architecture scatter
#
# As with the within-model ISM-vs-TISM scatter, perfect agreement lies on the
# dashed identity line. Comparing the spread here against the within-model
# spread tells you whether your candidate motifs are robust across architectures
# or merely artifacts of one model's inductive bias.

# %%
fig5, ax5 = plt.subplots(figsize=(5.5, 5.5))
bp_flat = bpnet_tism_np.ravel()
pom_flat = pom_tism_np.ravel()
ax5.scatter(bp_flat, pom_flat, alpha=0.25, s=6, color="#9467bd")
lim = float(np.max(np.abs(np.concatenate([bp_flat, pom_flat])))) * 1.05
ax5.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y = x")
ax5.set_xlim(-lim, lim)
ax5.set_ylim(-lim, lim)
ax5.set_aspect("equal", adjustable="box")
ax5.set_xlabel("BPNet TISM")
ax5.set_ylabel("Pomeranian TISM")
ax5.set_title(
    f"BPNet vs Pomeranian TISM  (Pearson R = {cross_pearson:.3f})\n"
    f"vs within-BPNet ISM↔TISM R = {pearson_r:.3f}"
)
ax5.legend(loc="upper left")

cross_scatter_path = (
    plots_dir / "chip_ar_mdapca2b_attribution_tism_cross_architecture_scatter.png"
)
fig5.savefig(cross_scatter_path, dpi=120, bbox_inches="tight")
print(f"Saved: {cross_scatter_path.relative_to(project_root)}")

# %%
