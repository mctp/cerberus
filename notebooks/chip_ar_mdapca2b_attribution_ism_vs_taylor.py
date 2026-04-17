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
import torch

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
model.eval()

# %% [markdown]
# ## 2. Pick a random test peak from the shipped fixture
#
# The BED fixture lists the exact intervals held out of training. No dataset
# / sampler / fold-split setup is needed — we only need the interval and its
# reference sequence for ISM, so we go through two library primitives:
#
# * :func:`cerberus.interval.load_intervals_bed` (gzip-aware) reads the
#   fixture directly.
# * :class:`cerberus.sequence.SequenceExtractor` gives us the one-hot the
#   model saw in training, straight from the FASTA referenced in
#   ``config.genome_config.fasta_path``.

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
target_model.eval()

# %% [markdown]
# ## 4. Extract the input one-hot and choose an ISM window
#
# :class:`SequenceExtractor` reads the FASTA referenced in the model's config
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

# %%
