# %% [markdown]
# # Chip-AR MDAPCA2b: Compare Pretrained BPNet and Pomeranian Models
#
# This notebook loads the two pretrained models shipped with the repository
# (BPNet and Pomeranian, both trained on the MDA-PCA-2b AR ChIP-seq dataset)
# and runs them on a fixed list of test intervals stored in
# `tests/data/fixtures/chip_ar_mdapca2b_intervals_test.bed.gz`.
#
# For each interval, we compare the model's predicted total log-counts against
# the observed total log-counts (from the target BigWig) and visualize the
# relationship as a scatter plot.

# %%
import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from paths import get_project_root
except ImportError:
    sys.path.append("notebooks")
    from paths import get_project_root

from cerberus.dataset import CerberusDataset
from cerberus.download import download_dataset, download_human_reference
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    compute_obs_total_log_counts,
    compute_total_log_counts,
    get_log_count_params,
)
from cerberus.utils import resolve_device

# %% [markdown]
# ## 1. Setup
#
# Locate the project root, download required reference data if needed, and
# pick a device. Both pretrained models share the same dataset so we only
# need to fetch it once.

# %%
project_root = get_project_root()
data_dir = project_root / "tests/data"
fixtures_dir = project_root / "tests/data/fixtures"

intervals_path = fixtures_dir / "chip_ar_mdapca2b_intervals_test.bed.gz"
if not intervals_path.exists():
    raise FileNotFoundError(f"Test intervals fixture not found: {intervals_path}")

device = resolve_device()
print(f"Using device: {device}")

print("Checking Data...")
download_human_reference(data_dir / "genome", name="hg38")
download_dataset(data_dir / "dataset", name="mdapca2b_ar")

# %% [markdown]
# ## 2. Load Test Intervals
#
# The fixture is a gzipped BED-like TSV written by
# `cerberus.interval.write_intervals_bed`. Columns are
# `chrom`, `start`, `end`, `strand`, `interval_source`.

# %%
test_intervals: list[Interval] = []
with gzip.open(intervals_path, "rt") as f:
    header = next(f)
    if not header.startswith("chrom\t"):
        raise ValueError(f"Unexpected header in {intervals_path}: {header!r}")
    for line in f:
        chrom, start, end, strand, _source = line.rstrip("\n").split("\t")
        test_intervals.append(Interval(chrom, int(start), int(end), strand))

print(f"Loaded {len(test_intervals)} test intervals")
print(f"Example interval: {test_intervals[0]} (length={len(test_intervals[0])})")

# %% [markdown]
# ## 3. Helper: Predict and Observe Log-Counts For One Model
#
# The two pretrained models have slightly different input lengths
# (BPNet: 2114, Pomeranian: 2112) so the same intervals must be center-cropped
# per model before inference. We loop over batches, recording both the
# model's predicted total log-counts and the observed total log-counts
# extracted from the target BigWig over the model's output window.


# %%
def predict_and_observe(
    checkpoint_dir: Path,
    intervals: list[Interval],
    device: torch.device | str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run an ensemble on `intervals` and return (predicted, observed, name).

    Both arrays are total log-counts per interval. Pseudocount handling is
    inferred from the model's loss class via ``get_log_count_params``.
    """
    print(f"\nLoading ensemble from {checkpoint_dir.relative_to(project_root)}")
    ensemble = ModelEnsemble(checkpoint_path=checkpoint_dir, device=device)
    config = ensemble.cerberus_config

    input_len = config.data_config.input_len
    output_len = config.data_config.output_len
    target_scale = config.data_config.target_scale
    name = config.model_config_.name
    log_inc_pc, pc = get_log_count_params(config.model_config_)
    print(
        f"  Model: {name} (input_len={input_len}, output_len={output_len}, "
        f"log_counts_include_pseudocount={log_inc_pc}, pseudocount={pc})"
    )

    # Center each fixture interval to this model's input window.
    centered = [iv.center(input_len) for iv in intervals]

    dataset = CerberusDataset(
        genome_config=config.genome_config,
        data_config=config.data_config,
        sampler_config=None,
        is_train=False,
    )
    if dataset.target_signal_extractor is None:
        raise RuntimeError("Target signal extractor is missing from the dataset.")

    pred_log_counts: list[float] = []
    obs_log_counts: list[float] = []

    for batched_output, batch_intervals in ensemble.predict_intervals_batched(
        centered,
        dataset,
        batch_size=batch_size,
    ):
        # Predicted total log-counts (B,)
        pred = compute_total_log_counts(
            batched_output,
            log_counts_include_pseudocount=log_inc_pc,
            pseudocount=pc,
        )
        pred_log_counts.extend(pred.detach().cpu().tolist())

        # Observed total log-counts: extract the raw target signal over the
        # model's output window (centered on the input interval) and sum.
        obs_signals = []
        for iv in batch_intervals:
            out_iv = iv.center(output_len)
            obs_signals.append(dataset.target_signal_extractor.extract(out_iv))
        raw = torch.stack(obs_signals)  # (B, C, output_len)
        obs = compute_obs_total_log_counts(
            raw,
            target_scale=target_scale,
            log_counts_include_pseudocount=log_inc_pc,
            pseudocount=pc,
        )
        obs_log_counts.extend(obs.detach().cpu().tolist())

    return np.asarray(pred_log_counts), np.asarray(obs_log_counts), name


# %% [markdown]
# ## 4. Run Inference for Both Models

# %%
bpnet_pred, bpnet_obs, bpnet_name = predict_and_observe(
    project_root / "pretrained/chip_ar_mdapca2b_bpnet",
    test_intervals,
    device=device,
)

pomeranian_pred, pomeranian_obs, pomeranian_name = predict_and_observe(
    project_root / "pretrained/chip_ar_mdapca2b_pomeranian",
    test_intervals,
    device=device,
)

print("\nFinished inference.")
print(
    f"  {bpnet_name}: {len(bpnet_pred)} intervals, "
    f"pred mean={bpnet_pred.mean():.3f}, obs mean={bpnet_obs.mean():.3f}"
)
print(
    f"  {pomeranian_name}: {len(pomeranian_pred)} intervals, "
    f"pred mean={pomeranian_pred.mean():.3f}, obs mean={pomeranian_obs.mean():.3f}"
)

# %% [markdown]
# ## 5. Scatter Plot — Predicted vs Observed Log-Counts
#
# One panel per model. Pearson correlation is annotated on each panel.
# The dashed red line is the y=x identity reference.


# %%
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


bpnet_r = _pearson(bpnet_pred, bpnet_obs)
pomeranian_r = _pearson(pomeranian_pred, pomeranian_obs)
cross_r = _pearson(bpnet_pred, pomeranian_pred)

print(f"Pearson R ({bpnet_name} pred vs obs):      {bpnet_r:.4f}")
print(f"Pearson R ({pomeranian_name} pred vs obs): {pomeranian_r:.4f}")
print(f"Pearson R ({bpnet_name} pred vs {pomeranian_name} pred): {cross_r:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Shared axis limits across all panels for fair visual comparison.
AXIS_LIM = (5.0, 10.0)

# Panel 1: BPNet pred vs obs
axes[0].scatter(bpnet_obs, bpnet_pred, alpha=0.1, s=4, color="#1f77b4")
axes[0].plot(AXIS_LIM, AXIS_LIM, "r--", linewidth=1, label="y = x")
axes[0].set_xlabel("Observed log-counts")
axes[0].set_ylabel("Predicted log-counts")
axes[0].set_title(f"{bpnet_name}\nPearson R = {bpnet_r:.4f}")
axes[0].legend(loc="upper left")

# Panel 2: Pomeranian pred vs obs
axes[1].scatter(pomeranian_obs, pomeranian_pred, alpha=0.1, s=4, color="#2ca02c")
axes[1].plot(AXIS_LIM, AXIS_LIM, "r--", linewidth=1, label="y = x")
axes[1].set_xlabel("Observed log-counts")
axes[1].set_ylabel("Predicted log-counts")
axes[1].set_title(f"{pomeranian_name}\nPearson R = {pomeranian_r:.4f}")
axes[1].legend(loc="upper left")

# Panel 3: Model agreement (BPNet vs Pomeranian predictions)
axes[2].scatter(bpnet_pred, pomeranian_pred, alpha=0.1, s=4, color="#9467bd")
axes[2].plot(AXIS_LIM, AXIS_LIM, "r--", linewidth=1, label="y = x")
axes[2].set_xlabel(f"{bpnet_name} predicted log-counts")
axes[2].set_ylabel(f"{pomeranian_name} predicted log-counts")
axes[2].set_title(f"{bpnet_name} vs {pomeranian_name}\nPearson R = {cross_r:.4f}")
axes[2].legend(loc="upper left")

for ax in axes:
    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

plots_dir = project_root / "notebooks/plots"
plots_dir.mkdir(exist_ok=True, parents=True)
out_path = plots_dir / "chip_ar_mdapca2b_compare_models.png"
plt.savefig(out_path, dpi=120)
print(f"\nSaved scatter plot to {out_path.relative_to(project_root)}")

# %%
