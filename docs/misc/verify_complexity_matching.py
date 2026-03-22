import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cerberus.complexity import compute_intervals_complexity
from cerberus.samplers import ComplexityMatchedSampler, IntervalSampler, RandomSampler


def load_chrom_sizes(fai_path):
    chrom_sizes = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.split("\t")
            chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def main():
    root = Path(__file__).resolve().parents[2]
    data_root = root / "tests/data"

    # Paths
    targets_path = data_root / "dataset/mdapca2b_ar/mdapca2b-ar.narrowPeak.gz"
    fasta_path = data_root / "genome/hg38/hg38.fa"
    fai_path = data_root / "genome/hg38/hg38.fa.fai"
    output_plot = root / "docs/misc/complexity_matching_results.png"

    if not fasta_path.exists():
        print(f"Error: FASTA not found at {fasta_path}")
        return

    print("Loading resources...")
    chrom_sizes = load_chrom_sizes(fai_path)

    # 1. Load Targets
    print("Loading targets...")
    # Using padded_size=2114 (standard BPNet input size)
    padded_size = 2114
    targets = IntervalSampler(
        file_path=targets_path,
        chrom_sizes=chrom_sizes,
        padded_size=padded_size,
        folds=None,
        exclude_intervals=None,
    )
    print(f"Loaded {len(targets)} targets.")

    # 2. Generate Random Candidates
    print("Generating random candidates...")
    # Generate 10x candidates to ensure good matching
    n_candidates = len(targets) * 10
    # Cap candidates if too many (for speed)
    if n_candidates > int(1e6):
        n_candidates = int(1e6)

    candidates = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=padded_size,
        num_intervals=n_candidates,
        folds=None,
        exclude_intervals=None,
        seed=42,
    )
    print(f"Generated {len(candidates)} candidates.")

    # 3. Match
    print("Matching candidates...")
    matched_sampler = ComplexityMatchedSampler(
        target_sampler=targets,
        candidate_sampler=candidates,
        fasta_path=fasta_path,
        chrom_sizes=chrom_sizes,
        bins=20,
        candidate_ratio=1.0,
        seed=42,
    )
    print(f"Selected {len(matched_sampler)} matched intervals.")

    # 4. Compute Metrics
    print("Computing metrics (this may take a moment)...")
    target_metrics = compute_intervals_complexity(targets, fasta_path)
    candidate_metrics = compute_intervals_complexity(candidates, fasta_path)
    matched_metrics = compute_intervals_complexity(matched_sampler, fasta_path)

    # Debug Stats
    def print_stats(name, metrics):
        # Filter NaNs
        valid = metrics[~np.isnan(metrics).any(axis=1)]
        print(f"\nStats for {name} (N={len(valid)}):")
        print(f"  GC:   mean={np.mean(valid[:, 0]):.3f}, std={np.std(valid[:, 0]):.3f}")
        print(f"  Dust: mean={np.mean(valid[:, 1]):.3f}, std={np.std(valid[:, 1]):.3f}")
        print(f"  CpG:  mean={np.mean(valid[:, 2]):.3f}, std={np.std(valid[:, 2]):.3f}")

    print_stats("Targets", target_metrics)
    print_stats("Candidates", candidate_metrics)
    print_stats("Matched", matched_metrics)

    # 5. Prepare Dataframe
    df_list = []

    # Limit number of random points plotted to avoid overplotting/slowness
    # But for distribution matching check, we want to see density.
    # We'll plot max 2000 randoms, all targets, all matched.

    def add_data(metrics, label, limit=None):
        count = 0
        indices = np.arange(len(metrics))
        if limit and len(metrics) > limit:
            np.random.shuffle(indices)
            indices = indices[:limit]

        for i in indices:
            row = metrics[i]
            if not np.isnan(row).any():
                df_list.append(
                    {
                        "GC Content": row[0],
                        "Normalized Dust Score": row[1],
                        "Normalized CpG Score": row[2],
                        "Region Type": label,
                    }
                )
                count += 1
        return count

    n_t = add_data(target_metrics, "Target", limit=10000)
    n_c = add_data(candidate_metrics, "Random Background", limit=10000)
    n_m = add_data(matched_metrics, "Matched Candidates", limit=10000)

    print(f"Plotting {n_t} targets, {n_c} background, {n_m} matched points.")

    df = pd.DataFrame(df_list)

    # 6. Plot
    print("Generating separated plots...")
    sns.set_theme(style="whitegrid")

    # 3 types x 3 pairs = 9 plots
    types = ["Target", "Random Background", "Matched Candidates"]
    # Cycle pairs to ensure all metrics appear on X axis
    pairs = [
        ("GC Content", "Normalized Dust Score"),
        ("Normalized Dust Score", "Normalized CpG Score"),
        ("Normalized CpG Score", "GC Content"),
    ]
    colors = {
        "Target": "blue",
        "Random Background": "gray",
        "Matched Candidates": "orange",
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # Adjust layout to make room for row labels
    plt.subplots_adjust(left=0.15)

    # We iterate rows (Types) and columns (Pairs)
    for i, t in enumerate(types):
        subset = df[df["Region Type"] == t]
        color = colors[t]

        for j, (x_var, y_var) in enumerate(pairs):
            ax = axes[i, j]

            # Scatter with very low alpha
            sns.scatterplot(
                data=subset,
                x=x_var,
                y=y_var,
                ax=ax,
                color=color,
                alpha=0.1,
                s=5,
                edgecolor=None,
            )

            # Add KDE density contours on top to see density
            try:
                sns.kdeplot(
                    data=subset,
                    x=x_var,
                    y=y_var,
                    ax=ax,
                    color=color,
                    alpha=1.0,
                    levels=5,
                    linewidths=1.0,
                )
            except Exception:
                # Fallback if KDE fails (e.g. too few points or singular)
                pass

            # Labels
            if i == 2:
                ax.set_xlabel(x_var)
            else:
                ax.set_xlabel("")

            # Y-Label logic: simple variable name
            ax.set_ylabel(y_var, fontsize=10)

            # Row Label on the left of the first column
            if j == 0:
                # Place text to the left of the axis
                ax.text(
                    -0.35,
                    0.5,
                    t,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )

            # Limits (metrics are 0-1)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

    plt.suptitle(
        f"Complexity Matching Verification\n(Targets: {len(targets)}, Candidates Pool: {len(candidates)})",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    main()
