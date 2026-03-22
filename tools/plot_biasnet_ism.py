#!/usr/bin/env python
"""Plot BiasNet ISM motif as IC logo + heatmap.

Loads a trained BiasNet from a run directory (containing hparams.yaml + model.pt)
or a standalone model.pt file, computes in-silico mutagenesis (ISM) over real
genomic sequences, and produces:
  1. A PNG plot with IC logo (top) and ISM delta heatmap (bottom), X-aligned.
  2. A CSV file with the raw ISM matrix for downstream analysis (e.g., in R).

Usage:
    python tools/plot_biasnet_ism.py <run_dir_or_model_pt> \\
        --fasta tests/data/genome/hg38/hg38.fa \\
        --peaks tests/data/k562_chrombpnet/peaks.bed \\
        [--output-dir plots/] [--prefix biasnet] \\
        [--n-seqs 1000] [--ism-window 31] [--device cuda]
"""

import argparse
import csv
import gzip
import logging
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pyfaidx
import torch
import yaml

logger = logging.getLogger(__name__)

NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_fold_dir(path: Path) -> tuple[Path, Path]:
    """Resolve a model path to (fold_dir, model.pt)."""
    if path.is_file() and path.suffix == ".pt":
        return path.parent, path
    elif (path / "single-fold" / "fold_0" / "model.pt").exists():
        fold_dir = path / "single-fold" / "fold_0"
        return fold_dir, fold_dir / "model.pt"
    elif (path / "model.pt").exists():
        return path, path / "model.pt"
    else:
        raise FileNotFoundError(
            f"Cannot find model.pt in {path}. Expected a run directory or .pt file."
        )


def load_config(path: Path) -> dict:
    """Load hparams.yaml from a run directory or model.pt sibling."""
    fold_dir, _ = _resolve_fold_dir(path)
    # Search for hparams.yaml in lightning_logs subdirectories
    candidates = list(fold_dir.rglob("hparams.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No hparams.yaml found in {fold_dir}")
    hparams_path = sorted(candidates)[-1]  # latest version
    with open(hparams_path) as f:
        return yaml.safe_load(f)


def _extract_bias_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract bias_model state dict from a Dalmatian checkpoint.

    Strips the ``bias_model.`` prefix from keys. If the state dict already
    contains bare BiasNet keys (no prefix), returns it unchanged.
    """
    prefix = "bias_model."
    bias_keys = [k for k in sd if k.startswith(prefix)]
    if bias_keys:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    # Already a standalone BiasNet state dict
    return sd


def load_biasnet(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a BiasNet model from a run directory or standalone .pt file.

    Supports:
      - Standalone BiasNet: hparams has ``model_config.name == "BiasNet"``
      - Dalmatian: hparams has ``model_config.name == "Dalmatian"``;
        extracts the ``bias_model`` subnetwork from the Dalmatian state dict.

    In both cases the returned model is a standalone :class:`BiasNet` instance.
    """
    from cerberus.models.biasnet import BiasNet
    from cerberus.models.dalmatian import _compute_shrinkage

    path = Path(path)
    _, model_pt = _resolve_fold_dir(path)

    config = load_config(path)
    model_name = config["model_config"]["name"]
    model_args = config["model_config"]["model_args"]
    data_config = config["data_config"]
    output_len = data_config["output_len"]
    output_bin_size = data_config["output_bin_size"]

    if model_name == "BiasNet":
        # Standalone BiasNet — all params are in model_args directly
        model = BiasNet(
            input_len=data_config["input_len"],
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=model_args["input_channels"],
            output_channels=model_args["output_channels"],
            filters=model_args["filters"],
            n_layers=model_args["n_layers"],
            dilations=model_args["dilations"],
            dil_kernel_size=model_args["dil_kernel_size"],
            conv_kernel_size=model_args["conv_kernel_size"],
            profile_kernel_size=model_args["profile_kernel_size"],
            dropout=model_args["dropout"],
            predict_total_count=model_args["predict_total_count"],
            residual=model_args["residual"],
            linear_head=model_args["linear_head"],
        )
    elif model_name == "Dalmatian":
        # Dalmatian — bias params are prefixed with bias_* in model_args;
        # use defaults from BiasNet for anything not explicitly set.
        bias_filters = model_args.get("bias_filters", 12)
        bias_n_layers = model_args.get("bias_n_layers", 5)
        bias_dilations = model_args.get("bias_dilations", [1] * bias_n_layers)
        bias_dil_kernel_size = model_args.get("bias_dil_kernel_size", 9)
        bias_conv_kernel_size = model_args.get("bias_conv_kernel_size", [11, 11])
        bias_profile_kernel_size = model_args.get("bias_profile_kernel_size", 45)
        bias_dropout = model_args.get("bias_dropout", 0.1)
        bias_linear_head = model_args.get("bias_linear_head", True)
        bias_residual = model_args.get("bias_residual", True)

        # Compute correct input_len for the bias subnetwork
        bias_shrinkage = _compute_shrinkage(
            bias_conv_kernel_size, bias_dilations,
            bias_dil_kernel_size, bias_profile_kernel_size,
        )
        bias_input_len = output_len + bias_shrinkage

        model = BiasNet(
            input_len=bias_input_len,
            output_len=output_len,
            output_bin_size=output_bin_size,
            input_channels=model_args["input_channels"],
            output_channels=model_args["output_channels"],
            filters=bias_filters,
            n_layers=bias_n_layers,
            dilations=bias_dilations,
            dil_kernel_size=bias_dil_kernel_size,
            conv_kernel_size=bias_conv_kernel_size,
            profile_kernel_size=bias_profile_kernel_size,
            dropout=bias_dropout,
            predict_total_count=False,  # Dalmatian's BiasNet never predicts total count
            residual=bias_residual,
            linear_head=bias_linear_head,
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_name}'. "
            f"Expected 'BiasNet' or 'Dalmatian'."
        )

    sd = torch.load(model_pt, map_location="cpu", weights_only=True)
    bias_sd = _extract_bias_state_dict(sd)
    model.load_state_dict(bias_sd)
    logger.info(
        f"Loaded BiasNet from {model_name} checkpoint: "
        f"input_len={model.input_len}, output_len={model.output_len}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def load_peak_intervals(peaks_path: Path, n: int = 5000, seed: int = 42):
    """Load peak intervals from a BED/narrowPeak file.

    Returns list of (chrom, start, end) tuples.
    """
    intervals = []
    opener = gzip.open if str(peaks_path).endswith(".gz") else open
    with opener(peaks_path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.strip().split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            intervals.append((chrom, start, end))

    rng = np.random.RandomState(seed)
    if len(intervals) > n:
        idx = rng.choice(len(intervals), size=n, replace=False)
        intervals = [intervals[i] for i in idx]

    return intervals


def extract_onehot(fasta: pyfaidx.Fasta, chrom: str, center: int,
                   length: int) -> np.ndarray | None:
    """Extract a one-hot encoded sequence centered on a position.

    Returns (4, length) float32 array, or None if sequence contains N's.
    """
    start = center - length // 2
    end = start + length
    if start < 0 or end > len(fasta[chrom]):
        return None

    seq = str(fasta[chrom][start:end]).upper()
    if "N" in seq:
        return None

    onehot = np.zeros((4, length), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in NUC_MAP:
            onehot[NUC_MAP[base], i] = 1.0
    return onehot


def get_real_sequences(fasta: pyfaidx.Fasta, intervals: list,
                       input_len: int, n_seqs: int = 1000,
                       seed: int = 42) -> list[np.ndarray]:
    """Extract real genomic one-hot sequences centered on peak midpoints."""
    rng = np.random.RandomState(seed)
    sequences = []
    attempts = 0
    max_attempts = len(intervals) * 3

    shuffled = list(intervals)
    rng.shuffle(shuffled)

    for chrom, start, end in shuffled:
        if len(sequences) >= n_seqs:
            break
        center = (start + end) // 2
        onehot = extract_onehot(fasta, chrom, center, input_len)
        if onehot is not None:
            sequences.append(onehot)
        attempts += 1
        if attempts >= max_attempts:
            break

    logger.info(f"Extracted {len(sequences)} real sequences from {attempts} attempts")
    return sequences


def get_background_sequences(
    fasta: pyfaidx.Fasta,
    peaks_path: Path,
    chrom_sizes: dict[str, int],
    input_len: int,
    n_seqs: int = 1000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Extract random genomic sequences outside peak regions.

    Uses :class:`cerberus.samplers.RandomSampler` with peak exclusions to
    generate background intervals, matching the training paradigm of BiasNet
    (negative-peak sampler).
    """
    from interlap import InterLap

    from cerberus.samplers import IntervalSampler, RandomSampler

    # Load peaks as exclusion zones
    peak_sampler = IntervalSampler(
        file_path=peaks_path,
        chrom_sizes=chrom_sizes,
        padded_size=input_len,
    )
    logger.info(f"Loaded {len(peak_sampler)} peaks for exclusion")

    exclude: dict[str, InterLap] = {}
    for iv in peak_sampler:
        if iv.chrom not in exclude:
            exclude[iv.chrom] = InterLap()
        exclude[iv.chrom].add((iv.start, iv.end))

    # Sample random background intervals outside peaks
    bg_sampler = RandomSampler(
        chrom_sizes=chrom_sizes,
        padded_size=input_len,
        num_intervals=n_seqs * 3,
        exclude_intervals=exclude,
        seed=seed,
        generate_on_init=True,
    )
    logger.info(f"Generated {len(bg_sampler)} background candidate intervals")

    sequences = []
    for iv in bg_sampler:
        if len(sequences) >= n_seqs:
            break
        center = (iv.start + iv.end) // 2
        onehot = extract_onehot(fasta, iv.chrom, center, input_len)
        if onehot is not None:
            sequences.append(onehot)

    logger.info(f"Extracted {len(sequences)} background sequences (outside peaks)")
    return sequences


# ---------------------------------------------------------------------------
# ISM computation
# ---------------------------------------------------------------------------

def compute_ism_real(model: torch.nn.Module, sequences: list[np.ndarray],
                     window: int = 31) -> np.ndarray:
    """ISM averaged over real genomic sequences.

    For each sequence, mutate each position in the center window to each
    nucleotide and measure the effect on the center output position.

    Returns:
        (4, window) array of average ISM scores.
    """
    model.eval()
    device = next(model.parameters()).device
    output_len: int = model.output_len  # type: ignore[assignment]
    input_len: int = model.input_len  # type: ignore[assignment]
    input_center = input_len // 2
    start = input_center - window // 2

    ism_sum = np.zeros((4, window))
    n = 0

    for seq in sequences:
        x_ref = torch.tensor(seq, device=device).unsqueeze(0)
        with torch.no_grad():
            ref_val = model(x_ref).logits[0, 0, output_len // 2].item()

        for pos_i in range(window):
            pos = start + pos_i
            for nuc in range(4):
                if seq[nuc, pos] == 1.0:
                    ism_sum[nuc, pos_i] += 0.0
                else:
                    mutant = x_ref.clone()
                    mutant[0, :, pos] = 0.0
                    mutant[0, nuc, pos] = 1.0
                    with torch.no_grad():
                        mut_val = model(mutant).logits[0, 0, output_len // 2].item()
                    ism_sum[nuc, pos_i] += mut_val - ref_val
        n += 1
        if n % 50 == 0:
            logger.info(f"  ISM: {n}/{len(sequences)} sequences")

    return ism_sum / n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_logo(ax, weights: np.ndarray, title: str, as_ic: bool = True):
    """Plot a weight matrix as a sequence logo."""
    _, L = weights.shape

    if as_ic:
        probs = np.exp(weights) / np.exp(weights).sum(axis=0, keepdims=True)
        log2_p = np.log2(probs + 1e-10)
        ic = np.clip(2.0 + (probs * log2_p).sum(axis=0), 0, 2)
        heights = probs * ic[np.newaxis, :]
    else:
        heights = weights

    for pos in range(L):
        col = heights[:, pos]

        if as_ic:
            order = np.argsort(col)
            bottom = 0.0
            for nuc_idx in order:
                h = col[nuc_idx]
                if h < 1e-6:
                    continue
                letter = NUCLEOTIDES[nuc_idx]
                color = NUC_COLORS[letter]
                ax.bar(pos, h, bottom=bottom, width=0.9, color=color,
                       edgecolor="white", linewidth=0.2)
                col_total = col.sum()
                if col_total > 0 and h / col_total > 0.15 and h > 0.05:
                    ax.text(pos, bottom + h / 2, letter, ha="center", va="center",
                            fontsize=6, fontweight="bold", color="white",
                            fontfamily="monospace",
                            path_effects=[path_effects.Stroke(linewidth=0.4,
                                                               foreground="black"),
                                          path_effects.Normal()])
                bottom += h
        else:
            pos_bottom = 0.0
            neg_bottom = 0.0
            for nuc_idx in np.argsort(col):
                h = col[nuc_idx]
                letter = NUCLEOTIDES[nuc_idx]
                color = NUC_COLORS[letter]
                if h >= 0:
                    ax.bar(pos, h, bottom=pos_bottom, width=0.9, color=color,
                           edgecolor="white", linewidth=0.2)
                    if abs(h) > 0.005:
                        ax.text(pos, pos_bottom + h / 2, letter, ha="center",
                                va="center", fontsize=5, fontweight="bold",
                                color="white", fontfamily="monospace",
                                path_effects=[path_effects.Stroke(linewidth=0.4,
                                                                   foreground="black"),
                                              path_effects.Normal()])
                    pos_bottom += h
                else:
                    ax.bar(pos, h, bottom=neg_bottom, width=0.9, color=color,
                           edgecolor="white", linewidth=0.2)
                    neg_bottom += h

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.3)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_ism_csv(ism: np.ndarray, window: int, n_seqs: int, out_path: Path):
    """Save ISM matrix as a CSV file.

    Columns: position, A, C, G, T
    Each row is a position in the ISM window with average delta values per nucleotide.
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# n_seqs", n_seqs])
        writer.writerow(["# window", window])
        writer.writerow(["position", "A", "C", "G", "T"])
        for pos_i in range(window):
            writer.writerow([pos_i, ism[0, pos_i], ism[1, pos_i],
                             ism[2, pos_i], ism[3, pos_i]])
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot BiasNet ISM motif as IC logo + heatmap")
    parser.add_argument("model_path", type=str,
                        help="Path to BiasNet run directory or model.pt file")
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to genome FASTA (auto-detected from hparams.yaml)")
    parser.add_argument("--peaks", type=str, default=None,
                        help="Path to peaks BED file (auto-detected from hparams.yaml)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as model directory)")
    parser.add_argument("--prefix", type=str, default="biasnet",
                        help="Output filename prefix")
    parser.add_argument("--n-seqs", type=int, default=1000,
                        help="Number of real sequences for ISM averaging")
    parser.add_argument("--ism-window", type=int, default=31,
                        help="Window size for ISM around center")
    parser.add_argument("--background", action="store_true", default=True,
                        help="Sample background sequences outside peaks (default)")
    parser.add_argument("--no-background", dest="background", action="store_false",
                        help="Sample sequences from peak centers instead of background")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g., 'cuda', 'cpu'). Auto-detects if not set.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model_path = Path(args.model_path)
    config = load_config(model_path)
    model = load_biasnet(model_path, device)

    # Resolve fasta and peaks from config if not provided
    fasta_path = args.fasta or config["genome_config"]["fasta_path"]
    peaks_path = args.peaks or config["sampler_config"]["sampler_args"]["intervals_path"]

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = model_path.parent if model_path.is_file() else model_path
    out_dir.mkdir(parents=True, exist_ok=True)

    window = args.ism_window

    # Load genome
    logger.info(f"Loading genome from {fasta_path}...")
    fasta = pyfaidx.Fasta(fasta_path)

    # Extract sequences
    if args.background:
        chrom_sizes = config["genome_config"]["chrom_sizes"]
        logger.info(f"Sampling background sequences outside peaks from {peaks_path}...")
        input_len: int = model.input_len  # type: ignore[assignment]
        sequences = get_background_sequences(
            fasta, Path(peaks_path), chrom_sizes, input_len,
            n_seqs=args.n_seqs,
        )
    else:
        logger.info(f"Loading peak intervals from {peaks_path}...")
        input_len: int = model.input_len  # type: ignore[assignment]
        intervals = load_peak_intervals(Path(peaks_path), n=args.n_seqs * 3)
        sequences = get_real_sequences(fasta, intervals, input_len,
                                       n_seqs=args.n_seqs)

    # Compute ISM
    n_seqs = len(sequences)
    seq_type = "background" if args.background else "peak"
    logger.info(f"Computing ISM on {n_seqs} {seq_type} sequences (window={window})...")
    ism = compute_ism_real(model, sequences, window=window)

    # Save CSV
    csv_path = out_dir / f"{args.prefix}_biasnet_ism.csv"
    save_ism_csv(ism, window, n_seqs, csv_path)

    # Plot: IC logo (top) + ISM heatmap (bottom), X-aligned
    fig = plt.figure(figsize=(max(12, window * 0.4), 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1],
                          width_ratios=[1, 0.02], hspace=0.3, wspace=0.03)
    ax_logo = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_logo)
    ax_cbar = fig.add_subplot(gs[1, 1])

    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.set_visible(False)

    plot_logo(ax_logo, ism,
              f"BiasNet ISM IC — {n_seqs} {seq_type} seqs (center {window}bp)",
              as_ic=True)
    ax_logo.set_ylabel("IC (bits)", fontsize=8)
    plt.setp(ax_logo.get_xticklabels(), visible=False)

    vmax = np.abs(ism).max()
    im = ax_heat.imshow(ism, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest",
                        extent=(-0.5, window - 0.5, 3.5, -0.5))
    ax_heat.set_yticks(range(4))
    ax_heat.set_yticklabels(NUCLEOTIDES, fontsize=8)
    ax_heat.set_xlabel("Position (centered)", fontsize=8)
    ax_heat.set_title("ISM Δ output heatmap", fontsize=9)

    fig.colorbar(im, cax=ax_cbar)

    png_path = out_dir / f"{args.prefix}_biasnet_ism.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
