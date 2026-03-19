#!/usr/bin/env python
"""Plot BiasNet pairwise ISM epistasis analysis.

Loads a trained BiasNet from a run directory (containing config.json + model.pt)
or a standalone model.pt file, computes pairwise in-silico mutagenesis (ISM) over
real genomic sequences to measure epistatic interactions between positions.

Produces:
  1. A PNG plot with IC logo (top), epistasis heatmap (middle), and interaction
     profile (bottom), all X-aligned.
  2. A CSV file with the full epistasis matrix in long format.

Usage:
    python tools/plot_biasnet_pairwise_ism.py <run_dir_or_model_pt> \\
        [--output-dir plots/] [--prefix biasnet] \\
        [--n-seqs 200] [--ism-window 31] [--batch-size 512] [--device cuda]
"""

import argparse
import csv
import gzip
import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pyfaidx
import torch

logger = logging.getLogger(__name__)

NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# Model loading (duplicated from plot_biasnet_ism.py)
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
    """Load config.json from a run directory or model.pt sibling."""
    fold_dir, _ = _resolve_fold_dir(path)
    config_path = fold_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {fold_dir}")
    with open(config_path) as f:
        return json.load(f)


def _extract_bias_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract bias_model state dict from a Dalmatian checkpoint."""
    prefix = "bias_model."
    bias_keys = [k for k in sd if k.startswith(prefix)]
    if bias_keys:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def load_biasnet(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a BiasNet model from a run directory or standalone .pt file.

    Supports standalone BiasNet and Dalmatian (extracts bias_model subnetwork).
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
        bias_filters = model_args.get("bias_filters", 12)
        bias_n_layers = model_args.get("bias_n_layers", 5)
        bias_dilations = model_args.get("bias_dilations", [1] * bias_n_layers)
        bias_dil_kernel_size = model_args.get("bias_dil_kernel_size", 9)
        bias_conv_kernel_size = model_args.get("bias_conv_kernel_size", [11, 11])
        bias_profile_kernel_size = model_args.get("bias_profile_kernel_size", 45)
        bias_dropout = model_args.get("bias_dropout", 0.1)
        bias_linear_head = model_args.get("bias_linear_head", True)
        bias_residual = model_args.get("bias_residual", True)

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
            predict_total_count=False,
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
# Sequence extraction (duplicated from plot_biasnet_ism.py)
# ---------------------------------------------------------------------------

def load_peak_intervals(peaks_path: Path, n: int = 5000, seed: int = 42):
    """Load peak intervals from a BED/narrowPeak file."""
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
    """Extract a one-hot encoded sequence centered on a position."""
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


def get_background_sequences(
    fasta: pyfaidx.Fasta,
    peaks_path: Path,
    chrom_sizes: dict[str, int],
    input_len: int,
    n_seqs: int = 1000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Extract random genomic sequences outside peak regions."""
    from interlap import InterLap
    from cerberus.samplers import IntervalSampler, RandomSampler

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


# ---------------------------------------------------------------------------
# Pairwise ISM computation
# ---------------------------------------------------------------------------

def _batch_forward(model: torch.nn.Module, batch: torch.Tensor,
                   output_idx: int, batch_size: int) -> np.ndarray:
    """Run model forward on a batch in chunks, return center output values."""
    n = batch.shape[0]
    vals = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        with torch.no_grad():
            out = model(batch[start:end]).logits[:, 0, output_idx]
            vals[start:end] = out.cpu().numpy()
    return vals


def compute_pairwise_ism(
    model: torch.nn.Module,
    sequences: list[np.ndarray],
    window: int = 31,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise ISM averaged over genomic sequences.

    For each sequence, mutates all single positions and all pairs of positions
    in the center window, then computes the epistasis term:
        epsilon(i,a,j,b) = delta(i->a, j->b) - delta(i->a) - delta(j->b)

    All mutants for a given sequence are batched on GPU for efficiency.

    Args:
        model: BiasNet model (on device, eval mode).
        sequences: List of one-hot arrays, each shape (4, input_len).
        window: Number of positions to scan around center.
        batch_size: GPU batch size for forward passes.

    Returns:
        ism_single: (4, window) average single-position ISM scores.
        epistasis: (window, 4, window, 4) average epistasis values.
    """
    model.eval()
    device = next(model.parameters()).device
    output_len: int = model.output_len  # type: ignore[assignment]
    input_len: int = model.input_len  # type: ignore[assignment]
    input_center = input_len // 2
    win_start = input_center - window // 2
    output_center = output_len // 2

    # Pre-compute pair indices
    pair_indices = list(combinations(range(window), 2))

    ism_single_sum = np.zeros((4, window), dtype=np.float64)
    epistasis_sum = np.zeros((window, 4, window, 4), dtype=np.float64)
    n = 0

    for seq_idx, seq in enumerate(sequences):
        x_ref = torch.tensor(seq, device=device).unsqueeze(0)  # (1, 4, L)

        # Reference value
        with torch.no_grad():
            ref_val = model(x_ref).logits[0, 0, output_center].item()

        # --- Single mutants: W * 3 non-reference mutations ---
        single_mutants = []
        single_index = []  # (pos_i, nuc)
        for pos_i in range(window):
            pos = win_start + pos_i
            ref_nuc = seq[:, pos].argmax()
            for nuc in range(4):
                if nuc == ref_nuc:
                    continue
                mut = x_ref.clone()
                mut[0, :, pos] = 0.0
                mut[0, nuc, pos] = 1.0
                single_mutants.append(mut)
                single_index.append((pos_i, nuc))

        single_batch = torch.cat(single_mutants, dim=0)  # (W*3, 4, L)
        single_vals = _batch_forward(model, single_batch, output_center, batch_size)

        # Build delta_single lookup: delta_single[pos_i, nuc] = mut_val - ref_val
        delta_single = np.zeros((window, 4), dtype=np.float64)
        for idx, (pos_i, nuc) in enumerate(single_index):
            delta_single[pos_i, nuc] = single_vals[idx] - ref_val

        # Accumulate single ISM
        for pos_i in range(window):
            for nuc in range(4):
                pos = win_start + pos_i
                if seq[nuc, pos] == 1.0:
                    pass  # reference nucleotide: delta = 0
                else:
                    ism_single_sum[nuc, pos_i] += delta_single[pos_i, nuc]

        # --- Double mutants: C(W,2) * 9 combinations ---
        double_mutants = []
        double_index = []  # (pos_i, nuc_a, pos_j, nuc_b)
        for pos_i, pos_j in pair_indices:
            abs_i = win_start + pos_i
            abs_j = win_start + pos_j
            ref_nuc_i = seq[:, abs_i].argmax()
            ref_nuc_j = seq[:, abs_j].argmax()
            for nuc_a in range(4):
                if nuc_a == ref_nuc_i:
                    continue
                for nuc_b in range(4):
                    if nuc_b == ref_nuc_j:
                        continue
                    mut = x_ref.clone()
                    mut[0, :, abs_i] = 0.0
                    mut[0, nuc_a, abs_i] = 1.0
                    mut[0, :, abs_j] = 0.0
                    mut[0, nuc_b, abs_j] = 1.0
                    double_mutants.append(mut)
                    double_index.append((pos_i, nuc_a, pos_j, nuc_b))

        double_batch = torch.cat(double_mutants, dim=0)
        double_vals = _batch_forward(model, double_batch, output_center, batch_size)

        # Compute epistasis for each double mutant
        for idx, (pos_i, nuc_a, pos_j, nuc_b) in enumerate(double_index):
            delta_double = double_vals[idx] - ref_val
            eps = delta_double - delta_single[pos_i, nuc_a] - delta_single[pos_j, nuc_b]
            epistasis_sum[pos_i, nuc_a, pos_j, nuc_b] += eps
            # Symmetric: fill (j, b, i, a) too
            epistasis_sum[pos_j, nuc_b, pos_i, nuc_a] += eps

        n += 1
        if n % 10 == 0:
            logger.info(f"  Pairwise ISM: {n}/{len(sequences)} sequences")

    logger.info(f"  Pairwise ISM: {n}/{len(sequences)} sequences (done)")

    ism_single = ism_single_sum / n
    epistasis = epistasis_sum / n
    return ism_single, epistasis


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

def save_pairwise_csv(
    epistasis: np.ndarray,
    ism_single: np.ndarray,
    window: int,
    n_seqs: int,
    out_path: Path,
):
    """Save pairwise epistasis in long format CSV.

    Columns: pos_i, nuc_i, pos_j, nuc_j, delta_single_i, delta_single_j, epistasis
    Only upper triangle (pos_i < pos_j) to avoid redundancy.
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# n_seqs", n_seqs])
        writer.writerow(["# window", window])
        writer.writerow([
            "pos_i", "nuc_i", "pos_j", "nuc_j",
            "delta_single_i", "delta_single_j", "epistasis",
        ])
        for pos_i in range(window):
            for pos_j in range(pos_i + 1, window):
                for nuc_a in range(4):
                    for nuc_b in range(4):
                        eps = epistasis[pos_i, nuc_a, pos_j, nuc_b]
                        ds_i = ism_single[nuc_a, pos_i]
                        ds_j = ism_single[nuc_b, pos_j]
                        writer.writerow([
                            pos_i, NUCLEOTIDES[nuc_a],
                            pos_j, NUCLEOTIDES[nuc_b],
                            f"{ds_i:.6f}", f"{ds_j:.6f}", f"{eps:.6f}",
                        ])
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot BiasNet pairwise ISM epistasis analysis")
    parser.add_argument("model_path", type=str,
                        help="Path to BiasNet run directory or model.pt file")
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to genome FASTA (auto-detected from config.json)")
    parser.add_argument("--peaks", type=str, default=None,
                        help="Path to peaks BED file (auto-detected from config.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as model directory)")
    parser.add_argument("--prefix", type=str, default="biasnet",
                        help="Output filename prefix")
    parser.add_argument("--n-seqs", type=int, default=200,
                        help="Number of sequences for ISM averaging (default: 200)")
    parser.add_argument("--ism-window", type=int, default=31,
                        help="Window size for ISM around center")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="GPU batch size for forward passes (default: 512)")
    parser.add_argument("--background", action="store_true", default=True,
                        help="Sample background sequences outside peaks (default)")
    parser.add_argument("--no-background", dest="background", action="store_false",
                        help="Sample sequences from peak centers instead of background")
    parser.add_argument("--detail-pair", type=int, nargs=2, default=[11, 19],
                        metavar=("POS_I", "POS_J"),
                        help="Position pair for 4x4 epistasis detail panel (default: 11 19)")
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
    input_len: int = model.input_len  # type: ignore[assignment]
    if args.background:
        chrom_sizes = config["genome_config"]["chrom_sizes"]
        logger.info(f"Sampling background sequences outside peaks from {peaks_path}...")
        sequences = get_background_sequences(
            fasta, Path(peaks_path), chrom_sizes, input_len,
            n_seqs=args.n_seqs,
        )
    else:
        logger.info(f"Loading peak intervals from {peaks_path}...")
        intervals = load_peak_intervals(Path(peaks_path), n=args.n_seqs * 3)
        sequences = get_real_sequences(fasta, intervals, input_len,
                                       n_seqs=args.n_seqs)

    # Compute pairwise ISM
    n_seqs = len(sequences)
    n_pairs = window * (window - 1) // 2
    n_double = n_pairs * 9
    seq_type = "background" if args.background else "peak"
    logger.info(
        f"Computing pairwise ISM on {n_seqs} {seq_type} sequences "
        f"(window={window}, {n_pairs} pairs, {n_double} double mutants/seq)..."
    )
    ism_single, epistasis = compute_pairwise_ism(
        model, sequences, window=window, batch_size=args.batch_size,
    )

    # Save CSV
    csv_path = out_dir / f"{args.prefix}_pairwise_ism.csv"
    save_pairwise_csv(epistasis, ism_single, window, n_seqs, csv_path)

    # --- Plot: 4 panels ---
    # Collapse epistasis to (W, W) by taking max |eps| over nucleotide combinations
    eps_collapsed = np.max(np.abs(epistasis), axis=(1, 3))  # (W, W)

    # Detail pair positions
    dp_i, dp_j = args.detail_pair
    if dp_i >= window or dp_j >= window:
        logger.warning(
            f"--detail-pair ({dp_i}, {dp_j}) out of range for window={window}, "
            f"skipping detail panel"
        )
        dp_i, dp_j = None, None

    fig = plt.figure(figsize=(max(14, window * 0.45), 13))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 2, 0.8, 1.5],
        width_ratios=[1, 0.03],
        hspace=0.4, wspace=0.03,
    )
    ax_logo = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])
    ax_profile = fig.add_subplot(gs[2, 0], sharex=ax_logo)

    # Hide unused colorbar cells
    for row in [0, 2, 3]:
        ax_empty = fig.add_subplot(gs[row, 1])
        ax_empty.set_visible(False)

    # Panel 1: IC logo from single ISM
    plot_logo(ax_logo, ism_single,
              f"BiasNet ISM IC — {n_seqs} {seq_type} seqs (center {window}bp)",
              as_ic=True)
    ax_logo.set_ylabel("IC (bits)", fontsize=8)
    plt.setp(ax_logo.get_xticklabels(), visible=False)

    # Panel 2: epistasis heatmap
    vmax = np.percentile(eps_collapsed[eps_collapsed > 0], 99) if eps_collapsed.max() > 0 else 1e-3
    im = ax_heat.imshow(
        eps_collapsed, aspect="equal", cmap="inferno",
        vmin=0, vmax=vmax, interpolation="nearest",
        extent=(-0.5, window - 0.5, window - 0.5, -0.5),
    )
    ax_heat.set_xlabel("Position", fontsize=8)
    ax_heat.set_ylabel("Position", fontsize=8)
    ax_heat.set_title(f"Pairwise epistasis  max|ε|  ({n_pairs} pairs)", fontsize=9)
    # Mark detail pair on heatmap
    if dp_i is not None and dp_j is not None:
        ax_heat.plot(dp_j, dp_i, "s", markersize=8, markeredgecolor="white",
                     markerfacecolor="none", markeredgewidth=1.5)
        ax_heat.plot(dp_i, dp_j, "s", markersize=8, markeredgecolor="white",
                     markerfacecolor="none", markeredgewidth=1.5)
    fig.colorbar(im, cax=ax_cbar, label="max |ε|")

    # Panel 3: interaction profile
    interaction_profile = eps_collapsed.sum(axis=1)
    ax_profile.bar(range(window), interaction_profile, width=0.8, color="#4C72B0",
                   edgecolor="white", linewidth=0.3)
    ax_profile.set_xlim(-0.5, window - 0.5)
    ax_profile.set_xlabel("Position (centered)", fontsize=8)
    ax_profile.set_ylabel("Σ max|ε|", fontsize=8)
    ax_profile.set_title("Interaction profile per position", fontsize=9)
    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)

    # Panel 4: 4x4 epistasis detail for selected pair
    ax_detail = fig.add_subplot(gs[3, 0])
    if dp_i is not None:
        # epistasis[pos_i, nuc_a, pos_j, nuc_b] -> (4, 4) matrix
        pair_eps = epistasis[dp_i, :, dp_j, :]  # (4, 4): rows=nuc at pos_i, cols=nuc at pos_j
        vabs = max(np.abs(pair_eps).max(), 1e-6)
        im4 = ax_detail.imshow(
            pair_eps, aspect="equal", cmap="RdBu_r",
            vmin=-vabs, vmax=vabs, interpolation="nearest",
        )
        ax_detail.set_xticks(range(4))
        ax_detail.set_xticklabels(NUCLEOTIDES, fontsize=10, fontweight="bold")
        ax_detail.set_yticks(range(4))
        ax_detail.set_yticklabels(NUCLEOTIDES, fontsize=10, fontweight="bold")
        ax_detail.set_xlabel(f"Nucleotide at position {dp_j}", fontsize=9)
        ax_detail.set_ylabel(f"Nucleotide at position {dp_i}", fontsize=9)
        ax_detail.set_title(
            f"Epistasis ε(pos {dp_i}, pos {dp_j})", fontsize=9,
        )
        # Annotate cells with values
        for ri in range(4):
            for ci in range(4):
                val = pair_eps[ri, ci]
                color = "white" if abs(val) > vabs * 0.5 else "black"
                ax_detail.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                               fontsize=8, color=color)
        fig.colorbar(im4, ax=ax_detail, fraction=0.046, pad=0.04, label="ε")
    else:
        ax_detail.set_visible(False)

    png_path = out_dir / f"{args.prefix}_pairwise_ism.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
