#!/usr/bin/env python
"""Score per-head prediction changes after motif substitution.

This follows the bpAI-TAC motif marginalization pattern as closely as the
Cerberus model API allows: build a no-motif background from dinucleotide-
shuffled inputs, substitute each motif into that background, and report the
paired motif-minus-background delta for each output head.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import cerberus
from cerberus.config import SamplerConfig
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.utils import resolve_device

logger = logging.getLogger(__name__)

_BASES = ("A", "C", "G", "T")
_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
_SUMMARY_METRICS = (
    "delta_log_count",
    "delta_exp_log_count",
    "delta_count_minus_pseudocount",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Substitute motifs into no-motif backgrounds and report per-head "
            "prediction changes for a trained Cerberus model."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="ModelEnsemble-compatible training root, e.g. models/run/single-fold.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--intervals-path",
        type=Path,
        required=True,
        help="BED file of intervals to probe. A materialized positive split is recommended.",
    )
    parser.add_argument(
        "--motif",
        action="append",
        default=[],
        help=(
            "Motif to substitute as NAME=ACGT. Can be repeated. Only A/C/G/T "
            "are accepted to keep the substituted sequence deterministic."
        ),
    )
    parser.add_argument(
        "--meme-path",
        type=Path,
        help=(
            "Optional MEME motif file. Matching motifs are converted to "
            "deterministic consensus sequences and added to --motif entries."
        ),
    )
    parser.add_argument(
        "--motif-name-regex",
        help=(
            "Regex used to select motifs from --meme-path, for example "
            "'^ANDR\\.'. If omitted, all MEME motifs are used."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        action="append",
        default=[0],
        help=(
            "Motif start offset relative to the centered motif position. Can be "
            "repeated. Default: 0."
        ),
    )
    parser.add_argument("--n-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--background-mode",
        choices=("dinuc-shuffle", "original"),
        default="dinuc-shuffle",
        help=(
            "No-motif background. bpAI-TAC used dinucleotide-shuffled "
            "sequences. Use 'original' to score against the input interval "
            "sequence instead."
        ),
    )
    parser.add_argument(
        "--chrombpnet-accessibility-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Score the ChromBPNet accessibility branch only, excluding the bias "
            "branch. Default: auto; use chrombpnet_wo_bias when the checkpoint "
            "exposes it."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help=(
            "Optional per-motif/per-head summary TSV path. Defaults to "
            "<output stem>.summary.tsv."
        ),
    )
    return parser


def _parse_inline_motifs(values: list[str]) -> list[tuple[str, str]]:
    motifs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Motif must be NAME=ACGT, got {value!r}")
        name, seq = value.split("=", 1)
        name = name.strip()
        seq = seq.strip().upper()
        if not name:
            raise ValueError(f"Motif name is empty in {value!r}")
        if not seq or any(base not in _BASE_TO_INDEX for base in seq):
            raise ValueError(
                f"Motif {name!r} must contain only A/C/G/T bases, got {seq!r}"
            )
        motifs.append((name, seq))
    return motifs


def _load_meme_consensus_motifs(
    meme_path: Path, motif_name_regex: str | None
) -> list[tuple[str, str]]:
    motif_re = re.compile(motif_name_regex) if motif_name_regex else None
    motifs: list[tuple[str, str]] = []
    current_name: str | None = None
    matrix: list[list[float]] = []
    in_matrix = False

    def finalize_current() -> None:
        if current_name is None or not matrix:
            return
        if motif_re is not None and motif_re.search(current_name) is None:
            return
        consensus = "".join(_BASES[int(np.argmax(row))] for row in matrix)
        motifs.append((current_name, consensus))

    with meme_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("MOTIF "):
                finalize_current()
                parts = line.split()
                current_name = parts[1] if len(parts) > 1 else None
                matrix = []
                in_matrix = False
                continue

            if current_name is None:
                continue
            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue
            if not in_matrix:
                continue

            parts = line.split()
            if len(parts) < 4:
                in_matrix = False
                continue
            try:
                row = [float(value) for value in parts[:4]]
            except ValueError:
                in_matrix = False
                continue
            matrix.append(row)

    finalize_current()
    return motifs


def _deduplicate_motif_names(motifs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: dict[str, int] = {}
    deduped: list[tuple[str, str]] = []
    for name, seq in motifs:
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}#{count + 1}"
        deduped.append((name, seq))
    return deduped


def _collect_motifs(args: argparse.Namespace) -> list[tuple[str, str]]:
    motifs = _parse_inline_motifs(args.motif)
    if args.meme_path is not None:
        motifs.extend(
            _load_meme_consensus_motifs(
                args.meme_path.resolve(), args.motif_name_regex
            )
        )
    motifs = _deduplicate_motif_names(motifs)
    if not motifs:
        raise ValueError(
            "At least one motif must be provided with --motif or selected from "
            "--meme-path."
        )
    return motifs


def _make_interval_dataset(ensemble: ModelEnsemble, intervals_path: Path, seed: int):
    cfg = ensemble.cerberus_config
    sampler_config = SamplerConfig(
        sampler_type="interval",
        padded_size=cfg.sampler_config.padded_size,
        sampler_args={"intervals_path": intervals_path.resolve()},
    )
    return CerberusDataset(
        genome_config=cfg.genome_config,
        data_config=cfg.data_config,
        sampler_config=sampler_config,
        in_memory=False,
        is_train=False,
        seed=seed,
    )


def _insert_motif(
    inputs: torch.Tensor, motif_seq: str, offset: int
) -> tuple[torch.Tensor, int, int]:
    seq_len = inputs.shape[-1]
    motif_len = len(motif_seq)
    start = seq_len // 2 - motif_len // 2 + offset
    end = start + motif_len
    if start < 0 or end > seq_len:
        raise ValueError(
            f"Motif {motif_seq!r} with offset {offset} falls outside input "
            f"length {seq_len}"
        )

    mutated = inputs.clone()
    mutated[:, :4, start:end] = 0.0
    for j, base in enumerate(motif_seq):
        mutated[:, _BASE_TO_INDEX[base], start + j] = 1.0
    return mutated, start, end


def _dinucleotide_shuffle_indices(indices: np.ndarray, seed: int) -> np.ndarray:
    """Return one dinucleotide-preserving shuffle of a nucleotide index array."""

    rng = np.random.RandomState(seed)
    next_indices: list[np.ndarray] = []
    for base_idx in range(len(_BASES)):
        followers = np.where(indices[:-1] == base_idx)[0] + 1
        if followers.shape[0] > 1:
            order = np.arange(followers.shape[0])
            order[:-1] = rng.permutation(followers.shape[0] - 1)
            followers = followers[order]
        next_indices.append(followers)

    shuffled = np.empty_like(indices)
    counters = np.zeros(len(_BASES), dtype=np.int64)
    source_position = 0
    shuffled[0] = indices[source_position]
    for position in range(1, indices.shape[0]):
        base_idx = int(indices[source_position])
        count = int(counters[base_idx])
        if count >= next_indices[base_idx].shape[0]:
            raise RuntimeError("Dinucleotide shuffle failed to build an Eulerian path.")
        source_position = int(next_indices[base_idx][count])
        counters[base_idx] += 1
        shuffled[position] = indices[source_position]
    return shuffled


def _dinucleotide_shuffle_inputs(
    inputs: torch.Tensor, *, seed: int, global_start: int
) -> torch.Tensor:
    if inputs.shape[1] < len(_BASES):
        raise ValueError(
            f"Expected at least {len(_BASES)} input channels for DNA one-hot tracks, "
            f"got {inputs.shape[1]}"
        )

    dna = inputs[:, : len(_BASES), :].detach().cpu()
    indices = dna.argmax(dim=1).numpy().astype(np.int64)
    shuffled_indices = np.stack(
        [
            _dinucleotide_shuffle_indices(seq_indices, seed + global_start + row_idx)
            for row_idx, seq_indices in enumerate(indices)
        ]
    )
    shuffled_onehot = torch.nn.functional.one_hot(
        torch.from_numpy(shuffled_indices), num_classes=len(_BASES)
    )
    shuffled_onehot = shuffled_onehot.permute(0, 2, 1).to(dtype=inputs.dtype)

    shuffled = inputs.clone()
    shuffled[:, : len(_BASES), :] = shuffled_onehot.to(device=shuffled.device)
    return shuffled


def _make_background_inputs(
    inputs: torch.Tensor, *, mode: str, seed: int, global_start: int
) -> torch.Tensor:
    if mode == "original":
        return inputs.clone()
    if mode == "dinuc-shuffle":
        return _dinucleotide_shuffle_inputs(
            inputs, seed=seed, global_start=global_start
        )
    raise ValueError(f"Unknown background mode: {mode}")


def _prediction_log_counts(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    out = model(inputs)
    log_counts = out.log_counts
    if log_counts.ndim != 2:
        raise ValueError(f"Expected log_counts shape (B, C), got {tuple(log_counts.shape)}")
    return log_counts.float()


def _parse_interval_repr(interval_repr: object) -> tuple[str, int, int, str]:
    text = str(interval_repr)
    chrom, rest = text.split(":", 1)
    span, *strand_part = rest.split("(")
    start_s, end_s = span.split("-", 1)
    strand = "+"
    if strand_part:
        strand = strand_part[0].rstrip(")") or "+"
    return chrom, int(start_s), int(end_s), strand


def _default_summary_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.summary.tsv")
    return output_path.with_name(f"{output_path.name}.summary.tsv")


def _float_or_blank(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.8g}"


def _summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "std": None,
            "min": None,
            "max": None,
            "frac_gt0": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": float(arr.shape[0]),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "std": float(arr.std(ddof=1)) if arr.shape[0] > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_gt0": float((arr > 0).mean()),
    }


def _wilcoxon_pvalue(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if np.allclose(arr, 0.0):
        return None
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    try:
        return float(wilcoxon(arr, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def _write_summary(
    summary_values: dict[
        tuple[str, str, int, str, int], dict[str, list[float]]
    ],
    summary_path: Path,
    background_mode: str,
) -> int:
    stat_names = ("mean", "median", "q25", "q75", "std", "min", "max", "frac_gt0")
    header = [
        "motif",
        "motif_sequence",
        "offset",
        "head",
        "head_index",
        "n",
        "background_mode",
    ]
    for metric in _SUMMARY_METRICS:
        header.extend([f"{metric}_{stat}" for stat in stat_names])
        header.append(f"{metric}_wilcoxon_p")

    rows_written = 0
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for key in sorted(summary_values):
            motif_name, motif_seq, offset, head_name, head_idx = key
            first_stats = _summarize_values(summary_values[key][_SUMMARY_METRICS[0]])
            row = [
                motif_name,
                motif_seq,
                offset,
                head_name,
                head_idx,
                int(first_stats["n"] or 0),
                background_mode,
            ]
            for metric in _SUMMARY_METRICS:
                metric_stats = _summarize_values(summary_values[key][metric])
                row.extend(_float_or_blank(metric_stats[stat]) for stat in stat_names)
                row.append(_float_or_blank(_wilcoxon_pvalue(summary_values[key][metric])))
            writer.writerow(row)
            rows_written += 1
    return rows_written


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()
    motifs = _collect_motifs(args)
    summary_output = args.summary_output or _default_summary_path(args.output)
    logger.info(
        "Scoring %d motif(s): %s",
        len(motifs),
        ", ".join(name for name, _ in motifs),
    )

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    ensemble = ModelEnsemble(args.checkpoint_dir.resolve(), device=device, fold=args.fold)
    cfg = ensemble.cerberus_config
    model = ensemble[str(args.fold)]
    accessibility_model = getattr(model, "chrombpnet_wo_bias", None)
    use_chrombpnet_accessibility_only = args.chrombpnet_accessibility_only
    if use_chrombpnet_accessibility_only is None:
        use_chrombpnet_accessibility_only = accessibility_model is not None

    if use_chrombpnet_accessibility_only:
        if accessibility_model is None:
            raise ValueError(
                "--chrombpnet-accessibility-only requested, but the model has no "
                "chrombpnet_wo_bias branch."
            )
        model = accessibility_model
        logger.info("Scoring accessibility-only branch: %s", model.__class__.__name__)
    model.to(device)
    model.eval()

    output_channels = list(cfg.model_config_.model_args.get("output_channels", []))
    if not output_channels:
        output_channels = [f"channel_{i}" for i in range(model.n_output_channels)]

    dataset = _make_interval_dataset(ensemble, args.intervals_path.resolve(), args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count_pseudocount = float(cfg.model_config_.count_pseudocount)
    rows_written = 0
    summary_values: dict[tuple[str, str, int, str, int], dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "index",
                "chrom",
                "start",
                "end",
                "strand",
                "background_mode",
                "motif",
                "motif_sequence",
                "offset",
                "insert_start",
                "insert_end",
                "head",
                "head_index",
                "no_motif_log_count",
                "motif_log_count",
                "delta_log_count",
                "no_motif_exp_log_count",
                "motif_exp_log_count",
                "delta_exp_log_count",
                "no_motif_count_minus_pseudocount",
                "motif_count_minus_pseudocount",
                "delta_count_minus_pseudocount",
            ]
        )

        exported = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].float()
                intervals = list(batch["intervals"])
                if exported >= args.n_examples:
                    break
                remaining = args.n_examples - exported
                if inputs.shape[0] > remaining:
                    inputs = inputs[:remaining]
                    intervals = intervals[:remaining]

                background_inputs = _make_background_inputs(
                    inputs,
                    mode=args.background_mode,
                    seed=args.seed,
                    global_start=exported,
                ).to(device)
                no_motif_log_counts = _prediction_log_counts(model, background_inputs)
                no_motif_exp = torch.exp(no_motif_log_counts)
                no_motif_minus_pc = (
                    no_motif_exp - count_pseudocount
                ).clamp_min(0.0)

                for motif_name, motif_seq in motifs:
                    for offset in args.offset:
                        mutated, insert_start, insert_end = _insert_motif(
                            background_inputs, motif_seq, offset
                        )
                        motif_log_counts = _prediction_log_counts(model, mutated)
                        delta_log = motif_log_counts - no_motif_log_counts
                        motif_exp = torch.exp(motif_log_counts)
                        delta_exp = motif_exp - no_motif_exp
                        motif_minus_pc = (
                            motif_exp - count_pseudocount
                        ).clamp_min(0.0)
                        delta_minus_pc = motif_minus_pc - no_motif_minus_pc

                        for row_idx, interval_repr in enumerate(intervals):
                            chrom, start, end, strand = _parse_interval_repr(
                                interval_repr
                            )
                            global_idx = exported + row_idx
                            for head_idx in range(no_motif_log_counts.shape[1]):
                                head_name = (
                                    output_channels[head_idx]
                                    if head_idx < len(output_channels)
                                    else f"channel_{head_idx}"
                                )
                                summary_key = (
                                    motif_name,
                                    motif_seq,
                                    offset,
                                    head_name,
                                    head_idx,
                                )
                                summary_values[summary_key]["delta_log_count"].append(
                                    float(delta_log[row_idx, head_idx].item())
                                )
                                summary_values[summary_key]["delta_exp_log_count"].append(
                                    float(delta_exp[row_idx, head_idx].item())
                                )
                                summary_values[summary_key][
                                    "delta_count_minus_pseudocount"
                                ].append(float(delta_minus_pc[row_idx, head_idx].item()))
                                writer.writerow(
                                    [
                                        global_idx,
                                        chrom,
                                        start,
                                        end,
                                        strand,
                                        args.background_mode,
                                        motif_name,
                                        motif_seq,
                                        offset,
                                        insert_start,
                                        insert_end,
                                        head_name,
                                        head_idx,
                                        f"{no_motif_log_counts[row_idx, head_idx].item():.8g}",
                                        f"{motif_log_counts[row_idx, head_idx].item():.8g}",
                                        f"{delta_log[row_idx, head_idx].item():.8g}",
                                        f"{no_motif_exp[row_idx, head_idx].item():.8g}",
                                        f"{motif_exp[row_idx, head_idx].item():.8g}",
                                        f"{delta_exp[row_idx, head_idx].item():.8g}",
                                        f"{no_motif_minus_pc[row_idx, head_idx].item():.8g}",
                                        f"{motif_minus_pc[row_idx, head_idx].item():.8g}",
                                        f"{delta_minus_pc[row_idx, head_idx].item():.8g}",
                                    ]
                                )
                                rows_written += 1

                exported += inputs.shape[0]
                if exported >= args.n_examples:
                    break

    logger.info("Wrote %d motif perturbation rows to %s", rows_written, args.output)
    summary_rows = _write_summary(
        summary_values, summary_output.resolve(), args.background_mode
    )
    logger.info("Wrote %d motif summary rows to %s", summary_rows, summary_output)


if __name__ == "__main__":
    main()
