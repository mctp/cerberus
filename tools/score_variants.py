#!/usr/bin/env python
"""Score variant effects using a trained Cerberus model.

For each variant in a VCF or TSV file, this tool predicts the effect on
model outputs by comparing predictions on reference vs alternative allele
sequences.  Supports single-fold and multi-fold ensemble models.

Usage:
    # From a VCF
    python tools/score_variants.py path/to/model_dir \\
        --vcf variants.vcf.gz --output effects.tsv

    # From a tab-delimited variant file
    python tools/score_variants.py path/to/model_dir \\
        --variants variants.tsv --output effects.tsv

    # With region filter and custom folds
    python tools/score_variants.py path/to/model_dir \\
        --vcf variants.vcf.gz --output effects.tsv \\
        --region chr1:1000000-2000000 --use-folds test+val
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
from collections.abc import Iterator
from pathlib import Path

import pyfaidx

import cerberus
from cerberus.interval import resolve_interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.predict_variants import VariantResult, score_variants_from_ensemble
from cerberus.utils import parse_use_folds, resolve_device
from cerberus.variants import load_variants, load_vcf

logger = logging.getLogger(__name__)


def _load_variants(args: argparse.Namespace) -> Iterator:
    """Load variants from VCF or TSV based on CLI args."""
    if args.vcf is not None:
        region = resolve_interval(args.region) if args.region else None
        return load_vcf(args.vcf, region=region)
    else:
        return load_variants(args.variants, zero_based=args.zero_based)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score variant effects using a trained Cerberus model."
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model directory (ModelEnsemble compatible).",
    )

    # -- Variant source (mutually exclusive) --
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--vcf",
        type=Path,
        default=None,
        help="Path to a VCF or BCF file (bgzipped + indexed for --region).",
    )
    source.add_argument(
        "--variants",
        type=Path,
        default=None,
        help=(
            "Path to a tab-delimited variant file with columns: "
            "chrom, pos, ref, alt (and optional id). "
            "Positions are 1-based by default; use --zero-based to change."
        ),
    )

    parser.add_argument(
        "--zero-based",
        action="store_true",
        default=False,
        help="Interpret --variants positions as 0-based (default: 1-based).",
    )

    # -- Output --
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("variant_effects.tsv"),
        help="Output TSV path (default: variant_effects.tsv). Supports .gz.",
    )

    # -- Region filter --
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "Restrict to variants in this region (e.g. 'chr1:1000000-2000000'). "
            "VCF must be indexed for region queries."
        ),
    )

    # -- FASTA override --
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help=(
            "Path to the reference genome FASTA. "
            "Default: auto-detected from model hparams (genome_config.fasta_path)."
        ),
    )

    # -- Inference options --
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of variants per inference batch (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: auto (default), cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--use_folds",
        type=str,
        default=None,
        help=(
            "Folds to use for ensemble prediction (e.g. 'test', 'test+val', 'all'). "
            "Default depends on model type."
        ),
    )

    return parser


def _write_results(
    results: Iterator[VariantResult],
    output_path: Path,
) -> int:
    """Write variant effect results to a TSV file.

    Returns the number of variants written.
    """
    opener = gzip.open if output_path.suffix == ".gz" else open
    n_written = 0
    header_written = False

    with opener(output_path, "wt", newline="") as f:  # type: ignore[call-overload]
        writer = csv.writer(f, delimiter="\t")

        for result in results:
            if not header_written:
                # Build header from the first result's effect keys
                metric_names = list(result.effects.keys())
                header = ["chrom", "pos_0based", "ref", "alt", "id"]
                for name in metric_names:
                    tensor = result.effects[name]
                    n_channels = tensor.numel()
                    if n_channels == 1:
                        header.append(name)
                    else:
                        for ch in range(n_channels):
                            header.append(f"{name}_ch{ch}")
                writer.writerow(header)
                header_written = True

            v = result.variant
            row: list[object] = [v.chrom, v.pos, v.ref, v.alt, v.id]
            for name in metric_names:
                tensor = result.effects[name]
                if tensor.numel() == 1:
                    row.append(f"{tensor.item():.6g}")
                else:
                    for val in tensor.flatten().tolist():
                        row.append(f"{val:.6g}")
            writer.writerow(row)
            n_written += 1

    return n_written


def main() -> None:
    cerberus.setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    # -- Load model --
    logger.info("Loading model from %s...", args.model_path)
    ensemble = ModelEnsemble(args.model_path, device=device)

    # -- FASTA --
    fasta = None
    if args.fasta is not None:
        fasta = pyfaidx.Fasta(str(args.fasta))
        logger.info("Using FASTA: %s", args.fasta)
    # else: score_variants_from_ensemble will open from config

    # -- Load variants --
    variants = _load_variants(args)

    # -- Use folds --
    use_folds = parse_use_folds(args.use_folds)
    logger.info("Using folds: %s", use_folds if use_folds else "default")

    # -- Score --
    results = score_variants_from_ensemble(
        ensemble=ensemble,
        variants=variants,
        fasta=fasta,
        batch_size=args.batch_size,
        use_folds=use_folds,
    )

    # -- Write output --
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = _write_results(results, output_path)

    logger.info("Wrote %d variant effects to %s", n_written, output_path)

    if fasta is not None:
        fasta.close()


if __name__ == "__main__":
    main()
