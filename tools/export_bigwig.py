#!/usr/bin/env python
"""Export model predictions to a BigWig file (genome-wide or specific regions).

Usage:
    # Genome-wide
    python tools/export_bigwig.py path/to/model_dir --output predictions.bw

    # Single region
    python tools/export_bigwig.py path/to/model_dir --output predictions.bw --region chr1:1000000-2000000

    # Regions from a BED file
    python tools/export_bigwig.py path/to/model_dir --output predictions.bw --regions-bed peaks.bed
"""
import argparse
import logging
import re
from pathlib import Path

import torch

import cerberus
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.predict_bigwig import predict_to_bigwig
from cerberus.interval import Interval

logger = logging.getLogger(__name__)


def _parse_region(region_str: str) -> Interval:
    """Parse 'chr1:1000000-2000000' into an Interval."""
    match = re.match(r"^(\w+):(\d+)-(\d+)$", region_str)
    if not match:
        raise ValueError(
            f"Invalid region format: '{region_str}'. Expected 'chrom:start-end' "
            f"(e.g. 'chr1:1000000-2000000')."
        )
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start >= end:
        raise ValueError(f"Region start ({start}) must be less than end ({end}).")
    return Interval(chrom, start, end, "+")


def _parse_bed(bed_path: str) -> list[Interval]:
    """Parse a BED file into a list of Intervals (supports .gz)."""
    import gzip

    path = Path(bed_path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    opener = gzip.open if path.suffix == ".gz" else open
    regions: list[Interval] = []
    with opener(path, "rt") as f:  # type: ignore[call-overload]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            regions.append(Interval(chrom, start, end, "+"))

    if not regions:
        raise ValueError(f"No regions found in BED file: {bed_path}")
    return regions


def main():
    parser = argparse.ArgumentParser(
        description="Export genome-wide model predictions to a BigWig file."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.bw",
        help="Output BigWig filename (default: predictions.bw).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window predictions. Defaults to output_len // 2.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for prediction (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--use_folds",
        type=str,
        default=None,
        help="Folds to use (e.g. 'test', 'test+val', 'all'). Default depends on model type.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Single region to predict (e.g. 'chr1:1000000-2000000').",
    )
    parser.add_argument(
        "--regions-bed",
        type=str,
        default=None,
        help="BED file with regions to predict (columns: chrom, start, end).",
    )
    args = parser.parse_args()

    if args.region and args.regions_bed:
        parser.error("--region and --regions-bed are mutually exclusive.")

    cerberus.setup_logging()

    # 1. Resolve device
    if args.device:
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # 2. Load Model Ensemble
    logger.info(f"Loading model from {args.model_path}...")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    tests_data = project_root / "tests/data"
    search_paths = [tests_data, project_root, Path.cwd()]
    ensemble = ModelEnsemble(args.model_path, device=device, search_paths=search_paths)

    # 3. Parse use_folds
    use_folds = None
    if args.use_folds:
        use_folds = []
        parts = re.split(r"[+,]", args.use_folds)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p == "all":
                use_folds.extend(["train", "test", "val"])
            else:
                use_folds.append(p)
        use_folds = list(dict.fromkeys(use_folds))

    logger.info(f"Using folds configuration: {use_folds if use_folds else 'Default'}")

    # 4. Configure Dataset
    logger.info("Configuring dataset...")
    cerberus_config = ensemble.cerberus_config
    data_config = cerberus_config["data_config"]
    genome_config = cerberus_config["genome_config"]

    # No observed targets needed for bigwig export — avoid loading target extractors
    data_config["targets"] = {}

    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=None,
        in_memory=False,
        is_train=False,
    )

    # 5. Parse regions (if any)
    regions: list[Interval] | None = None
    if args.region:
        regions = [_parse_region(args.region)]
        logger.info(f"Predicting single region: {regions[0]}")
    elif args.regions_bed:
        regions = _parse_bed(args.regions_bed)
        logger.info(f"Predicting {len(regions)} regions from {args.regions_bed}")

    # 6. Run prediction and write BigWig
    output_path = Path(args.output)
    mode = f"{len(regions)} region(s)" if regions else "genome-wide"
    logger.info(f"Starting {mode} prediction (stride={args.stride}, batch_size={args.batch_size})...")

    predict_to_bigwig(
        output_path=output_path,
        dataset=dataset,
        model_ensemble=ensemble,
        stride=args.stride,
        use_folds=use_folds,
        batch_size=args.batch_size,
        regions=regions,
    )

    logger.info(f"BigWig written to {output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
