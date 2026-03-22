#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import torch
import csv
import gzip
import re
import json

import cerberus
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.samplers import IntervalSampler, MultiSampler, create_sampler
from cerberus.genome import create_genome_folds
from cerberus.exclude import get_exclude_intervals
from cerberus.output import compute_total_log_counts, compute_obs_log_counts
from cerberus.config import get_log_count_params, SamplerConfig
from cerberus.module import instantiate_metrics_and_loss

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Export predicted vs observed log-counts to TSV.")
    parser.add_argument("model_path", type=str, help="Path to the trained model directory.")
    parser.add_argument("peaks", type=str, help="Path to the peak file (bed/narrowPeak).")
    parser.add_argument("bigwig", type=str, help="Path to the bigwig file for observed counts.")
    parser.add_argument("--output", type=str, default="predictions.tsv", help="Output filename (TSV).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu).")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to process (default: all).")
    parser.add_argument("--use_folds", type=str, default=None, help="Folds to use (e.g. 'test', 'test+val'). Default depends on model type.")
    parser.add_argument(
        "--eval-split", type=str, default="test", choices=["test", "val", "train", "all"],
        help="Which chromosome split to evaluate on: 'test' (default, held-out test chromosomes), "
             "'val' (validation chromosomes), 'train' (training chromosomes), or 'all' (every chromosome). "
             "Use 'all' only for exploratory analysis — it includes training chromosomes and inflates metrics."
    )
    parser.add_argument("--include-background", action="store_true", help="Include complexity-matched background (non-peak) intervals alongside peaks, replicating the training evaluation setup. Requires the model to have been trained with a 'peak' sampler. Adds an 'interval_source' column to the output (sampler class name).")
    parser.add_argument("--background-ratio", type=float, default=None, help="Ratio of background intervals to peaks (default: taken from model's sampler config, typically 1.0).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for background interval generation (only used with --include-background).")

    args = parser.parse_args()

    cerberus.setup_logging()

    # 1. Load Model Ensemble
    logger.info(f"Loading model from {args.model_path}...")

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

    # Provide search paths for resolving files (e.g. genome fasta) from other environments
    # We include tests/data as a likely location for test resources relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    tests_data = project_root / "tests/data"
    search_paths = [tests_data, project_root, Path.cwd()]
    ensemble = ModelEnsemble(args.model_path, device=device, search_paths=search_paths)

    # Configure use_folds
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

    # 2. Configure Dataset
    logger.info("Configuring dataset...")
    cerberus_config = ensemble.cerberus_config
    data_config = cerberus_config.data_config
    genome_config = cerberus_config.genome_config
    model_config = cerberus_config.model_config_

    # Override targets with provided bigwig
    if not data_config.targets:
        raise ValueError("Model configuration has no targets defined.")

    if len(data_config.targets) > 1:
        raise ValueError(f"Model has multiple targets ({list(data_config.targets.keys())}), which is not supported by this script.")

    # Override the single target — data_config is frozen, so use model_copy
    key = list(data_config.targets.keys())[0]
    new_targets = dict(data_config.targets)
    new_targets[key] = Path(args.bigwig)
    data_config = data_config.model_copy(update={"targets": new_targets})
    logger.info(f"Overriding target '{key}' with {args.bigwig}")

    # Create Dataset
    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=None,
        in_memory=False,
        is_train=False
    )

    # 3. Load Peaks (and optionally background), then restrict to the requested chromosome split.
    #
    # By default (--eval-split test) only held-out test chromosomes are evaluated so that
    # metrics.json reflects genuine generalisation performance.  Pass --eval-split all to
    # reproduce the old behaviour of scoring every peak regardless of fold membership.
    logger.info(f"Loading peaks from {args.peaks}...")
    padded_size = data_config.input_len

    # Build fold definitions — needed for split filtering in every code path.
    folds = create_genome_folds(
        genome_config.chrom_sizes,
        genome_config.fold_type,
        genome_config.fold_args,
    )
    exclude_intervals = get_exclude_intervals(genome_config.exclude_intervals)

    # Resolve test_fold / val_fold indices from the stored genome config.
    fold_args = genome_config.fold_args
    test_fold_idx: int | None = fold_args["test_fold"]
    val_fold_idx: int | None = fold_args["val_fold"]

    if args.eval_split != "all" and test_fold_idx is None and val_fold_idx is None:
        raise ValueError(
            f"--eval-split '{args.eval_split}' requires 'test_fold' and/or 'val_fold' to be "
            f"defined in the model's genome config fold_args, but neither was found. "
            f"Use --eval-split all to evaluate on every chromosome."
        )

    if args.include_background:
        # Replicate training evaluation: peaks + complexity-matched background.
        # Requires a 'peak' sampler config stored in the model checkpoint.
        sampler_config = cerberus_config.sampler_config
        if sampler_config.sampler_type != "peak":
            raise ValueError(
                f"--include-background only supports 'peak' sampler type, "
                f"got '{sampler_config.sampler_type}'. "
                f"Only models trained with PeakSampler can generate complexity-matched background."
            )

        # Override peaks path with the user-provided file; keep all other sampler args.
        # Override padded_size to input_len: the stored training padded_size may differ
        # (e.g. larger window for complexity computation), but predict_intervals_batched
        # requires intervals pre-sized to input_len.
        orig_sampler_args = sampler_config.sampler_args
        bg_ratio = args.background_ratio if args.background_ratio is not None else orig_sampler_args["background_ratio"]
        bg_sampler_args = {
            "intervals_path": Path(args.peaks),
            "background_ratio": bg_ratio,
            "complexity_center_size": orig_sampler_args["complexity_center_size"] if "complexity_center_size" in orig_sampler_args else None,
        }
        bg_sampler_config = SamplerConfig(
            sampler_type=sampler_config.sampler_type,
            padded_size=padded_size,
            sampler_args=bg_sampler_args,
        )

        logger.info(
            f"Building PeakSampler with background_ratio={bg_ratio} "
            f"(seed={args.seed})..."
        )
        combined_sampler = create_sampler(
            bg_sampler_config,
            genome_config.chrom_sizes,
            folds=folds,
            exclude_intervals=exclude_intervals,
            fasta_path=genome_config.fasta_path,
            seed=args.seed,
        )

        # PeakSampler is always a MultiSampler; get_interval_source() lives there.
        if not isinstance(combined_sampler, MultiSampler):
            raise RuntimeError("Expected PeakSampler to be a MultiSampler instance.")

        # Restrict to the requested chromosome split so backgrounds come from the
        # same fold as the peaks (matching the training split_folds() behaviour).
        if args.eval_split != "all":
            train_s, val_s, test_s = combined_sampler.split_folds(test_fold_idx, val_fold_idx)
            split_sampler: MultiSampler = {"test": test_s, "val": val_s, "train": train_s}[args.eval_split]
            if not isinstance(split_sampler, MultiSampler):
                raise RuntimeError(f"Expected split sampler to be a MultiSampler.")
            combined_sampler = split_sampler

        n_combined = len(combined_sampler)
        all_intervals = [combined_sampler[i] for i in range(n_combined)]
        all_interval_source = [combined_sampler.get_interval_source(i) for i in range(n_combined)]
        sampler_to_use: IntervalSampler | list = all_intervals
        n_peaks = sum(1 for s in all_interval_source if s == "IntervalSampler")
        logger.info(
            f"Combined sampler ({args.eval_split} split): {n_peaks} peaks + "
            f"{n_combined - n_peaks} background intervals."
        )
    else:
        # Peaks only.
        peak_sampler = IntervalSampler(
            file_path=Path(args.peaks),
            chrom_sizes=genome_config.chrom_sizes,
            padded_size=padded_size,
            folds=folds,
            exclude_intervals=exclude_intervals,
        )

        if args.eval_split != "all":
            train_sp, val_sp, test_sp = peak_sampler.split_folds(test_fold_idx, val_fold_idx)
            chosen = {"test": test_sp, "val": val_sp, "train": train_sp}[args.eval_split]
            all_intervals = list(chosen)
        else:
            all_intervals = list(peak_sampler)

        all_interval_source = ["IntervalSampler"] * len(all_intervals)
        sampler_to_use = all_intervals

    if len(all_intervals) == 0:
        logger.warning("No intervals found. Exiting.")
        return
    logger.info(f"Loaded {len(all_intervals)} intervals ({args.eval_split} split).")

    # 3.5 Setup Metrics and Loss
    logger.info("Configuring metrics and loss...")
    metrics, criterion = instantiate_metrics_and_loss(model_config, device=device)

    # Determine pseudocount parameters from the loss class.
    # MSE losses train log_counts in log(count + pseudocount) space;
    # Poisson/NB losses use log(count) directly.
    log_counts_include_pseudocount, count_pseudocount = get_log_count_params(model_config)

    total_loss = 0.0
    total_samples = 0

    # 4. Predict and Collect
    logger.info("Running prediction...")

    results = []  # List of tuples (chrom, start, end, strand, pred_interval, pred, obs[, interval_source])

    # Use selected folds
    batch_gen = ensemble.predict_intervals_batched(
        sampler_to_use,
        dataset,
        use_folds=use_folds,
        aggregation="model",
        batch_size=args.batch_size
    )

    batch_count = 0
    interval_idx = 0  # Tracks position in all_interval_source for the current batch
    output_len = data_config.output_len

    for batch_output, batch_intervals in batch_gen:
        if args.max_batches is not None and batch_count >= args.max_batches:
            logger.info(f"Reached max_batches ({args.max_batches}). Stopping.")
            break
        batch_count += 1

        # 4a. Get Predicted Log Counts
        try:
            pred_log_total = compute_total_log_counts(
                batch_output,
                log_counts_include_pseudocount=log_counts_include_pseudocount,
                pseudocount=count_pseudocount,
            )
            preds_batch = pred_log_total.cpu().numpy()
        except ValueError:
            logger.warning("Model output does not have log_counts or log_rates. Skipping batch.")
            continue

        # 4b. Get Observed Log Counts
        targets_list = []
        raw_obs_list = []

        for interval in batch_intervals:
            # 1. Get transformed data for metrics/loss
            data = dataset.get_interval(interval)
            targets_list.append(data["targets"])

            # 2. Get raw data for exported "observed" counts
            raw_target = dataset.get_raw_targets(interval, crop_to_output_len=True)
            raw_obs_list.append(raw_target)

        targets = torch.stack(targets_list)

        # Calculate observed total from RAW counts, in the same log-space as the loss.
        raw_obs = torch.stack(raw_obs_list)
        obs_log_total = compute_obs_log_counts(
            raw_obs,
            target_scale=data_config.target_scale,
            log_counts_include_pseudocount=log_counts_include_pseudocount,
            pseudocount=count_pseudocount,
        )
        obs_batch = obs_log_total.numpy()

        # 4c. Update Metrics and Loss
        targets_device = targets.to(device)
        with torch.no_grad():
            metrics.update(batch_output, targets_device)
            batch_loss = criterion(batch_output, targets_device)
            total_loss += batch_loss.item() * targets.size(0)
            total_samples += targets.size(0)

        # 4d. Collect Metadata
        for i, interval in enumerate(batch_intervals):
            # Interval here is the input interval (padded/centered)
            # We also calculate the output interval (prediction window)
            output_int = interval.center(output_len)
            pred_str = f"{output_int.chrom}:{output_int.start}-{output_int.end}"

            row = (
                interval.chrom,
                interval.start,
                interval.end,
                interval.strand,
                pred_str,
                preds_batch[i],
                obs_batch[i],
            )
            if args.include_background:
                row = row + (all_interval_source[interval_idx + i],)
            results.append(row)

        interval_idx += len(batch_intervals)

    if not results:
        logger.warning("No predictions generated.")
        return

    # 5. Write to TSV
    output_path = Path(args.output)
    logger.info(f"Writing results to {output_path}...")

    if output_path.suffix == ".gz":
        f = gzip.open(output_path, "wt", newline="")
    else:
        f = open(output_path, "w", newline="")

    try:
        writer = csv.writer(f, delimiter="\t")
        header = ["chrom", "start", "end", "strand", "pred_interval", "predicted_log_count", "observed_log_count"]
        if args.include_background:
            header.append("interval_source")
        writer.writerow(header)
        writer.writerows(results)
    finally:
        f.close()

    # 6. Compute and Save Metrics
    logger.info("Computing final metrics...")
    final_metrics = metrics.compute()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    summary_metrics = {k: v.item() for k, v in final_metrics.items()}
    summary_metrics["loss"] = avg_loss

    # Get params
    first_model = next(iter(ensemble.values()))
    num_params = sum(p.numel() for p in first_model.parameters())

    summary = {
        "model_name": str(args.model_path),
        "parameters": num_params,
        "metrics": summary_metrics
    }

    summary_path = output_path.with_suffix("").with_suffix(".metrics.json")
    logger.info(f"Writing metrics to {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done.")

if __name__ == "__main__":
    main()
