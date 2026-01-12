#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import dataclasses
import csv
import gzip

from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.samplers import IntervalSampler
from cerberus.output import ProfileCountOutput

def main():
    parser = argparse.ArgumentParser(description="Export predicted vs observed log-counts to TSV.")
    parser.add_argument("model_path", type=str, help="Path to the trained model directory.")
    parser.add_argument("peaks", type=str, help="Path to the peak file (bed/narrowPeak).")
    parser.add_argument("bigwig", type=str, help="Path to the bigwig file for observed counts.")
    parser.add_argument("--output", type=str, default="predictions.tsv", help="Output filename (TSV).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")

    args = parser.parse_args()
    
    # 1. Load Model Ensemble
    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Provide search paths for resolving files (e.g. genome fasta) from other environments
    # We include tests/data as a likely location for test resources relative to project root
    search_paths = [Path("tests/data"), Path.cwd()]
    ensemble = ModelEnsemble(args.model_path, device=device, search_paths=search_paths)
    
    # 2. Configure Dataset
    print("Configuring dataset...")
    cerberus_config = ensemble.cerberus_config
    data_config = cerberus_config["data_config"]
    genome_config = cerberus_config["genome_config"]
    
    # Override targets with provided bigwig
    if not data_config["targets"]:
        print("Warning: Model data_config has no targets. Adding 'default' target.")
        data_config["targets"] = {"default": Path(args.bigwig)}
    else:
        for key in data_config["targets"]:
            data_config["targets"][key] = Path(args.bigwig)
            print(f"Overriding target '{key}' with {args.bigwig}")
            
    # Create Dataset
    dataset = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=None,
        in_memory=False,
        is_train=False 
    )
    
    # 3. Load Peaks
    print(f"Loading peaks from {args.peaks}...")
    padded_size = data_config["input_len"]
    
    # Create a minimal IntervalSampler
    dummy_folds = [{} for _ in range(5)]
    dummy_exclude = {}
    
    peak_sampler = IntervalSampler(
        file_path=Path(args.peaks),
        chrom_sizes=genome_config["chrom_sizes"],
        padded_size=padded_size,
        exclude_intervals=dummy_exclude,
        folds=dummy_folds
    )
    
    print(f"Loaded {len(peak_sampler)} intervals.")
    if len(peak_sampler) == 0:
        print("No intervals found. Exiting.")
        return

    # 4. Predict and Collect
    print("Running prediction...")
    
    results = [] # List of tuples (chrom, start, end, strand, pred, obs)
    
    # Use all folds to ensure coverage
    batch_gen = ensemble.predict_intervals_batched(
        peak_sampler,
        dataset,
        use_folds=["test", "val", "train"],
        aggregation="model",
        batch_size=args.batch_size
    )
    
    for batch_output, batch_intervals in batch_gen:
        # 4a. Get Predicted Log Counts
        preds_batch = None
        if hasattr(batch_output, "log_counts"):
            log_counts = batch_output.log_counts
            if log_counts.ndim == 2 and log_counts.shape[1] > 1:
                pred_log_total = torch.logsumexp(log_counts, dim=1)
            else:
                pred_log_total = log_counts.flatten()
            preds_batch = pred_log_total.cpu().numpy()
            
        elif hasattr(batch_output, "log_rates"):
             log_rates = batch_output.log_rates
             if log_rates.shape[1] > 1:
                 pred_log_total = torch.logsumexp(log_rates.flatten(start_dim=1), dim=-1)
             else:
                 pred_log_total = torch.logsumexp(log_rates, dim=(1,2)).flatten()
             preds_batch = pred_log_total.cpu().numpy()
        else:
             print("Warning: Model output does not have log_counts or log_rates. Skipping batch.")
             continue

        # 4b. Get Observed Log Counts
        targets_list = []
        for interval in batch_intervals:
            data = dataset.get_interval(interval)
            targets_list.append(data["targets"])
            
        targets = torch.stack(targets_list)
        obs_total = targets.sum(dim=(1, 2))
        obs_log_total = torch.log1p(obs_total)
        obs_batch = obs_log_total.numpy()
        
        # 4c. Collect Metadata
        for i, interval in enumerate(batch_intervals):
            # Interval here is the input interval (padded/centered)
            # Should we output the original peak interval?
            # The IntervalSampler stores centered intervals.
            # But the 'interval' object has the coordinates used for prediction.
            # Let's use that.
            results.append((
                interval.chrom,
                interval.start,
                interval.end,
                interval.strand,
                preds_batch[i],
                obs_batch[i]
            ))

    if not results:
        print("No predictions generated.")
        return

    # 5. Write to TSV
    output_path = Path(args.output)
    print(f"Writing results to {output_path}...")
    
    if output_path.suffix == ".gz":
        f = gzip.open(output_path, "wt", newline="")
    else:
        f = open(output_path, "w", newline="")
        
    try:
        writer = csv.writer(f, delimiter="\t")
        # Header
        writer.writerow(["chrom", "start", "end", "strand", "predicted_log_count", "observed_log_count"])
        # Rows
        writer.writerows(results)
    finally:
        f.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
