#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import csv
import gzip
import re
import json

from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.samplers import IntervalSampler
from cerberus.output import ProfileCountOutput, ProfileLogRates, ProfileLogits
from cerberus.config import import_class

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

    args = parser.parse_args()
    
    # 1. Load Model Ensemble
    print(f"Loading model from {args.model_path}...")
    
    if args.device:
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
        
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
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
        use_folds = list(set(use_folds))
        
    print(f"Using folds configuration: {use_folds if use_folds else 'Default'}")

    # 2. Configure Dataset
    print("Configuring dataset...")
    cerberus_config = ensemble.cerberus_config
    data_config = cerberus_config["data_config"]
    genome_config = cerberus_config["genome_config"]
    model_config = cerberus_config["model_config"]
    
    # Override targets with provided bigwig
    if not data_config["targets"]:
        raise ValueError("Model configuration has no targets defined.")

    if len(data_config["targets"]) > 1:
        raise ValueError(f"Model has multiple targets ({list(data_config['targets'].keys())}), which is not supported by this script.")

    # Override the single target
    key = list(data_config["targets"].keys())[0]
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
    peak_sampler = IntervalSampler(
        file_path=Path(args.peaks),
        chrom_sizes=genome_config["chrom_sizes"],
        padded_size=padded_size
    )
    
    print(f"Loaded {len(peak_sampler)} intervals.")
    if len(peak_sampler) == 0:
        print("No intervals found. Exiting.")
        return

    # 3.5 Setup Metrics and Loss
    print("Configuring metrics and loss...")
    
    metrics_cls = import_class(model_config["metrics_cls"])
    metrics_args = model_config["metrics_args"]
    metrics = metrics_cls(**metrics_args).to(device)

    loss_cls = import_class(model_config["loss_cls"])
    loss_args = model_config["loss_args"]
    criterion = loss_cls(**loss_args).to(device)

    total_loss = 0.0
    total_samples = 0

    # 4. Predict and Collect
    print("Running prediction...")
    
    results = [] # List of tuples (chrom, start, end, strand, pred, obs)
    
    # Use selected folds
    batch_gen = ensemble.predict_intervals_batched(
        peak_sampler,
        dataset,
        use_folds=use_folds,
        aggregation="model",
        batch_size=args.batch_size
    )
    
    batch_count = 0
    output_len = data_config["output_len"]

    for batch_output, batch_intervals in batch_gen:
        if args.max_batches is not None and batch_count >= args.max_batches:
            print(f"Reached max_batches ({args.max_batches}). Stopping.")
            break
        batch_count += 1

        # 4a. Get Predicted Log Counts
        preds_batch = None
        if isinstance(batch_output, ProfileCountOutput):
            log_counts = batch_output.log_counts
            if log_counts.ndim == 2 and log_counts.shape[1] > 1:
                pred_log_total = torch.logsumexp(log_counts, dim=1)
            else:
                pred_log_total = log_counts.flatten()
            preds_batch = pred_log_total.cpu().numpy()
            
        elif isinstance(batch_output, ProfileLogRates):
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
        
        # 4c. Update Metrics and Loss
        targets_device = targets.to(device)
        metrics.update(batch_output, targets_device)
        
        with torch.no_grad():
             batch_loss = criterion(batch_output, targets_device)
             total_loss += batch_loss.item() * targets.size(0)
             total_samples += targets.size(0)
        
        # 4d. Collect Metadata
        for i, interval in enumerate(batch_intervals):
            # Interval here is the input interval (padded/centered)
            # We also calculate the output interval (prediction window)
            output_int = interval.center(output_len)
            pred_str = f"{output_int.chrom}:{output_int.start}-{output_int.end}"

            results.append((
                interval.chrom,
                interval.start,
                interval.end,
                interval.strand,
                pred_str,
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
        writer.writerow(["chrom", "start", "end", "strand", "pred_interval", "predicted_log_count", "observed_log_count"])
        # Rows
        writer.writerows(results)
    finally:
        f.close()
        
    # 6. Compute and Save Metrics
    print("Computing final metrics...")
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
    
    summary_path = str(output_path) + ".metrics.json"
    print(f"Writing metrics to {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
