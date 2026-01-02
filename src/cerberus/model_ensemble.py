from pathlib import Path
import torch
from torch import nn
import numpy as np
import re
import dataclasses
from typing import Any, Iterable
import itertools
from collections import defaultdict

from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
    PredictConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_folds
from cerberus.entrypoints import instantiate
from cerberus.output import ModelOutput
from cerberus.interval import Interval


class ModelEnsemble(nn.ModuleDict):
    """
    Wraps a dictionary of models (fold_idx -> model) and manages
    selection based on genomic intervals.
    """
    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
        genome_config: GenomeConfig,
        device: torch.device,
    ):
        loader = _ModelManager(
            checkpoint_path, model_config, data_config, train_config, genome_config, device
        )
        models, folds = loader.load_models_and_folds()

        super().__init__(models)
        self.folds = folds
        self.output_len = data_config["output_len"]
        self.output_bin_size = data_config["output_bin_size"]

    @staticmethod
    def _unbatch_modeloutput(batched_output: ModelOutput, batch_size: int) -> list[dict[str, Any]]:
        """
        Splits a batched output (ModelOutput) into a list of individual intervals (list of dicts).
        Skips non-tensor fields (replicates them or ignores? Ignores for unbinding).
        """
        batched_output_dict = dataclasses.asdict(batched_output)

        unbatched_components = {}
        for key, val in batched_output_dict.items():
            if isinstance(val, torch.Tensor):
                unbatched_components[key] = list(torch.unbind(val, dim=0))
            else:
                # For metadata fields like out_interval, replicate
                unbatched_components[key] = [val] * batch_size
        
        # Reassemble into list of dicts
        result_list = []
        keys = unbatched_components.keys()
        for i in range(batch_size):
            item = {key: unbatched_components[key][i] for key in keys}
            result_list.append(item)
            
        return result_list

    def forward(
        self, 
        x: torch.Tensor, 
        intervals: list[Interval] | None = None, 
        use_folds: list[str] = ["test", "val"],
        aggregation: str = "model" # "model", "interval+model"
    ) -> ModelOutput:
        """
        Runs inference on applicable models based on intervals.
        
        Args:
            x: Input tensor (Batch, Channels, Length).
            intervals: List of Interval objects corresponding to x.
            use_folds: List of fold roles to include ('train', 'test', 'val').
            aggregation: Aggregation mode ("model", "interval+model").
            
        Returns:
            ModelOutput: The aggregated output.
        """
        # 1. Run models -> list[ModelOutput] (one per model, batched)
        batch_outputs = self._forward_models(x, intervals, use_folds)
        
        if not batch_outputs:
            raise RuntimeError("No model outputs generated.")

        if aggregation == "model":
            # Aggregate over models, return batched [aggregated]
            return self._aggregate_models(batch_outputs, method="mean")
        
        if aggregation == "interval+model":
            if intervals is None:
                raise ValueError("Intervals are required for interval aggregation.")

            # Center intervals to output length
            centered_intervals = [i.center(self.output_len) for i in intervals]

            # Aggregate over models
            aggregated_batch = self._aggregate_models(batch_outputs, method="mean")
            output_cls = type(aggregated_batch)
            
            # Unbatch
            unbatched = self._unbatch_modeloutput(aggregated_batch, len(centered_intervals))
            
            # Merge over intervals
            merged = self._aggregate_intervals(unbatched, centered_intervals, output_cls=output_cls)
            return merged

        raise ValueError(f"Unknown aggregation mode: {aggregation} (supported: 'model', 'interval+model')")

    def _forward_models(
        self, 
        x: torch.Tensor, 
        intervals: list[Interval] | None = None, 
        use_folds: list[str] = ["test", "val"]
    ) -> list[ModelOutput]:
        if not intervals:
            # Fallback: run all models if no intervals provided (or for single model)
            models_to_run = self.values()
        else:
            # Determine applicable models
            target_partitions = set()
            
            # For each interval, find which partition it belongs to
            # We assume all intervals in the batch belong to the same partition
            interval = intervals[0]
            for fold_idx, fold_map in enumerate(self.folds):
                if interval.chrom in fold_map:
                    if any(fold_map[interval.chrom].find((interval.start, interval.end - 1))):
                        target_partitions.add(fold_idx)
            
            models_to_load = set()
            if len(self.folds) > 0 and target_partitions:
                for p_idx in target_partitions:

                    # 1. 'test'
                    if "test" in use_folds:
                        models_to_load.add(p_idx)
                    
                    # 2. 'val'
                    if "val" in use_folds:
                        val_model_idx = (p_idx - 1) % len(self.folds)
                        models_to_load.add(val_model_idx)
                        
                    # 3. 'train'
                    if "train" in use_folds:
                        test_model = p_idx
                        val_model = (p_idx - 1) % len(self.folds)
                        for i in range(len(self.folds)):
                            if i != test_model and i != val_model:
                                models_to_load.add(i)
                
                # Retrieve models by index string
                models_to_run = []
                for idx in sorted(list(models_to_load)):
                    key = str(idx)
                    if key in self:
                        models_to_run.append(self[key])
            else:
                # If single model or no target partitions found, run all models
                models_to_run = self.values()

        batch_outputs = []
        with torch.no_grad():
            for model in models_to_run:
                out = model(x)
                batch_outputs.append(out)
        return batch_outputs


    def _aggregate_intervals(
        self,
        outputs: list[dict[str, Any]],
        intervals: list[Interval],
        output_cls: type[ModelOutput] | None = None,
    ) -> ModelOutput:
        """
        Aggregates overlapping predictions into a single merged output.
        """
        if not outputs or not intervals:
            raise ValueError("No outputs or intervals to aggregate")
        
        # Filter for keys that are tensors (ignore metadata like out_interval)
        keys = [k for k in outputs[0].keys() if isinstance(outputs[0][k], torch.Tensor)]

        # Compute merged interval, assuming all intervals are on the same chrom and strand
        chrom = intervals[0].chrom
        strand = intervals[0].strand
        min_start = min(i.start for i in intervals)
        max_end = max(i.end for i in intervals)
        merged_interval = Interval(chrom, min_start, max_end, strand)

        if len(merged_interval) % self.output_bin_size != 0:
             # This might happen if intervals are disjoint/weird, but assume user handles alignment
             pass
        
        aggregated_components = {}
        
        for key in keys:
            component_outputs = [out[key] for out in outputs] # list[Tensor]
            agg_comp = self._aggregate_tensor_track_values(
                component_outputs, intervals, merged_interval
            )
            # Convert numpy to tensor (CPU)
            aggregated_components[key] = torch.from_numpy(agg_comp)

        if output_cls is None:
            raise ValueError("output_cls must be provided to aggregate_intervals")

        return output_cls(**aggregated_components, out_interval=merged_interval)

    def _aggregate_tensor_track_values(
        self,
        outputs: list[torch.Tensor],
        intervals: list[Interval],
        merged_interval: Interval,
    ) -> np.ndarray:
        """
        Aggregates a list of tensors into a single merged array.
        """
        # Merged extent
        min_start = merged_interval.start
        max_end = merged_interval.end
        
        span_bp = max_end - min_start
        n_bins = span_bp // self.output_bin_size
        
        # outputs[0] is (C, L) or (C,)
        sample_out = outputs[0]
        n_channels = sample_out.shape[0]
        
        # Accumulators
        accumulator = np.zeros((n_channels, n_bins), dtype=np.float32)
        counts = np.zeros((1, n_bins), dtype=np.float32)
        
        for out_tensor, interval in zip(outputs, intervals):
            # Convert to numpy
            val = out_tensor.cpu().numpy() # (C, L) or (C,)
            
            # Check dimensions
            # Profile: last dim * bin_size == output_len
            is_profile = (val.shape[-1] * self.output_bin_size == self.output_len)
            
            # Relative start bin
            rel_start_bp = interval.start - min_start
            rel_start_bin = rel_start_bp // self.output_bin_size
            
            if is_profile:
                # val is (C, L_bins)
                l_bins = val.shape[-1]
                end_bin = rel_start_bin + l_bins
                
                # Bounds check (simple clipping or assume fits)
                # Since intervals are within merged_interval, it should fit unless precision issues
                
                accumulator[:, rel_start_bin : end_bin] += val
                counts[:, rel_start_bin : end_bin] += 1.0
            else:
                # Scalar (C,)
                # Broadcast to interval length to create a constant track over the interval
                # Interval length in bins
                int_bins = self.output_len // self.output_bin_size
                end_bin = rel_start_bin + int_bins
                
                # val is (C,) -> (C, int_bins)
                val_b = np.expand_dims(val, -1)
                
                accumulator[:, rel_start_bin : end_bin] += val_b
                counts[:, rel_start_bin : end_bin] += 1.0

        # Average
        counts = np.maximum(counts, 1.0)
        final_values = accumulator / counts
        
        return final_values

    def _aggregate_models(
        self, outputs: list[ModelOutput], method: str
    ) -> ModelOutput:
        """
        Aggregates a list of model outputs.
        
        Args:
            outputs: List of outputs from forward().
            method: Aggregation method ("mean" or "median").
            
        Returns:
            Aggregated ModelOutput.
        """
        if not outputs:
            raise ValueError("No outputs to aggregate.")

        output_dicts = [dataclasses.asdict(out) for out in outputs]
        # Filter tensor keys
        keys = [k for k in output_dicts[0].keys() if isinstance(output_dicts[0][k], torch.Tensor)]
        
        aggregated_elements = {}
        
        for key in keys:
            stacked = torch.stack([out[key] for out in output_dicts])
            
            if method == "mean":
                # mean over N_Models dimension (dim 0)
                aggregated_elements[key] = torch.mean(stacked, dim=0)
            elif method == "median":
                # median over N_Models dimension (dim 0)
                aggregated_elements[key] = torch.median(stacked, dim=0).values
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        
        # Reconstruct object
        cls = type(outputs[0])
        # Preserve out_interval if consistent? 
        # Usually for batched aggregation, out_interval is None or same.
        # We take the first one.
        out_int = outputs[0].out_interval
        
        return cls(**aggregated_elements, out_interval=out_int)

    def predict_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        predict_config: PredictConfig,
        device: str | None = None,
        batch_size: int = 64,
    ) -> ModelOutput:
        """
        Predicts and aggregates outputs for multiple intervals in batches.
        Always returns a single aggregated ModelOutput.
        """
        iterator = iter(intervals)
        try:
            first_interval = next(iterator)
        except StopIteration:
            raise ValueError("No intervals provided for prediction.")

        input_len = dataset.data_config["input_len"]
        output_len = dataset.data_config["output_len"]
        
        # Default to model aggregation, or interval+model if specified
        aggregation = predict_config.get("aggregation", "model")
        if aggregation not in ["model", "interval+model"]:
            aggregation = "model" # Enforce model aggregation if invalid/legacy value
        
        if device is None:
             device = "cuda" if torch.cuda.is_available() else "cpu"
        
        results = []
        output_cls = None
        
        # Chain back the first interval
        full_iterator = itertools.chain([first_interval], iterator)
        
        global_index = 0
        while True:
            # Get next batch
            batch_intervals = list(itertools.islice(full_iterator, batch_size))
            if not batch_intervals:
                break
                
            # 1. Prepare Batch Data
            inputs_list = []
            for interval in batch_intervals:
                if len(interval) != input_len:
                    raise ValueError(
                        f"Interval {interval} (index {global_index}) has length {len(interval)}, "
                        f"expected {input_len}."
                    )
                data = dataset.get_interval(interval)
                inputs_list.append(data["inputs"])
                global_index += 1

            # Stack into (Batch, Channels, Length)
            batch_inputs = torch.stack(inputs_list).to(device)

            # 2. Run Models (returns single ModelOutput)
            batched_output = self.forward(
                batch_inputs, 
                intervals=batch_intervals, 
                use_folds=predict_config["use_folds"],
                aggregation=aggregation
            )
            
            output_cls = type(batched_output)

            if aggregation == "interval+model":
                # Result is already merged for the batch
                out_dict = dataclasses.asdict(batched_output)
                out_int = batched_output.out_interval
                results.append((out_dict, out_int))
            else:
                # Result is batched
                unbatched = self._unbatch_modeloutput(batched_output, len(batch_intervals))
                for interval, output in zip(batch_intervals, unbatched):
                    output_interval = interval.center(output_len)
                    results.append((output, output_interval))

        # 4. Final Aggregation
        if not results:
             raise RuntimeError("No results generated.")
             
        outputs_list = [r[0] for r in results]
        intervals_list = [r[1] for r in results]
        
        merged = self._aggregate_intervals(
            outputs_list, 
            intervals_list, 
            output_cls=output_cls
        )
        return merged

    def predict_output_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        predict_config: PredictConfig,
        device: str | None = None,
        batch_size: int = 64,
    ) -> list[ModelOutput]:
        """
        Predicts outputs for a list of target intervals by tiling them with input intervals.

        For each target interval, this function generates the necessary input intervals to cover it,
        runs the prediction (using predict_intervals), and returns the aggregated result.

        Args:
            intervals: Iterable of target intervals.
            dataset: CerberusDataset instance containing data configuration.
            predict_config: PredictConfig instance containing 'stride'.
            device: Device to run models on.
            batch_size: Number of intervals to process in each batch.

        Returns:
            list[ModelOutput]: A list of ModelOutput objects.
        """
        input_len = dataset.data_config["input_len"]
        output_len = dataset.data_config["output_len"]
        stride = predict_config["stride"]

        if device is None:
             device = "cuda" if torch.cuda.is_available() else "cpu"

        offset = (input_len - output_len) // 2

        results = []

        for target in intervals:
            target_input_intervals = []
            current_start = target.start

            while current_start < target.end:
                # Calculate input interval corresponding to output window at current_start
                input_start = current_start - offset
                input_end = input_start + input_len

                target_input_intervals.append(
                    Interval(target.chrom, input_start, input_end, target.strand)
                )

                current_start += stride

            if target_input_intervals:
                prediction = self.predict_intervals(
                    target_input_intervals,
                    dataset,
                    predict_config,
                    device,
                    batch_size,
                )
                results.append(prediction)

        return results


class _ModelManager:
    """
    Manages loading and caching of models for different folds.
    """
    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
        genome_config: GenomeConfig,
        device: torch.device,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.genome_config = genome_config
        self.device = device
        
        self.cache: dict[str, nn.Module] = {}
        self.is_multifold = self._detect_multifold()
        
        if self.is_multifold:
            self.folds = create_genome_folds(
                genome_config["chrom_sizes"],
                genome_config["fold_type"],
                genome_config["fold_args"],
            )
        else:
            self.folds = []

    def _detect_multifold(self) -> bool:
        if self.checkpoint_path.is_dir():
            # Check for fold_0, fold_1, etc.
            return any((self.checkpoint_path / f"fold_{i}").exists() for i in range(2))
        return False

    def _select_best_checkpoint(self, checkpoints: list[Path]) -> Path:
        """
        Selects the best checkpoint from a list based on validation loss.
        Assumes filename format like '...val_loss=0.1234...' or '...val_loss-0.1234...'.
        If no metric found, sorts by name.
        """
        def get_val_loss(p: Path) -> float:
            # Pattern to match val_loss=0.1234
            # Usually PyTorch Lightning formats it as key=value
            # Use specific regex to avoid capturing trailing dot from .ckpt
            match = re.search(r"val_loss[=_](\d+\.?\d*)", p.name)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            return float('inf')
        
        # Sort by val_loss ascending, then by name (for determinism)
        sorted_ckpts = sorted(checkpoints, key=lambda p: (get_val_loss(p), p.name))
        return sorted_ckpts[0]

    def load_models_and_folds(self) -> tuple[dict[str, nn.Module], list]:
        """
        Returns loaded models and fold configuration.
        """
        models_dict = {}
        
        if not self.is_multifold:
            models_dict["single"] = self._load_model("single", self.checkpoint_path)
        else:
            # Load all folds
            k = len(self.folds)
            for fold_idx in range(k):
                fold_dir = self.checkpoint_path / f"fold_{fold_idx}"
                checkpoints = list(fold_dir.glob("*.ckpt"))
                if not checkpoints:
                     checkpoints = list(fold_dir.rglob("*.ckpt"))
                
                if checkpoints:
                    ckpt_path = self._select_best_checkpoint(checkpoints)
                    models_dict[str(fold_idx)] = self._load_model(f"fold_{fold_idx}", ckpt_path)
                else:
                    print(f"Warning: No checkpoint found for fold {fold_idx} in {fold_dir}")
            
        return models_dict, self.folds

    def _load_model(self, key: str, path: Path) -> nn.Module:
        if key in self.cache:
            return self.cache[key]
        
        if path.is_dir():
             checkpoints = list(path.rglob("*.ckpt"))
             if not checkpoints:
                 raise FileNotFoundError(f"No checkpoint found in {path}")
             path = self._select_best_checkpoint(checkpoints)

        print(f"Loading model from {path} for {key}...")
        module = instantiate(
            model_config=self.model_config,
            data_config=self.data_config,
            train_config=self.train_config,
        )
        checkpoint = torch.load(path, map_location=self.device)
        module.load_state_dict(checkpoint["state_dict"])
        module.to(self.device)
        module.eval()
        
        self.cache[key] = module.model
        return module.model
