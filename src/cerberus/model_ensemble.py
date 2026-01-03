from pathlib import Path
import torch
from torch import nn
import numpy as np
import re
import dataclasses
from typing import Any, Iterable, Sequence
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
        genome_config: GenomeConfig,
        device: torch.device,
    ):
        loader = _ModelManager(
            checkpoint_path, model_config, data_config, genome_config, device
        )
        models, folds = loader.load_models_and_folds()

        super().__init__(models)
        self.folds = folds
        self.device = device
        self.output_len = data_config["output_len"]
        self.output_bin_size = data_config["output_bin_size"]

    @staticmethod
    def _unbatch_modeloutput(batched_output: ModelOutput, batch_size: int) -> list[dict[str, Any]]:
        """
        Splits a batched output (ModelOutput) into a list of individual interval dictionaries.
        
        Args:
            batched_output: The batched ModelOutput object to split.
            batch_size: The number of items in the batch.
            
        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing an unbatched output.
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

    @staticmethod
    def _aggregate_tensor_track_values(
        outputs: list[torch.Tensor],
        intervals: list[Interval],
        merged_interval: Interval,
        output_len: int,
        output_bin_size: int,
    ) -> np.ndarray:
        """
        Aggregates a list of tensors into a single merged array.
        
        Args:
            outputs: List of tensors to aggregate.
            intervals: List of intervals corresponding to the tensors.
            merged_interval: The overall interval covering all inputs.
            output_len: The expected length of profile outputs.
            output_bin_size: The bin size of the outputs.
            
        Returns:
            np.ndarray: The aggregated values as a numpy array.
        """
        # Merged extent
        min_start = merged_interval.start
        max_end = merged_interval.end
        
        span_bp = max_end - min_start
        n_bins = span_bp // output_bin_size
        
        # outputs[0] is (C, L) or (C,)
        sample_out = outputs[0]
        n_channels = sample_out.shape[0]
        
        # Accumulators
        accumulator = np.zeros((n_channels, n_bins), dtype=np.float32)
        counts = np.zeros((1, n_bins), dtype=np.float32)
        
        for out_tensor, interval in zip(outputs, intervals):
            # Convert to numpy
            val = out_tensor.cpu().numpy() # (C, L) or (C,)
            
            # Relative start bin
            rel_start_bp = interval.start - min_start
            rel_start_bin = rel_start_bp // output_bin_size

            # Expected bins for this interval
            interval_len_bp = interval.end - interval.start
            interval_bins = interval_len_bp // output_bin_size
            
            # Check dimensions
            # It is a profile/track if it has spatial dimension and matches the interval length
            is_profile = (val.ndim >= 2 and val.shape[-1] == interval_bins)
            
            if is_profile:
                # val is (C, L_bins)
                end_bin = rel_start_bin + interval_bins
                
                # Bounds check (simple clipping or assume fits)
                # Since intervals are within merged_interval, it should fit unless precision issues
                
                accumulator[:, rel_start_bin : end_bin] += val
                counts[:, rel_start_bin : end_bin] += 1.0
            else:
                # Scalar (C,)
                # Broadcast to interval length to create a constant track over the interval
                # Interval length in bins
                end_bin = rel_start_bin + interval_bins
                
                # val is (C,) -> (C, int_bins)
                if val.ndim == 1:
                    val_b = np.expand_dims(val, -1)
                else:
                    val_b = val

                accumulator[:, rel_start_bin : end_bin] += val_b
                counts[:, rel_start_bin : end_bin] += 1.0

        # Average
        counts = np.maximum(counts, 1.0)
        final_values = accumulator / counts
        
        return final_values

    @staticmethod
    def _aggregate_intervals(
        outputs: list[dict[str, Any]],
        intervals: list[Interval],
        output_len: int,
        output_bin_size: int,
        output_cls: type[ModelOutput] | None = None,
    ) -> ModelOutput:
        """
        Aggregates overlapping predictions into a single merged output.
        
        Args:
            outputs: List of unbatched output dictionaries.
            intervals: List of intervals corresponding to the outputs.
            output_len: The length of the output profile in bins/bp (used for profile detection).
            output_bin_size: The size of each bin in base pairs.
            output_cls: The ModelOutput class to use for the result.
            
        Returns:
            ModelOutput: The merged output object covering the union of intervals.
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
        
        aggregated_components = {}
        
        for key in keys:
            component_outputs = [out[key] for out in outputs] # list[Tensor]
            agg_comp = ModelEnsemble._aggregate_tensor_track_values(
                component_outputs, intervals, merged_interval, output_len, output_bin_size
            )
            # Convert numpy to tensor (CPU)
            aggregated_components[key] = torch.from_numpy(agg_comp)

        if output_cls is None:
            raise ValueError("output_cls must be provided to aggregate_intervals")

        return output_cls(**aggregated_components, out_interval=merged_interval)

    @staticmethod
    def _aggregate_models(
        outputs: Sequence[ModelOutput], method: str
    ) -> ModelOutput:
        """
        Aggregates a list of model outputs.
        
        Args:
            outputs: List of outputs to aggregate.
            method: Aggregation method ("mean" or "median").
            
        Returns:
            ModelOutput: The aggregated output.
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

    def forward(
        self, 
        x: torch.Tensor, 
        intervals: list[Interval] | None = None, 
        use_folds: list[str] = ["test", "val"],
        aggregation: str = "model" # "model", "interval+model"
    ) -> ModelOutput:
        """
        Runs the forward pass on selected models and aggregates their outputs.

        This method identifies which models to run based on the provided `intervals` (using the
        fold configuration) and the `use_folds` filter. It then aggregates the results according
        to the specified `aggregation` strategy.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Length). The length must
                match the model's expected input length.
            intervals (list[Interval] | None): Optional list of genomic intervals corresponding to
                the inputs in `x`. Used to determine which fold-specific models to execute.
                If provided, must have the same length as the batch dimension of `x`.
                If None, all models matching `use_folds` are executed.
            use_folds (list[str]): List of fold roles to include in the ensemble. Allowed values
                are 'train', 'test', 'val'. Defaults to ["test", "val"].
            aggregation (str): Strategy for aggregating model outputs.
                - "model": Aggregates outputs across models for the same input (returns batched output).
                - "interval+model": Aggregates across models AND merges overlapping intervals (returns single unbatched output).
                Defaults to "model".

        Returns:
            ModelOutput: The aggregated model output. The exact type depends on the models used.
        
        Raises:
            RuntimeError: If no models are selected for execution.
            ValueError: If `aggregation` is invalid or if `intervals` are missing when required.
        """
        # 1. Run models -> list[ModelOutput] (one per model, batched)
        batch_outputs = self._forward_models(x, intervals, use_folds)
        
        if not batch_outputs:
            raise RuntimeError("No model outputs generated.")

        if aggregation == "model":
            # Aggregate over models, return batched [aggregated]
            return ModelEnsemble._aggregate_models(batch_outputs, method="mean")
        
        if aggregation == "interval+model":
            if intervals is None:
                raise ValueError("Intervals are required for interval aggregation.")

            # Center intervals to output length
            centered_intervals = [i.center(self.output_len) for i in intervals]

            # Aggregate over models
            aggregated_batch = ModelEnsemble._aggregate_models(batch_outputs, method="mean")
            output_cls = type(aggregated_batch)
            
            # Unbatch
            unbatched = ModelEnsemble._unbatch_modeloutput(aggregated_batch, len(centered_intervals))
            
            # Merge over intervals
            merged = ModelEnsemble._aggregate_intervals(
                unbatched, 
                centered_intervals, 
                output_len=self.output_len,
                output_bin_size=self.output_bin_size,
                output_cls=output_cls
            )
            return merged

        raise ValueError(f"Unknown aggregation mode: {aggregation} (supported: 'model', 'interval+model')")

    def _forward_models(
        self, 
        x: torch.Tensor, 
        intervals: list[Interval] | None = None, 
        use_folds: list[str] = ["test", "val"]
    ) -> list[ModelOutput]:
        """
        Runs the forward pass for selected models.
        
        Args:
            x: Input tensor (Batch, Channels, Length).
            intervals: Optional list of intervals to determine which models to run.
            use_folds: List of fold roles to use ('train', 'test', 'val').
            
        Returns:
            list[ModelOutput]: A list of outputs from the selected models.
        """
        if not intervals:
            # Fallback: run all models if no intervals provided (or for single model)
            models_to_run = self.values()
        else:
            # Determine applicable models
            target_partitions = set()
            
            # We assume all intervals in the batch belong to the same partition
            interval = intervals[0]
            for fold_idx, fold_map in enumerate(self.folds):
                if interval.chrom in fold_map:
                    if any(fold_map[interval.chrom].find((interval.start, interval.end - 1))):
                        target_partitions.add(fold_idx)

            # Determine which models to load based on use_folds            
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

        # Run models
        batch_outputs = []
        with torch.no_grad():
            for model in models_to_run:
                out = model(x)
                batch_outputs.append(out)
        return batch_outputs

    def predict_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        predict_config: PredictConfig,
        batch_size: int = 64,
    ) -> ModelOutput:
        """
        Runs prediction for a sequence of intervals, batching them efficiently.

        This method iterates over the provided intervals, fetches input data from the dataset,
        runs the model ensemble, and aggregates the results into a single unified output.

        Input Constraints:
        - All intervals must have a length exactly equal to `dataset.data_config["input_len"]`.
        - The `dataset` must be configured to provide "inputs".

        Args:
            intervals (Iterable[Interval]): An iterable of genomic intervals to predict on.
                Each interval must match the model's input length.
            dataset (CerberusDataset): The dataset used to retrieve input sequences/signals
                for the intervals.
            predict_config (PredictConfig): Configuration dictionary containing:
                - "use_folds" (list[str]): Folds to use (e.g., ["test"]).
                - "aggregation" (str, optional): Aggregation mode ("model" or "interval+model").
                  Defaults to "model".
            batch_size (int): Number of intervals to process in a single batch. Defaults to 64.

        Returns:
            ModelOutput: A single aggregated ModelOutput object containing the merged predictions
            for all provided intervals.

        Raises:
            RuntimeError: If no results are generated (e.g., input `intervals` was empty).
        """
        input_len = dataset.data_config["input_len"]
        output_len = dataset.data_config["output_len"]
        
        # Default to model aggregation, or interval+model if specified
        aggregation = predict_config.get("aggregation", "model")
        if aggregation not in ["model", "interval+model"]:
            aggregation = "model" # Enforce model aggregation if invalid/legacy value
        
        results = []
        output_cls = None
        
        for batch_intervals_tuple in itertools.batched(intervals, batch_size):
            batch_intervals = list(batch_intervals_tuple)
                
            # 1. Prepare Batch Data
            inputs_list = []
            for interval in batch_intervals:
                data = dataset.get_interval(interval)
                inputs_list.append(data["inputs"])

            # Stack into (Batch, Channels, Length)
            batch_inputs = torch.stack(inputs_list).to(self.device)

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
                unbatched = ModelEnsemble._unbatch_modeloutput(batched_output, len(batch_intervals))
                for interval, output in zip(batch_intervals, unbatched):
                    output_interval = interval.center(output_len)
                    results.append((output, output_interval))

        # 4. Final Aggregation
        if not results:
             raise RuntimeError("No results generated.")
             
        outputs_list = [r[0] for r in results]
        intervals_list = [r[1] for r in results]
        
        merged = ModelEnsemble._aggregate_intervals(
            outputs_list, 
            intervals_list, 
            output_len=output_len,
            output_bin_size=self.output_bin_size,
            output_cls=output_cls
        )
        return merged

    def predict_output_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        predict_config: PredictConfig,
        batch_size: int = 64,
    ) -> list[ModelOutput]:
        """
        Predicts outputs for arbitrary target intervals by tiling them with fixed-length inputs.

        For each target interval in `intervals`, this method generates a series of overlapping
        input intervals (of length `input_len`) that cover the target, respecting the
        configured `stride`. It then runs predictions and merges the results.

        Args:
            intervals (Iterable[Interval]): An iterable of target genomic intervals. These can be
                of arbitrary length (typically larger than the model output length).
            dataset (CerberusDataset): The dataset used to retrieve data. Must have `input_len`
                and `output_len` in `data_config`.
            predict_config (PredictConfig): Configuration dictionary. Must contain:
                - "stride" (int): The stride/step size for tiling input intervals.
                - "use_folds" (list[str]): Folds to use.
            batch_size (int): Batch size for processing tiles. Defaults to 64.

        Returns:
            list[ModelOutput]: A list of ModelOutput objects, one for each interval in `intervals`.
        """
        input_len = dataset.data_config["input_len"]
        output_len = dataset.data_config["output_len"]
        stride = predict_config["stride"]

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
        genome_config: GenomeConfig,
        device: torch.device,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config = model_config
        self.data_config = data_config
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
        """
        Detects if the checkpoint path contains multiple folds.
        
        Returns:
            bool: True if multiple folds are detected, False otherwise.
        """
        if self.checkpoint_path.is_dir():
            # Check for fold_0, fold_1, etc.
            return any((self.checkpoint_path / f"fold_{i}").exists() for i in range(2))
        return False

    def _select_best_checkpoint(self, checkpoints: list[Path]) -> Path:
        """
        Selects the best checkpoint from a list based on validation loss.
        
        Args:
            checkpoints: List of checkpoint paths.
            
        Returns:
            Path: The path to the best checkpoint.
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
        Loads models and fold configuration.
        
        Returns:
            tuple[dict[str, nn.Module], list]: A dictionary of loaded models and a list of fold maps.
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
        """
        Loads a single model from a checkpoint path.
        
        Args:
            key: Key to cache the model under.
            path: Path to the checkpoint or directory containing checkpoints.
            
        Returns:
            nn.Module: The loaded model.
        """
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
            train_config=None,
        )
        checkpoint = torch.load(path, map_location=self.device)
        module.load_state_dict(checkpoint["state_dict"])
        module.to(self.device)
        module.eval()
        
        self.cache[key] = module.model
        return module.model
