from pathlib import Path
import torch
from torch import nn
import re
import dataclasses
from typing import Iterable, Iterator
import itertools
import yaml

from cerberus.config import (
    CerberusConfig,
    GenomeConfig,
    DataConfig,
    ModelConfig,
    parse_hparams_config,
)
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_folds
from cerberus.module import instantiate_model
from cerberus.output import (
    ModelOutput, 
    unbatch_modeloutput, 
    aggregate_intervals, 
    aggregate_models
)
from cerberus.interval import Interval


class ModelEnsemble(nn.ModuleDict):
    """
    Wraps a dictionary of models (fold_idx -> model) and manages
    selection based on genomic intervals.
    """
    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config: ModelConfig | None = None,
        data_config: DataConfig | None = None,
        genome_config: GenomeConfig | None = None,
        device: torch.device | str | None = None,
        search_paths: list[Path] | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        path = Path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(f"checkpoint_path must be a directory: {path}")

        # Resolve configuration
        hparams_path = self._find_hparams(path)
        self.cerberus_config : CerberusConfig = parse_hparams_config(
            hparams_path, search_paths=search_paths
        )

        if model_config is not None:
            self.cerberus_config["model_config"] = model_config
        if data_config is not None:
            self.cerberus_config["data_config"] = data_config
        if genome_config is not None:
            self.cerberus_config["genome_config"] = genome_config

        model_config = self.cerberus_config["model_config"]
        data_config = self.cerberus_config["data_config"]
        genome_config = self.cerberus_config["genome_config"]

        loader = _ModelManager(
            path, model_config, data_config, genome_config, device
        )
        models, folds = loader.load_models_and_folds()

        super().__init__(models)
        self.folds = folds
        self.device = device

    def _find_hparams(self, checkpoint_dir: Path) -> Path:
        """
        Recursively searches for hparams.yaml in the checkpoint directory.
        Returns the most recently modified one.
        """
        candidates = list(checkpoint_dir.rglob("hparams.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No hparams.yaml found in {checkpoint_dir}")
        
        # Sort by modification time (newest first)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

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
            return aggregate_models(batch_outputs, method="mean")
        
        if aggregation == "interval+model":
            if intervals is None:
                raise ValueError("Intervals are required for interval aggregation.")

            # Center intervals to output length
            centered_intervals = [
                i.center(self.cerberus_config["data_config"]["output_len"])
                for i in intervals
            ]

            # Aggregate over models
            aggregated_batch = aggregate_models(batch_outputs, method="mean")
            output_cls = type(aggregated_batch)
            
            # Unbatch
            unbatched = unbatch_modeloutput(aggregated_batch, len(centered_intervals))

            # Merge over intervals
            merged = aggregate_intervals(
                unbatched,
                centered_intervals,
                output_len=self.cerberus_config["data_config"]["output_len"],
                output_bin_size=self.cerberus_config["data_config"]["output_bin_size"],
                output_cls=output_cls,
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
            # The rotation logic assumes:
            # - Model 'i' treats Partition 'i' as TEST.
            # - Model 'i' treats Partition '(i+1)%k' as VAL.
            # Therefore:
            # - To get TEST predictions for Partition 'p', we need Model 'p'.
            # - To get VAL predictions for Partition 'p', we need Model 'p-1' (mod k).
            
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

    def predict_intervals_batched(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        use_folds: list[str] = ["test", "val"],
        aggregation: str = "model",
        batch_size: int = 64,
    ) -> Iterator[tuple[ModelOutput, list[Interval]]]:
        """
        Runs prediction for a sequence of intervals in batches, yielding results as they are computed.
        
        This method is a generator that duplicates the batch processing logic of `predict_intervals`
        but returns each batch immediately instead of aggregating everything at the end.
        
        Args:
            intervals: Iterable of genomic intervals.
            dataset: Dataset for input retrieval.
            use_folds: List of folds to use.
            aggregation: Aggregation mode ("model" or "interval+model").
            batch_size: Batch size.
            
        Yields:
            tuple[ModelOutput, list[Interval]]: A tuple containing the batch output and the list of
            intervals in that batch.
            
            If aggregation="model", ModelOutput is batched.
            If aggregation="interval+model", ModelOutput is merged/unbatched for that batch's spatial extent.
        """
        input_len = dataset.data_config["input_len"]
        
        if aggregation not in ["model", "interval+model"]:
            raise ValueError(
                f"aggregation must be 'model' or 'interval+model', got '{aggregation}'"
            )

        for batch_intervals_tuple in itertools.batched(intervals, batch_size):
            batch_intervals = list(batch_intervals_tuple)

            # 1. Validate Interval Lengths
            for interval in batch_intervals:
                if len(interval) != input_len:
                    raise ValueError(
                        f"Interval length {len(interval)} does not match model input length {input_len}. "
                        f"Interval: {interval}"
                    )

            # 2. Prepare Batch Data
            inputs_list = []
            for interval in batch_intervals:
                data = dataset.get_interval(interval)
                inputs_list.append(data["inputs"])

            # Stack into (Batch, Channels, Length)
            batch_inputs = torch.stack(inputs_list).to(self.device)

            # 3. Run Models
            batched_output = self.forward(
                batch_inputs,
                intervals=batch_intervals,
                use_folds=use_folds,
                aggregation=aggregation,
            )
            
            yield batched_output, batch_intervals

    def predict_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        use_folds: list[str] = ["test", "val"],
        aggregation: str = "model",
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
            use_folds (list[str]): Folds to use (e.g., ["test"]). Defaults to ["test", "val"].
            aggregation (str): Aggregation mode ("model" or "interval+model"). Defaults to "model".
            batch_size (int): Number of intervals to process in a single batch. Defaults to 64.

        Returns:
            ModelOutput: A single aggregated ModelOutput object containing the merged predictions
            for all provided intervals.

        Raises:
            RuntimeError: If no results are generated (e.g., input `intervals` was empty).
        """
        output_len = dataset.data_config["output_len"]

        results = []
        output_cls = None

        for batched_output, batch_intervals in self.predict_intervals_batched(
            intervals, dataset, use_folds, aggregation, batch_size
        ):
            output_cls = type(batched_output)

            if aggregation == "interval+model":
                # Result is already merged for the batch
                out_dict = dataclasses.asdict(batched_output)
                out_int = batched_output.out_interval
                results.append((out_dict, out_int))
            else:
                # Result is batched
                unbatched = unbatch_modeloutput(batched_output, len(batch_intervals))
                for interval, output in zip(batch_intervals, unbatched):
                    output_interval = interval.center(output_len)
                    results.append((output, output_interval))

        # 4. Final Aggregation
        if not results:
             raise RuntimeError("No results generated.")
             
        outputs_list = [r[0] for r in results]
        intervals_list = [r[1] for r in results]
        
        merged = aggregate_intervals(
            outputs_list,
            intervals_list,
            output_len=output_len,
            output_bin_size=self.cerberus_config["data_config"]["output_bin_size"],
            output_cls=output_cls,
        )
        return merged

    def predict_output_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        stride: int | None = None,
        use_folds: list[str] = ["test", "val"],
        aggregation: str = "model",
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
            stride (int | None): The stride/step size for tiling input intervals.
                If None, defaults to output_len // 2.
            use_folds (list[str]): Folds to use. Defaults to ["test", "val"].
            aggregation (str): Aggregation mode. Defaults to "model".
            batch_size (int): Batch size for processing tiles. Defaults to 64.

        Returns:
            list[ModelOutput]: A list of ModelOutput objects, one for each interval in `intervals`.
        """
        input_len = dataset.data_config["input_len"]
        output_len = dataset.data_config["output_len"]

        if stride is None:
            stride = output_len // 2

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
                    use_folds=use_folds,
                    aggregation=aggregation,
                    batch_size=batch_size,
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
        
        # Parse metadata
        meta_path = self.checkpoint_path / "ensemble_metadata.yaml"
            
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
            self.fold_indices = meta.get("folds", [])

        # Create fold mappings for routing
        self.folds = create_genome_folds(
            genome_config["chrom_sizes"],
            genome_config["fold_type"],
            genome_config["fold_args"],
        )

    def _select_best_checkpoint(self, checkpoints: list[Path]) -> Path:
        """
        Selects the best checkpoint from a list based on validation loss.
        """
        def get_val_loss(p: Path) -> float:
            # Pattern to match val_loss=0.1234
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
        """
        models_dict = {}
        
        for fold_idx in self.fold_indices:
            fold_dir = self.checkpoint_path / f"fold_{fold_idx}"
            
            # Find checkpoint
            checkpoints = list(fold_dir.glob("*.ckpt"))
            if not checkpoints:
                 checkpoints = list(fold_dir.rglob("*.ckpt"))
            
            if checkpoints:
                ckpt_path = self._select_best_checkpoint(checkpoints)
                models_dict[str(fold_idx)] = self._load_model(f"fold_{fold_idx}", ckpt_path)
            else:
                print(f"Warning: No checkpoint found for fold {fold_idx} in {fold_dir}")
            
        return models_dict, self.folds

    def _load_model(self, key: str, ckpt_file: Path) -> nn.Module:
        """
        Loads a single model from a checkpoint file.
        """
        if key in self.cache:
            return self.cache[key]
        
        print(f"Loading model from {ckpt_file} for {key}...")
        # Instantiate only the backbone model to avoid overhead of CerberusModule (metrics, loss, etc.)
        model = instantiate_model(
            model_config=self.model_config,
            data_config=self.data_config,
        )
        checkpoint = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        
        # Strip "model." prefix from state_dict keys because we are loading into the backbone model directly,
        # not the CerberusModule wrapper where these weights were originally saved under 'self.model'.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                key = k[6:]
                # Handle torch.compile prefix:
                # If the model was compiled during training, keys will have "_orig_mod." prefix.
                # Since we are loading into an uncompiled model here, we must strip this prefix.
                if key.startswith("_orig_mod."):
                    key = key[10:]
                new_state_dict[key] = v
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        
        self.cache[key] = model
        return model


def update_ensemble_metadata(root_dir: Path | str, fold: int):
    """
    Updates or creates ensemble_metadata.yaml in the root directory
    to include the specified fold.
    """
    path = Path(root_dir)
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "ensemble_metadata.yaml"
    
    existing_folds = set()
    if meta_path.exists():
        with open(meta_path, "r") as f:
            try:
                meta = yaml.safe_load(f)
                if meta and "folds" in meta:
                     existing_folds = set(meta["folds"])
            except yaml.YAMLError:
                pass
    
    existing_folds.add(fold)
    
    with open(meta_path, "w") as f:
        yaml.dump({"folds": sorted(list(existing_folds))}, f)
