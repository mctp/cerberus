import dataclasses
import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
)
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_folds
from cerberus.interval import Interval
from cerberus.module import instantiate_model
from cerberus.output import (
    ModelOutput,
    aggregate_intervals,
    aggregate_models,
    unbatch_modeloutput,
)

logger = logging.getLogger(__name__)


def parse_hparams_config(path: str | Path) -> CerberusConfig:
    """Parse a ``hparams.yaml`` file into a validated :class:`CerberusConfig`.

    Args:
        path: Path to the YAML file (typically written by Lightning's
            ``save_hyperparameters``).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValidationError: If the YAML content fails schema validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hparams file not found at: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)
    return CerberusConfig.model_validate(data)


def find_latest_hparams(search_root: Path | str) -> Path:
    """Find the most recently modified ``hparams.yaml`` under ``search_root``."""
    root = Path(search_root)
    candidates = list(root.rglob("hparams.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No hparams.yaml found in {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_val_loss(path: Path) -> float:
    match = re.search(r"val_loss[=_](\d+\.?\d*)", path.name)
    if match is None:
        return float("inf")
    try:
        return float(match.group(1))
    except ValueError:
        return float("inf")


def select_best_checkpoint(checkpoints: list[Path]) -> Path:
    """Select the best checkpoint by lowest val_loss in filename."""
    if not checkpoints:
        raise ValueError("No checkpoints provided.")
    return sorted(checkpoints, key=lambda p: (_parse_val_loss(p), p.name))[0]


def _extract_backbone_state_dict_from_lightning(
    raw_state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Strip Lightning/Cerberus prefixes and keep only backbone weights."""
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not key.startswith("model."):
            continue
        new_key = key[6:]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:]
        state_dict[new_key] = value
    return state_dict


def load_backbone_weights_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | str,
    strict: bool = True,
) -> None:
    """Load backbone weights from either a clean ``model.pt`` or Lightning ``.ckpt``."""
    if isinstance(device, str):
        device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".pt":
        logger.info("Loading clean state dict: %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        return

    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(
            f"Unsupported checkpoint format '{checkpoint_path.suffix}' for {checkpoint_path}"
        )

    logger.info("Loading Lightning checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Checkpoint missing 'state_dict' mapping: {checkpoint_path}"
        )

    state_dict = _extract_backbone_state_dict_from_lightning(raw_state)
    if not state_dict:
        raise ValueError(
            f"No 'model.'-prefixed weights found in checkpoint: {checkpoint_path}"
        )

    model.load_state_dict(state_dict, strict=strict)


def load_backbone_weights_from_fold_dir(
    model: nn.Module,
    fold_dir: Path | str,
    device: torch.device | str,
    strict: bool = True,
) -> Path:
    """Load a model from ``fold_dir`` by preferring ``model.pt`` then best ``.ckpt``."""
    fold_dir = Path(fold_dir)
    pt_path = fold_dir / "model.pt"
    if pt_path.exists():
        load_backbone_weights_from_checkpoint(
            model=model,
            checkpoint_path=pt_path,
            device=device,
            strict=strict,
        )
        return pt_path

    checkpoints = list(fold_dir.glob("*.ckpt"))
    if not checkpoints:
        checkpoints = list(fold_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No model.pt or .ckpt files found in fold directory: {fold_dir}"
        )

    ckpt_path = select_best_checkpoint(checkpoints)
    load_backbone_weights_from_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        device=device,
        strict=strict,
    )
    return ckpt_path


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
        fold: int | None = None,
    ):
        """
        Args:
            checkpoint_path: Training root directory containing
                ``ensemble_metadata.yaml`` and ``fold_N/`` subdirs.
            model_config / data_config / genome_config: Optional overrides
                of the values parsed from the training-root ``hparams.yaml``.
            device: Inference device. Auto-selects CUDA → CPU if ``None``.
            fold: Load only this specific fold's model. When ``None`` (default),
                load every fold in ``ensemble_metadata.yaml``. Useful for
                per-fold tools (attribution, TF-MoDISco export) that would
                otherwise waste memory loading unused folds.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        path = Path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(f"checkpoint_path must be a directory: {path}")

        # Resolve configuration. When a specific fold is requested, prefer the
        # hparams.yaml inside that fold's directory (train_multi writes
        # fold-specific test_fold / val_fold values there); fall back to the
        # latest hparams under the root if the fold-specific one is absent
        # (e.g. single-fold train_single layout with no fold_N subdir).
        hparams_search_root: Path = path
        if fold is not None:
            fold_dir_candidate = path / f"fold_{fold}"
            if fold_dir_candidate.is_dir():
                try:
                    find_latest_hparams(fold_dir_candidate)
                    hparams_search_root = fold_dir_candidate
                except FileNotFoundError:
                    pass
        hparams_path = find_latest_hparams(hparams_search_root)
        self.cerberus_config: CerberusConfig = parse_hparams_config(hparams_path)

        overrides = {}
        if model_config is not None:
            overrides["model_config_"] = model_config
        if data_config is not None:
            overrides["data_config"] = data_config
        if genome_config is not None:
            overrides["genome_config"] = genome_config
        if overrides:
            self.cerberus_config = self.cerberus_config.model_copy(update=overrides)

        model_config = self.cerberus_config.model_config_
        data_config = self.cerberus_config.data_config
        genome_config = self.cerberus_config.genome_config

        loader = _ModelManager(
            path, model_config, data_config, genome_config, device, fold=fold
        )
        models, folds = loader.load_models_and_folds()

        super().__init__(models)
        self.folds = folds
        self.device = device

    def _resolve_use_folds(self, use_folds: list[str] | None) -> list[str]:
        """
        Resolves the default folds to use if not provided.
        Default logic:
        - If single-fold model (1 loaded model): use ["train", "test", "val"] (all)
        - If multi-fold model: use ["test", "val"]
        """
        if use_folds is not None:
            return use_folds

        if len(self) == 1:
            return ["train", "test", "val"]
        else:
            return ["test", "val"]

    def forward(
        self,
        x: torch.Tensor,
        intervals: list[Interval] | None = None,
        use_folds: list[str] | None = None,
        aggregation: str = "model",  # "model", "interval+model"
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
            use_folds (list[str] | None): List of fold roles to include in the ensemble. Allowed values
                are 'train', 'test', 'val'. Defaults to None (resolves based on ensemble size).
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
        use_folds = self._resolve_use_folds(use_folds)

        # 1. Run models -> list[ModelOutput] (one per model, batched)
        batch_outputs, masks = self._forward_models(x, intervals, use_folds)

        if not batch_outputs:
            raise RuntimeError("No model outputs generated.")

        if aggregation == "model":
            # Aggregate over models, return batched [aggregated]
            return aggregate_models(batch_outputs, method="mean", masks=masks)

        if aggregation == "interval+model":
            if intervals is None:
                raise ValueError("Intervals are required for interval aggregation.")

            # Center intervals to output length
            centered_intervals = [
                i.center(self.cerberus_config.data_config.output_len) for i in intervals
            ]

            # Aggregate over models
            aggregated_batch = aggregate_models(
                batch_outputs, method="mean", masks=masks
            )
            output_cls = type(aggregated_batch)

            # Unbatch
            unbatched = unbatch_modeloutput(aggregated_batch, len(centered_intervals))

            # Merge over intervals
            merged = aggregate_intervals(
                unbatched,
                centered_intervals,
                output_len=self.cerberus_config.data_config.output_len,
                output_bin_size=self.cerberus_config.data_config.output_bin_size,
                output_cls=output_cls,
            )
            return merged

        raise ValueError(
            f"Unknown aggregation mode: {aggregation} (supported: 'model', 'interval+model')"
        )

    def _get_partitions_for_interval(self, interval: Interval) -> set[int]:
        """Return the set of fold partition indices that contain *interval*.

        Queries each fold's InterLap tree.  Returns an empty set when no
        partition matches (e.g. chromosome not in any fold map).
        """
        partitions: set[int] = set()
        for fold_idx, fold_map in enumerate(self.folds):
            if interval.chrom in fold_map:
                if any(
                    fold_map[interval.chrom].find(
                        (interval.start, interval.end - 1)
                    )
                ):
                    partitions.add(fold_idx)
        return partitions

    def _partitions_to_model_indices(
        self, partitions: set[int], use_folds: list[str]
    ) -> set[int]:
        """Map partition indices to model indices using the fold rotation.

        The rotation logic assumes:
        - Model *i* treats Partition *i* as TEST.
        - Model *i* treats Partition *(i+1) % k* as VAL.
        """
        model_indices: set[int] = set()
        n_folds = len(self.folds)
        if n_folds == 0:
            return model_indices
        for p_idx in partitions:
            if "test" in use_folds:
                model_indices.add(p_idx)
            if "val" in use_folds:
                model_indices.add((p_idx - 1) % n_folds)
            if "train" in use_folds:
                test_model = p_idx
                val_model = (p_idx - 1) % n_folds
                for i in range(n_folds):
                    if i != test_model and i != val_model:
                        model_indices.add(i)
        return model_indices

    def _forward_models(
        self,
        x: torch.Tensor,
        intervals: list[Interval] | None = None,
        use_folds: list[str] | None = None,
    ) -> tuple[list[ModelOutput], list[torch.Tensor] | None]:
        """Run the forward pass, routing each sample to the correct fold models.

        When *intervals* are provided and the ensemble has multiple folds,
        each sample is routed to its fold-appropriate model(s) based on its
        interval.  Models that do not see a given sample produce zeros for
        that position, and a boolean mask is returned so the caller can
        average only over models that contributed to each sample.

        Args:
            x: Input tensor ``(Batch, Channels, Length)``.
            intervals: Optional per-sample intervals for fold routing.
            use_folds: Fold roles to include (``'train'``, ``'test'``,
                ``'val'``).

        Returns:
            ``(outputs, masks)`` where *outputs* is one ``ModelOutput`` per
            model (full-batch shaped) and *masks* is a parallel list of
            ``(B,)`` bool tensors indicating which samples each model
            contributed to.  *masks* is ``None`` when every model sees
            every sample (no intervals, single model, or no folds).
        """
        use_folds = self._resolve_use_folds(use_folds)
        batch_size = x.shape[0]

        if not intervals or not self.folds:
            # No routing needed — run all models on full batch
            batch_outputs = []
            with torch.no_grad():
                for model in self.values():
                    batch_outputs.append(model(x))
            return batch_outputs, None

        # -- Per-sample routing --
        # Map each model index to the list of sample indices it should process
        model_to_samples: dict[int, list[int]] = defaultdict(list)

        for i, interval in enumerate(intervals):
            partitions = self._get_partitions_for_interval(interval)
            model_indices = self._partitions_to_model_indices(partitions, use_folds)
            if not model_indices:
                # No matching fold — fall back to all models for this sample.
                # Common cause: interval is on a chromosome absent from the
                # fold maps.  Could also indicate a fold configuration gap.
                logger.debug(
                    "No fold partition matched interval %s; "
                    "falling back to all models for this sample.",
                    interval,
                )
                for key in self:
                    model_indices.add(int(key))
            for m_idx in model_indices:
                model_to_samples[m_idx].append(i)

        # -- Run each model on its sub-batch and scatter back --
        batch_outputs: list[ModelOutput] = []
        masks: list[torch.Tensor] = []

        with torch.no_grad():
            for m_idx in sorted(model_to_samples):
                key = str(m_idx)
                if key not in self:
                    continue

                sample_indices = model_to_samples[m_idx]
                idx_tensor = torch.tensor(sample_indices, device=x.device, dtype=torch.long)

                sub_x = x[idx_tensor]
                sub_out = self[key](sub_x)

                # Scatter sub_out into full-batch-shaped output.
                # Non-participating positions are filled with zeros.  The fill
                # value is irrelevant: aggregate_models() multiplies by the
                # boolean mask (0.0 for non-contributing positions) before
                # summing, so any fill value × 0.0 = 0.0 — it never reaches
                # the output.  Zero is used because torch.zeros is cheap and
                # the masks (returned alongside) are the authoritative record
                # of which samples each model contributed to.
                full_out_fields: dict[str, Any] = {}
                for field in dataclasses.fields(sub_out):
                    val = getattr(sub_out, field.name)
                    if isinstance(val, torch.Tensor):
                        full = torch.zeros(
                            (batch_size, *val.shape[1:]),
                            dtype=val.dtype,
                            device=val.device,
                        )
                        full[idx_tensor] = val
                        full_out_fields[field.name] = full
                    else:
                        full_out_fields[field.name] = val

                output_cls = type(sub_out)
                batch_outputs.append(output_cls(**full_out_fields))

                mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
                mask[idx_tensor] = True
                masks.append(mask)

        if not batch_outputs:
            # No models matched any sample — fall back to all models, no routing
            with torch.no_grad():
                for model in self.values():
                    batch_outputs.append(model(x))
            return batch_outputs, None

        return batch_outputs, masks

    def predict_intervals_batched(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        use_folds: list[str] | None = None,
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
            use_folds: List of folds to use. Defaults to None (resolves based on ensemble size).
            aggregation: Aggregation mode ("model" or "interval+model").
            batch_size: Batch size.

        Yields:
            tuple[ModelOutput, list[Interval]]: A tuple containing the batch output and the list of
            intervals in that batch.

            If aggregation="model", ModelOutput is batched.
            If aggregation="interval+model", ModelOutput is merged/unbatched for that batch's spatial extent.
        """
        use_folds = self._resolve_use_folds(use_folds)
        input_len = dataset.data_config.input_len

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
        use_folds: list[str] | None = None,
        aggregation: str = "model",
        batch_size: int = 64,
    ) -> ModelOutput:
        """
        Runs prediction for a sequence of intervals, batching them efficiently.

        This method iterates over the provided intervals, fetches input data from the dataset,
        runs the model ensemble, and aggregates the results into a single unified output.

        Input Constraints:
        - All intervals must have a length exactly equal to ``dataset.data_config.input_len``.
        - The `dataset` must be configured to provide "inputs".

        Args:
            intervals (Iterable[Interval]): An iterable of genomic intervals to predict on.
                Each interval must match the model's input length.
            dataset (CerberusDataset): The dataset used to retrieve input sequences/signals
                for the intervals.
            use_folds (list[str] | None): Folds to use (e.g., ["test"]). Defaults to None (resolves based on ensemble size).
            aggregation (str): Aggregation mode ("model" or "interval+model"). Defaults to "model".
            batch_size (int): Number of intervals to process in a single batch. Defaults to 64.

        Returns:
            ModelOutput: A single aggregated ModelOutput object containing the merged predictions
            for all provided intervals.

        Raises:
            RuntimeError: If no results are generated (e.g., input `intervals` was empty).
        """
        use_folds = self._resolve_use_folds(use_folds)
        output_len = dataset.data_config.output_len

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
                for interval, output in zip(batch_intervals, unbatched, strict=True):
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
            output_bin_size=self.cerberus_config.data_config.output_bin_size,
            output_cls=output_cls,
        )
        return merged

    def predict_output_intervals(
        self,
        intervals: Iterable[Interval],
        dataset: CerberusDataset,
        stride: int | None = None,
        use_folds: list[str] | None = None,
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
            use_folds (list[str] | None): Folds to use. Defaults to None (resolves based on ensemble size).
            aggregation (str): Aggregation mode. Defaults to "model".
            batch_size (int): Batch size for processing tiles. Defaults to 64.

        Returns:
            list[ModelOutput]: A list of ModelOutput objects, one for each interval in `intervals`.
        """
        use_folds = self._resolve_use_folds(use_folds)
        input_len = dataset.data_config.input_len
        output_len = dataset.data_config.output_len

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

    Prefers clean ``model.pt`` state dicts (written by training since the
    pretrained-weight-loading update) and falls back to Lightning ``.ckpt``
    checkpoints for backward compatibility with older training runs.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config: ModelConfig,
        data_config: DataConfig,
        genome_config: GenomeConfig,
        device: torch.device,
        fold: int | None = None,
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
            all_folds = meta.get("folds", [])

        if fold is not None:
            if fold not in all_folds:
                raise ValueError(
                    f"fold={fold} is not present in ensemble_metadata.yaml "
                    f"(available folds: {all_folds})"
                )
            self.fold_indices = [fold]
        else:
            self.fold_indices = all_folds

        # Create fold mappings for routing
        self.folds = create_genome_folds(
            genome_config.chrom_sizes,
            genome_config.fold_type,
            genome_config.fold_args,
        )

    def load_models_and_folds(self) -> tuple[dict[str, nn.Module], list]:
        """
        Loads models and fold configuration.

        Prefers ``model.pt`` (clean state dict) when available, falling back
        to the best Lightning ``.ckpt`` checkpoint otherwise.
        """
        models_dict = {}

        for fold_idx in self.fold_indices:
            fold_dir = self.checkpoint_path / f"fold_{fold_idx}"
            key = f"fold_{fold_idx}"
            if not fold_dir.is_dir():
                logger.warning("Fold directory missing for fold %s: %s", fold_idx, fold_dir)
                continue
            try:
                models_dict[str(fold_idx)] = self._load_model_from_fold(key, fold_dir)
            except FileNotFoundError:
                logger.warning("No checkpoint found for fold %s in %s", fold_idx, fold_dir)

        return models_dict, self.folds

    def _load_model_from_fold(self, key: str, fold_dir: Path) -> nn.Module:
        """
        Load one fold model, caching by key for repeated use.
        """
        if key in self.cache:
            return self.cache[key]

        model = instantiate_model(
            model_config=self.model_config,
            data_config=self.data_config,
        )
        checkpoint_path = load_backbone_weights_from_fold_dir(
            model=model,
            fold_dir=fold_dir,
            device=self.device,
            strict=True,
        )
        logger.info("Loaded model for %s from %s", key, checkpoint_path)
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

    existing_folds: set[int] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            try:
                meta = yaml.safe_load(f)
                if meta and "folds" in meta:
                    existing_folds = set(meta["folds"])
            except yaml.YAMLError:
                logger.warning(
                    f"Corrupt ensemble metadata at {meta_path}; "
                    "existing fold information will be lost"
                )

    existing_folds.add(fold)

    with open(meta_path, "w") as f:
        yaml.dump({"folds": sorted(list(existing_folds))}, f)
