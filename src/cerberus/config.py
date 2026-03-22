from __future__ import annotations

from typing import Any, Union, Annotated, Literal
from pathlib import Path
import yaml
import logging
import importlib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo

logger = logging.getLogger(__name__)


def import_class(name: str) -> Any:
    """
    Dynamically imports a class from a module string (e.g., 'package.module.Class').
    """
    if not isinstance(name, str):
        raise TypeError(f"Class name must be a string, got {type(name)}")

    try:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{name}': {e}")


# --- Path Resolution Utilities ---


def _resolve_path(path: Path, search_paths: list[Path] | None = None) -> Path:
    """
    Attempts to resolve a path that might be relative to a different root.

    If the path exists, it is returned as is.
    If not, and search_paths are provided, it checks if the path (or its suffixes)
    exist relative to any of the search paths.
    """
    if path.exists():
        return path

    if search_paths:
        for base in search_paths:
            # 1. Check if path is relative to base
            candidate = base / path
            if candidate.exists():
                return candidate.resolve()

            # 2. If path is absolute, try to match suffixes
            if path.is_absolute():
                parts = path.parts
                # Try progressively shorter suffixes of the original path
                # e.g. /a/b/c/d/file.txt -> d/file.txt, c/d/file.txt, etc.
                for i in range(len(parts) - 1, 0, -1):
                    suffix = Path(*parts[i:])
                    candidate = base / suffix
                    if candidate.exists():
                        return candidate.resolve()
    return path


def _validate_path(
    path: str | Path,
    description: str,
    check_exists: bool = True,
    search_paths: list[Path] | None = None,
) -> Path:
    """Validates that a path exists (optional) and returns it as a Path object."""
    p = Path(path)

    if check_exists:
        if not p.exists():
            # Attempt resolution
            resolved = _resolve_path(p, search_paths)
            if resolved.exists():
                return resolved
            # If still not found
            raise FileNotFoundError(
                f"{description} not found at: {p} (and could not be resolved in search paths)"
            )

    return p


def _validate_file_dict(
    data: dict[str, Any],
    description: str,
    search_paths: list[Path] | None = None,
) -> dict[str, Path]:
    """Validates a dictionary of name -> filepath mappings."""
    if not isinstance(data, dict):
        raise TypeError(f"{description} must be a dictionary")

    validated = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"{description} keys must be strings")
        validated[k] = _validate_path(v, f"{description} file '{k}'", search_paths=search_paths)
    return validated


# --- Configuration Schemas ---


class FoldArgs(BaseModel):
    """Arguments for the chromosome-partition fold strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=0)
    test_fold: int | None = Field(default=None, ge=0)
    val_fold: int | None = Field(default=None, ge=0)


# --- Sampler Args Sub-Models ---


class IntervalSamplerArgs(BaseModel):
    """Arguments for the interval sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervals_path: Path

    @field_validator("intervals_path", mode="before")
    @classmethod
    def resolve_intervals_path(cls, v: Any, info: ValidationInfo) -> Path:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "intervals file", search_paths=search_paths)


class SlidingWindowSamplerArgs(BaseModel):
    """Arguments for the sliding window sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stride: int = Field(gt=0)


class RandomSamplerArgs(BaseModel):
    """Arguments for the random sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_intervals: int = Field(gt=0)


class PeakSamplerArgs(BaseModel):
    """Arguments for the peak sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None

    @field_validator("intervals_path", mode="before")
    @classmethod
    def resolve_intervals_path(cls, v: Any, info: ValidationInfo) -> Path:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "intervals file", search_paths=search_paths)


class NegativePeakSamplerArgs(BaseModel):
    """Arguments for the negative peak sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervals_path: Path
    background_ratio: float = Field(default=1.0, gt=0)
    complexity_center_size: int | None = None

    @field_validator("intervals_path", mode="before")
    @classmethod
    def resolve_intervals_path(cls, v: Any, info: ValidationInfo) -> Path:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "intervals file", search_paths=search_paths)


class ComplexityMatchedSamplerArgs(BaseModel):
    """Arguments for the complexity-matched sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_sampler: "SamplerConfig"
    candidate_sampler: "SamplerConfig"
    bins: int = Field(gt=0)
    candidate_ratio: float = Field(gt=0)
    metrics: list[str]


SamplerArgsUnion = Union[
    IntervalSamplerArgs,
    SlidingWindowSamplerArgs,
    RandomSamplerArgs,
    PeakSamplerArgs,
    NegativePeakSamplerArgs,
    ComplexityMatchedSamplerArgs,
]

_SAMPLER_ARGS_TYPE_MAP: dict[str, type[BaseModel]] = {
    "interval": IntervalSamplerArgs,
    "sliding_window": SlidingWindowSamplerArgs,
    "random": RandomSamplerArgs,
    "peak": PeakSamplerArgs,
    "negative_peak": NegativePeakSamplerArgs,
    "complexity_matched": ComplexityMatchedSamplerArgs,
}

# --- Main Config Models ---


class GenomeConfig(BaseModel):
    """
    Configuration for the genome assembly.

    Attributes:
        name: Name of the genome assembly (e.g., 'hg38').
        fasta_path: Path to the FASTA file.
        exclude_intervals: Dictionary mapping names to BED files of regions to exclude.
        allowed_chroms: List of chromosome names to include.
        chrom_sizes: Dictionary mapping chromosome names to their lengths.
        fold_type: Strategy for creating folds. Currently only 'chrom_partition' is supported.
        fold_args: Arguments for the folding strategy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: FoldArgs

    @field_validator("fasta_path", mode="before")
    @classmethod
    def resolve_fasta_path(cls, v: Any, info: ValidationInfo) -> Path:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_path(v, "Genome file", search_paths=search_paths)

    @field_validator("exclude_intervals", mode="before")
    @classmethod
    def resolve_exclude_intervals(cls, v: Any, info: ValidationInfo) -> dict[str, Path]:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_file_dict(v, "exclude_intervals", search_paths=search_paths)

    @model_validator(mode="after")
    def filter_chrom_sizes(self) -> "GenomeConfig":
        """Ensure chrom_sizes only contains allowed_chroms and all are present."""
        allowed_set = set(self.allowed_chroms)
        filtered = {k: v for k, v in self.chrom_sizes.items() if k in allowed_set}
        if len(filtered) != len(allowed_set):
            missing = allowed_set - set(filtered.keys())
            raise ValueError(f"chrom_sizes missing entries for allowed_chroms: {missing}")
        if filtered != self.chrom_sizes:
            # Reconstruct with filtered chrom_sizes (bypass frozen via model_construct)
            return self.model_copy(update={"chrom_sizes": filtered})
        return self


class SamplerConfig(BaseModel):
    """
    Configuration for data samplers.

    Attributes:
        sampler_type: Type of sampler to use ('interval', 'sliding_window', 'random',
            'complexity_matched', 'peak', 'negative_peak').
        padded_size: Length of the intervals yielded by the sampler (after padding/centering).
        sampler_args: Typed arguments specific to the sampler type.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sampler_type: str
    padded_size: int = Field(gt=0)
    sampler_args: SamplerArgsUnion

    @model_validator(mode="before")
    @classmethod
    def resolve_sampler_args(cls, data: Any, info: ValidationInfo) -> Any:
        """Route sampler_args to the correct typed model based on sampler_type."""
        if isinstance(data, dict):
            sampler_type = data.get("sampler_type")
            args = data.get("sampler_args", {})
            if sampler_type in _SAMPLER_ARGS_TYPE_MAP and isinstance(args, dict):
                data = dict(data)  # don't mutate the original
                ctx = info.context if info and info.context else None
                data["sampler_args"] = _SAMPLER_ARGS_TYPE_MAP[sampler_type].model_validate(
                    args, context=ctx
                )
        return data


class DataConfig(BaseModel):
    """
    Configuration for input/output data handling.

    Attributes:
        inputs: Dictionary mapping input channel names to bigWig file paths.
        targets: Dictionary mapping target channel names to bigWig file paths.
        input_len: Length of the input sequence window.
        output_len: Length of the output signal window.
        max_jitter: Maximum random shift applied to the interval center during training.
        output_bin_size: Size of bins for signal aggregation (1 means raw signal).
        encoding: DNA encoding strategy (e.g., 'ACGT').
        log_transform: Whether to apply log(x+1) transformation to signal.
        reverse_complement: Whether to apply reverse complement augmentation.
        target_scale: Multiplicative scaling factor for targets.
        use_sequence: Whether to use sequence input (default: True).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: dict[str, Path]
    targets: dict[str, Path]
    input_len: int = Field(gt=0)
    output_len: int = Field(gt=0)
    max_jitter: int = Field(ge=0)
    output_bin_size: int = Field(gt=0)
    encoding: str
    log_transform: bool
    reverse_complement: bool
    use_sequence: bool
    target_scale: float = Field(gt=0)

    @field_validator("inputs", mode="before")
    @classmethod
    def resolve_inputs(cls, v: Any, info: ValidationInfo) -> dict[str, Path]:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_file_dict(v, "inputs", search_paths=search_paths)

    @field_validator("targets", mode="before")
    @classmethod
    def resolve_targets(cls, v: Any, info: ValidationInfo) -> dict[str, Path]:
        search_paths = info.context.get("search_paths") if info.context else None
        return _validate_file_dict(v, "targets", search_paths=search_paths)

    @model_validator(mode="after")
    def check_rc_requires_sequence(self) -> "DataConfig":
        if self.reverse_complement and not self.use_sequence:
            raise ValueError(
                "reverse_complement=True requires use_sequence=True. "
                "Reverse complement operates on DNA sequence channels."
            )
        return self


class TrainConfig(BaseModel):
    """
    Configuration for training hyperparameters.

    Attributes:
        batch_size: Batch size for training/validation.
        max_epochs: Maximum number of epochs to train.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for optimizer.
        patience: Patience for early stopping.
        optimizer: Optimizer name (e.g., 'adam', 'adamw', 'sgd').
        filter_bias_and_bn: Whether to exclude bias and batch norm parameters from weight decay.
        adam_eps: Epsilon for Adam/AdamW optimizer numerical stability (default: 1e-8).
        gradient_clip_val: Maximum gradient norm for gradient clipping (default: None = disabled).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    patience: int = Field(ge=0)
    optimizer: str
    scheduler_type: str
    scheduler_args: dict[str, Any]
    filter_bias_and_bn: bool
    reload_dataloaders_every_n_epochs: int = Field(ge=0)
    adam_eps: float = Field(gt=0)
    gradient_clip_val: float | None = Field(default=None, gt=0)


class PretrainedConfig(BaseModel):
    """Configuration for loading pretrained weights into a model or sub-module.

    Attributes:
        weights_path: Path to a .pt state dict file (clean, no "model." prefix).
        source: Sub-module prefix to extract from the source state dict.
            None uses all keys.
        target: Named sub-module to load into. None loads into the whole model.
        freeze: If True, freeze all parameters in the loaded (sub)module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights_path: str
    source: str | None
    target: str | None
    freeze: bool


class ModelConfig(BaseModel):
    """
    Configuration for the model architecture.

    Attributes:
        name: Name of the model.
        model_cls: Fully qualified class name string of the model.
        loss_cls: Fully qualified class name string of the loss.
        loss_args: Loss-specific keyword arguments.
        metrics_cls: Fully qualified class name string of the metric collection.
        metrics_args: Metrics-specific keyword arguments.
        model_args: Model-specific keyword arguments.
        pretrained: List of pretrained weight configs to load after
            model instantiation. Empty list means no pretrained weights.
        count_pseudocount: Additive offset before log-transforming count targets,
            specified in scaled units (i.e. raw coverage × target_scale).
            A value of 0.0 means no pseudocount (for Poisson/NB losses).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    model_cls: str
    loss_cls: str
    loss_args: dict[str, Any]
    metrics_cls: str
    metrics_args: dict[str, Any]
    model_args: dict[str, Any]
    pretrained: list[PretrainedConfig] = Field(default_factory=list)
    count_pseudocount: float = Field(default=0.0, ge=0)

    @field_validator("model_args", mode="after")
    @classmethod
    def validate_model_args(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "input_channels" in v:
            ic = v["input_channels"]
            if not isinstance(ic, (list, tuple)) or not all(isinstance(c, str) for c in ic):
                raise TypeError("model_args['input_channels'] must be a list of strings")
            if len(ic) == 0:
                raise ValueError("model_args['input_channels'] must not be empty")
        if "output_channels" in v:
            oc = v["output_channels"]
            if not isinstance(oc, (list, tuple)) or not all(isinstance(c, str) for c in oc):
                raise TypeError("model_args['output_channels'] must be a list of strings")
            if len(oc) == 0:
                raise ValueError("model_args['output_channels'] must not be empty")
        if "output_type" in v:
            ot = v["output_type"]
            valid_types = {"signal", "decoupled"}
            if ot not in valid_types:
                raise ValueError(f"model_args['output_type'] must be one of {valid_types}")
        return v


class CerberusConfig(BaseModel):
    """
    Combined configuration for Cerberus.

    Note: The ``model_config_`` field uses an alias ``"model_config"`` because
    Pydantic V2 reserves the ``model_config`` name for its own ConfigDict.
    In YAML/dict serialization the key is ``"model_config"``, but in Python
    code the attribute is ``model_config_``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    train_config: TrainConfig
    genome_config: GenomeConfig
    data_config: DataConfig
    sampler_config: SamplerConfig
    model_config_: ModelConfig = Field(alias="model_config")

    @model_validator(mode="after")
    def cross_validate(self) -> "CerberusConfig":
        """Cross-validate data/sampler and data/model compatibility."""
        # Data-sampler compatibility
        input_len = self.data_config.input_len
        max_jitter = self.data_config.max_jitter
        padded_size = self.sampler_config.padded_size
        required_size = input_len + 2 * max_jitter
        if padded_size < required_size:
            raise ValueError(
                f"Sampler padded_size ({padded_size}) is smaller than required size "
                f"({required_size} = input_len {input_len} + 2 * max_jitter {max_jitter}). "
                "Please increase padded_size or decrease input_len/max_jitter."
            )

        # Data-model compatibility
        model_args = self.model_config_.model_args
        if "output_channels" in model_args:
            target_channels = set(self.data_config.targets.keys())
            model_outputs = set(model_args["output_channels"])
            if target_channels != model_outputs:
                raise ValueError(
                    f"Model output channels {model_outputs} do not match data targets {target_channels}"
                )
        if "input_channels" in model_args:
            input_tracks = set(self.data_config.inputs.keys())
            model_inputs = set(model_args["input_channels"])
            if not input_tracks.issubset(model_inputs):
                missing = input_tracks - model_inputs
                raise ValueError(f"Data inputs {missing} are not in model input channels")

        return self


# Resolve forward references for recursive SamplerConfig
ComplexityMatchedSamplerArgs.model_rebuild()
SamplerConfig.model_rebuild()
CerberusConfig.model_rebuild()


# --- Utility Functions ---


def get_log_count_params(model_config: ModelConfig) -> tuple[bool, float]:
    """Determines log-count transform parameters from the model configuration.

    Losses with ``uses_count_pseudocount = True`` (MSE-family, Dalmatian) train
    log_counts in log(count + pseudocount) space, while Poisson/NB losses use
    log(count) directly.

    Args:
        model_config: Model configuration (must contain ``loss_cls`` and
            ``count_pseudocount`` fields).

    Returns:
        Tuple of (log_counts_include_pseudocount, count_pseudocount):
            - log_counts_include_pseudocount: True if the loss uses
              log(count + pseudocount) space.
            - count_pseudocount: The pseudocount value from model_config
              (scaled units), or 0.0 for losses that don't use pseudocount.
    """
    loss_cls = import_class(model_config.loss_cls)
    log_counts_include_pseudocount = loss_cls.uses_count_pseudocount
    if log_counts_include_pseudocount:
        count_pseudocount = model_config.count_pseudocount
    else:
        count_pseudocount = 0.0
    return log_counts_include_pseudocount, count_pseudocount


def parse_hparams_config(
    path: str | Path,
    search_paths: list[Path] | None = None,
) -> CerberusConfig:
    """
    Parses a hparams.yaml file and returns validated configuration objects.

    Args:
        path: Path to the hparams.yaml file.
        search_paths: List of directories to search for referenced files if not found at original paths.

    Returns:
        CerberusConfig: Validated and frozen configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the file content is invalid or missing required sections.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hparams file not found at: {p}")

    # Add hparams directory to search paths by default
    if search_paths is None:
        search_paths = []
    if p.parent not in search_paths:
        search_paths = [*search_paths, p.parent]

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("hparams file must contain a dictionary")

    # Required top-level keys
    required_keys = {
        "train_config",
        "genome_config",
        "data_config",
        "sampler_config",
        "model_config",
    }

    if not all(key in data for key in required_keys):
        missing = required_keys - data.keys()
        raise ValueError(f"hparams missing required sections: {missing}")

    # Extract only known keys (Lightning may add extra hparams)
    config_data = {k: data[k] for k in required_keys}

    # Backwards compatibility: backfill pretrained for old YAML files
    raw_model_config = config_data["model_config"]
    if "pretrained" not in raw_model_config:
        logger.warning(
            "hparams.yaml at %s is missing 'pretrained' field in model_config. "
            "Defaulting to pretrained=[]. Retrain the model to update the config. "
            "This backwards compatibility shim will be removed in a future release.",
            p,
        )
        raw_model_config["pretrained"] = []

    # Backwards compatibility: migrate legacy count_pseudocount from data_config
    raw_data_config = config_data["data_config"]
    if "count_pseudocount" in raw_data_config and "count_pseudocount" not in raw_model_config:
        target_scale = raw_data_config.get("target_scale", 1.0)
        raw_pseudocount = raw_data_config["count_pseudocount"]
        scaled = raw_pseudocount * target_scale
        raw_model_config["count_pseudocount"] = scaled
        logger.warning(
            "hparams.yaml at %s has legacy count_pseudocount=%.4g in data_config. "
            "Migrated to model_config.count_pseudocount=%.4g (raw × target_scale=%.4g). "
            "Retrain the model to update the config.",
            p,
            raw_pseudocount,
            scaled,
            target_scale,
        )
    # Remove count_pseudocount from data_config if present (legacy field)
    raw_data_config.pop("count_pseudocount", None)

    config = CerberusConfig.model_validate(
        config_data, context={"search_paths": search_paths}
    )

    logger.info(f"Successfully parsed hparams from {p}")

    return config
