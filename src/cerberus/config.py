from __future__ import annotations

from typing import Any
from pathlib import Path
import yaml
import logging
import importlib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


def import_class(name: str) -> Any:
    """Dynamically imports a class from a module string (e.g., 'package.module.Class')."""
    try:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{name}': {e}")


# --- Configuration Schemas ---


class GenomeConfig(BaseModel):
    """Configuration for the genome assembly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: dict[str, Any]

    @model_validator(mode="after")
    def filter_chrom_sizes(self) -> "GenomeConfig":
        """Ensure chrom_sizes only contains allowed_chroms and all are present."""
        allowed_set = set(self.allowed_chroms)
        filtered = {k: v for k, v in self.chrom_sizes.items() if k in allowed_set}
        if len(filtered) != len(allowed_set):
            missing = allowed_set - set(filtered.keys())
            raise ValueError(f"chrom_sizes missing entries for allowed_chroms: {missing}")
        if filtered != self.chrom_sizes:
            return self.model_copy(update={"chrom_sizes": filtered})
        return self


class SamplerConfig(BaseModel):
    """Configuration for data samplers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sampler_type: str
    padded_size: int = Field(gt=0)
    sampler_args: dict[str, Any]


class DataConfig(BaseModel):
    """Configuration for input/output data handling."""

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

    @model_validator(mode="after")
    def check_rc_requires_sequence(self) -> "DataConfig":
        if self.reverse_complement and not self.use_sequence:
            raise ValueError(
                "reverse_complement=True requires use_sequence=True. "
                "Reverse complement operates on DNA sequence channels."
            )
        return self


class TrainConfig(BaseModel):
    """Configuration for training hyperparameters."""

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
    """Configuration for loading pretrained weights into a model or sub-module."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights_path: str
    source: str | None
    target: str | None
    freeze: bool


class ModelConfig(BaseModel):
    """Configuration for the model architecture."""

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
            valid_types = {"signal", "decoupled"}
            if v["output_type"] not in valid_types:
                raise ValueError(f"model_args['output_type'] must be one of {valid_types}")
        return v


class CerberusConfig(BaseModel):
    """Combined configuration for Cerberus.

    Note: ``model_config_`` uses alias ``"model_config"`` because Pydantic V2
    reserves the name. YAML key is ``"model_config"``; Python attribute is
    ``model_config_``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    train_config: TrainConfig
    genome_config: GenomeConfig
    data_config: DataConfig
    sampler_config: SamplerConfig
    model_config_: ModelConfig = Field(alias="model_config")

    @model_validator(mode="after")
    def cross_validate(self) -> "CerberusConfig":
        """Cross-validate data/sampler and data/model compatibility."""
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

        model_args = self.model_config_.model_args
        if "output_channels" in model_args:
            target_channels = set(self.data_config.targets.keys())
            model_outputs = set(model_args["output_channels"])
            if target_channels != model_outputs:
                raise ValueError(
                    f"Model output channels {model_outputs} do not match "
                    f"data targets {target_channels}"
                )
        if "input_channels" in model_args:
            input_tracks = set(self.data_config.inputs.keys())
            model_inputs = set(model_args["input_channels"])
            if not input_tracks.issubset(model_inputs):
                missing = input_tracks - model_inputs
                raise ValueError(f"Data inputs {missing} are not in model input channels")

        return self


# --- Utility Functions ---


def get_log_count_params(model_config: ModelConfig) -> tuple[bool, float]:
    """Determines log-count transform parameters from the model configuration.

    Returns (log_counts_include_pseudocount, count_pseudocount).
    """
    loss_cls = import_class(model_config.loss_cls)
    if loss_cls.uses_count_pseudocount:
        return True, model_config.count_pseudocount
    return False, 0.0


def parse_hparams_config(path: str | Path) -> CerberusConfig:
    """Parses a hparams.yaml file and returns a validated CerberusConfig.

    Args:
        path: Path to the hparams.yaml file.

    Returns:
        Validated and frozen CerberusConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the content is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hparams file not found at: {p}")

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    return CerberusConfig.model_validate(data)
