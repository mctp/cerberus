"""Pydantic V2 configuration schemas for Cerberus.

All configuration types are frozen ``BaseModel`` classes with ``extra="forbid"``.
Fields are accessed via attribute access (``config.key``).  Mutations produce
new instances via ``config.model_copy(update={...})``.

The top-level ``CerberusConfig`` aggregates all sections and performs
cross-validation (padded_size vs input_len, channel matching) at
construction time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenomeConfig(BaseModel):
    """Configuration for the genome assembly.

    Attributes:
        name: Name of the genome assembly (e.g. ``'hg38'``).
        fasta_path: Path to the FASTA file.
        exclude_intervals: Mapping of names to BED files of regions to exclude.
        allowed_chroms: Chromosome names to include.
        chrom_sizes: Mapping of chromosome names to their lengths in bp.
            Must contain exactly the chromosomes listed in ``allowed_chroms``.
        fold_type: Strategy for creating folds (currently ``'chrom_partition'``).
        fold_args: Arguments for the fold strategy.  For ``'chrom_partition'``:
            ``k`` (int), ``test_fold`` (int | None), ``val_fold`` (int | None).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: dict[str, Any]

    @model_validator(mode="after")
    def check_chrom_sizes_complete(self) -> GenomeConfig:
        """Every allowed_chrom must have an entry in chrom_sizes."""
        missing = set(self.allowed_chroms) - set(self.chrom_sizes)
        if missing:
            raise ValueError(
                f"chrom_sizes missing entries for allowed_chroms: {missing}"
            )
        return self


class SamplerConfig(BaseModel):
    """Configuration for data samplers.

    Attributes:
        sampler_type: Sampler kind — ``'interval'``, ``'sliding_window'``,
            ``'random'``, ``'complexity_matched'``, ``'peak'``, or
            ``'negative_peak'``.
        padded_size: Length of intervals yielded by the sampler (after
            padding/centering).
        sampler_args: Sampler-specific keyword arguments (schema depends on
            ``sampler_type``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sampler_type: str
    padded_size: int = Field(gt=0)
    sampler_args: dict[str, Any]


class DataConfig(BaseModel):
    """Configuration for input/output data handling.

    Attributes:
        inputs: Mapping of input channel names to bigWig file paths.
        targets: Mapping of target channel names to bigWig file paths.
        input_len: Length of the input sequence window in bp.
        output_len: Length of the output signal window in bp.
        max_jitter: Maximum random shift applied to the interval center
            during training.
        output_bin_size: Bin size for signal aggregation (1 = raw signal).
        encoding: DNA encoding strategy (e.g. ``'ACGT'``).
        log_transform: Whether to apply ``log(x + 1)`` to signal.
        reverse_complement: Whether to apply reverse-complement augmentation.
        use_sequence: Whether to include one-hot sequence as input.
        target_scale: Multiplicative scaling factor applied to targets.
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

    @model_validator(mode="after")
    def check_rc_requires_sequence(self) -> DataConfig:
        if self.reverse_complement and not self.use_sequence:
            raise ValueError(
                "reverse_complement=True requires use_sequence=True. "
                "Reverse complement operates on DNA sequence channels."
            )
        return self


class TrainConfig(BaseModel):
    """Configuration for training hyperparameters.

    Attributes:
        batch_size: Batch size for training and validation.
        max_epochs: Maximum number of training epochs.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for the optimizer.
        patience: Early-stopping patience (epochs without improvement).
        optimizer: Optimizer name (``'adam'``, ``'adamw'``, ``'sgd'``).
        scheduler_type: Learning-rate scheduler name (``'default'`` to disable).
        scheduler_args: Scheduler-specific keyword arguments.
        filter_bias_and_bn: Exclude bias and batch-norm parameters from
            weight decay.
        reload_dataloaders_every_n_epochs: How often to reload data loaders.
        adam_eps: Epsilon for Adam/AdamW numerical stability.
        gradient_clip_val: Maximum gradient norm for clipping (``None`` to
            disable).
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
    """Configuration for loading pretrained weights into a model sub-module.

    Parameter freezing is handled separately via :class:`FreezeSpec` on
    ``ModelConfig.freeze`` — this class only describes *loading*.

    Attributes:
        weights_path: Path to a ``.pt`` state-dict file.
        source: Sub-module prefix to extract from the source state dict
            (``None`` uses all keys).
        target: Named sub-module to load into (``None`` loads into the
            whole model).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights_path: str
    source: str | None
    target: str | None


class FreezeSpec(BaseModel):
    """Declarative freeze rule applied after model instantiation.

    Patterns are exact paths into the model's named hierarchy — not
    globs.  Exactly one of the following must hold:

    - ``pattern`` equals the name of a module in ``named_modules()``
      (e.g. ``"bias_model"``, ``"res_layers.0"``) — freezes every
      parameter in that subtree and, if ``eval_mode`` is ``True``,
      calls ``.eval()`` on the module root so that Dropout / BatchNorm
      descendants stop firing / drifting.
    - ``pattern`` equals the name of a parameter in
      ``named_parameters()`` (e.g. ``"iconv.weight"``) — freezes that
      single parameter.  ``eval_mode`` is ignored for parameter-only
      matches.

    A zero-match pattern raises at ``apply_freeze`` time so typos
    cannot silently become no-ops.

    Attributes:
        pattern: Exact module or parameter path to freeze.
        eval_mode: If ``True`` (default), call ``.eval()`` on the
            matched module root. Ignored when ``pattern`` names a
            parameter rather than a module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: str
    eval_mode: bool = True


class ModelConfig(BaseModel):
    """Configuration for the model architecture.

    Attributes:
        name: Human-readable model name.
        model_cls: Fully qualified class name of the model.
        loss_cls: Fully qualified class name of the loss.
        loss_args: Loss constructor keyword arguments.
        metrics_cls: Fully qualified class name of the metric collection.
        metrics_args: Metrics constructor keyword arguments.
        model_args: Model constructor keyword arguments.
        pretrained: List of pretrained-weight configs to apply after
            instantiation.  Empty list means train from scratch.
        freeze: List of freeze rules applied after the model is built
            and pretrained weights (if any) are loaded.  Empty list
            means all parameters remain trainable.  See
            :class:`FreezeSpec` for pattern semantics.
        count_pseudocount: Additive offset before log-transforming count
            targets, in scaled units (raw coverage × ``target_scale``).
            Set to ``0.0`` for losses that do not use a pseudocount
            (Poisson / NB).
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
    freeze: list[FreezeSpec] = Field(default_factory=list)
    count_pseudocount: float = Field(default=0.0, ge=0)


class CerberusConfig(BaseModel):
    """Combined configuration for Cerberus.

    Aggregates all sub-configs and performs cross-validation at construction
    time (padded_size vs input_len + jitter, output channel matching).

    Note:
        The ``model_config_`` attribute uses the alias ``"model_config"``
        because Pydantic V2 reserves the bare name.  In YAML / dict
        serialization the key is ``"model_config"``; in Python code use
        ``cerberus_config.model_config_``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    train_config: TrainConfig
    genome_config: GenomeConfig
    data_config: DataConfig
    sampler_config: SamplerConfig
    model_config_: ModelConfig = Field(alias="model_config")

    @model_validator(mode="after")
    def cross_validate(self) -> CerberusConfig:
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
                raise ValueError(
                    f"Data inputs {missing} are not in model input channels"
                )

        return self
