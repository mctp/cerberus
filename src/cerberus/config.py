from typing import TypedDict, Any
from pathlib import Path
import yaml
import logging
import importlib

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

# --- Configuration Schemas ---


class GenomeConfig(TypedDict):
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
                   For 'chrom_partition', required keys: 'k' (int),
                   'test_fold' (int), 'val_fold' (int).
                   test_fold and val_fold can be omitted if passed directly
                   to CerberusDataModule or train_single.
    """

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: dict[str, Any]


class SamplerConfig(TypedDict):
    """
    Configuration for data samplers.

    Attributes:
        sampler_type: Type of sampler to use ('interval', 'sliding_window', 'random',
            'complexity_matched', 'peak', 'negative_peak').
        padded_size: Length of the intervals yielded by the sampler (after padding/centering).
        sampler_args: Dictionary of arguments specific to the sampler type.

    Sampler Args Requirements by Type:
    - 'interval':
        - intervals_path: Path to BED/narrowPeak file containing regions of interest.
    - 'sliding_window':
        - stride: Step size for generating sliding windows across the genome.
    - 'random':
        - num_intervals: Number of random intervals to generate.
    - 'complexity_matched':
        - target_sampler: Configuration for the target sampler.
        - candidate_sampler: Configuration for the candidate sampler.
        - bins: Number of bins.
        - candidate_ratio: Ratio of candidates to targets.
        - metrics: List of metrics (e.g. ['gc']).
    - 'peak':
        - intervals_path: Path to peaks file.
        - background_ratio: Ratio of background intervals to peaks (default: 1.0).
    - 'negative_peak':
        - intervals_path: Path to peaks file (used for exclusion and complexity reference).
        - background_ratio: Number of background intervals per peak (controls epoch size).
    """

    sampler_type: str
    padded_size: int
    sampler_args: dict[str, Any]


class DataConfig(TypedDict):
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
        count_pseudocount: Additive offset before log-transforming count targets, specified
            in raw coverage units (i.e. approximately read_length). propagate_pseudocount
            (called from instantiate()) multiplies this by target_scale before injecting
            into loss_args and metrics_args so that loss and metrics always receive the
            value in their native (scaled) units.
            A value of 1.0 with target_scale=1.0 reproduces the original log1p behaviour.
            Set to 0.0 for losses that do not use a pseudocount (e.g. Poisson/NB).
    """

    inputs: dict[str, Path]
    targets: dict[str, Path]
    input_len: int
    output_len: int
    max_jitter: int
    output_bin_size: int
    encoding: str
    log_transform: bool
    reverse_complement: bool
    use_sequence: bool
    target_scale: float
    count_pseudocount: float


class TrainConfig(TypedDict):
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
            chrombpnet-pytorch uses 1e-7 for BPNet-style models.
        gradient_clip_val: Maximum gradient norm for gradient clipping (default: None = disabled).
            Passed to pl.Trainer as gradient_clip_val. A value of 1.0 is a reasonable
            safeguard for unnormalized networks like BPNet.
    """

    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    optimizer: str
    scheduler_type: str
    scheduler_args: dict[str, Any]
    filter_bias_and_bn: bool
    reload_dataloaders_every_n_epochs: int
    adam_eps: float
    gradient_clip_val: float | None


class PretrainedConfig(TypedDict):
    """Configuration for loading pretrained weights into a model or sub-module.

    Attributes:
        weights_path: Path to a .pt state dict file (clean, no "model." prefix).
        source: Sub-module prefix to extract from the source state dict.
            None uses all keys. E.g. ``"bias_model"`` extracts and strips keys
            starting with ``"bias_model."``, enabling loading a sub-module
            from a full-model checkpoint.
        target: Named sub-module to load into. None loads into the whole model.
            E.g. ``"bias_model"`` loads into ``model.bias_model``.
        freeze: If True, freeze all parameters in the loaded (sub)module
            by setting requires_grad=False.
    """

    weights_path: str
    source: str | None
    target: str | None
    freeze: bool


class ModelConfig(TypedDict):
    """
    Configuration for the model architecture.

    Attributes:
        name: Name of the model.
        model_cls: Fully qualified class name string of the model.
        loss_cls: Fully qualified class name string of the loss.
        metrics_cls: Fully qualified class name string of the metric collection.
        model_args: Model-specific keyword arguments.
        pretrained: List of pretrained weight configs to load after
            model instantiation. Empty list means no pretrained weights.
    """

    name: str
    model_cls: str
    loss_cls: str
    loss_args: dict[str, Any]
    metrics_cls: str
    metrics_args: dict[str, Any]
    model_args: dict[str, Any]
    pretrained: list[PretrainedConfig]


class CerberusConfig(TypedDict):
    """
    Combined configuration for Cerberus.
    """
    train_config: TrainConfig
    genome_config: GenomeConfig
    data_config: DataConfig
    sampler_config: SamplerConfig
    model_config: ModelConfig


# --- Validation Logic ---


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
                for i in range(len(parts)-1, 0, -1):
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
            raise FileNotFoundError(f"{description} not found at: {p} (and could not be resolved in search paths)")
    
    return p


def _validate_file_dict(
    data: dict, 
    description: str,
    search_paths: list[Path] | None = None
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


def validate_genome_config(
    config: GenomeConfig,
    search_paths: list[Path] | None = None
) -> GenomeConfig:
    """
    Validates the genome configuration and returns a GenomeConfig object.
    
    Checks for required keys, correct types, and consistency between allowed_chroms and chrom_sizes.
    
    Args:
        config: Dictionary containing genome configuration.
        
    Returns:
        GenomeConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are inconsistent.
        FileNotFoundError: If specified files do not exist.
    """
    if not isinstance(config, dict):
        raise TypeError("Genome config must be a dictionary")

    # Check for required fields
    required_keys = {
        "name",
        "fasta_path",
        "allowed_chroms",
        "chrom_sizes",
        "exclude_intervals",
        "fold_type",
        "fold_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Genome config missing required keys: {missing}")
    
    if not isinstance(config["fold_type"], str):
        raise TypeError("fold_type must be a string")
    
    if not isinstance(config["fold_args"], dict):
        raise TypeError("fold_args must be a dictionary")

    # Validate common fold_args keys if present
    for key in ["k", "val_fold", "test_fold"]:
        if key in config["fold_args"]:
            if not isinstance(config["fold_args"][key], int):
                raise TypeError(f"fold_args['{key}'] must be an integer")
            if config["fold_args"][key] < 0:
                raise ValueError(f"fold_args['{key}'] must be non-negative")

    path_val = config["fasta_path"]
    if not isinstance(path_val, (str, Path)):
        raise TypeError("fasta_path must be a string or Path")

    p = _validate_path(path_val, "Genome file", search_paths=search_paths)

    exclude_intervals = _validate_file_dict(
        config["exclude_intervals"], "exclude_intervals", search_paths=search_paths
    )

    if not isinstance(config["allowed_chroms"], list):
        raise TypeError("allowed_chroms must be a list of strings")
    if not all(isinstance(c, str) for c in config["allowed_chroms"]):
        raise TypeError("allowed_chroms must contain only strings")

    if not isinstance(config["chrom_sizes"], dict):
        raise TypeError("chrom_sizes must be a dictionary")
    # Validate chrom_sizes contents
    for k, v in config["chrom_sizes"].items():
        if not isinstance(k, str):
            raise TypeError(f"chrom_sizes keys must be strings, got {type(k)}")
        if not isinstance(v, int):
            raise TypeError(f"chrom_sizes values must be integers, got {type(v)}")

    # Ensure chrom_sizes only contains allowed_chroms
    allowed_set = set(config["allowed_chroms"])
    filtered_sizes = {
        k: v for k, v in config["chrom_sizes"].items() if k in allowed_set
    }

    # Check if all allowed chroms have sizes
    if len(filtered_sizes) != len(allowed_set):
        missing = allowed_set - set(filtered_sizes.keys())
        raise ValueError(f"chrom_sizes missing entries for allowed_chroms: {missing}")

    # Return valid config (cast/ensure types)
    return {
        "name": str(config["name"]),
        "fasta_path": p,
        "exclude_intervals": exclude_intervals,
        "allowed_chroms": config["allowed_chroms"],
        "chrom_sizes": filtered_sizes,
        "fold_type": config["fold_type"],
        "fold_args": config["fold_args"],
    }


def validate_data_config(
    config: DataConfig,
    search_paths: list[Path] | None = None
) -> DataConfig:
    """
    Validates the data configuration dictionary.
    
    Checks for required keys and correct types for data parameters.
    
    Args:
        config: Dictionary containing data configuration.
        
    Returns:
        DataConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid (e.g., negative lengths).
        FileNotFoundError: If specified files do not exist.
    """
    if not isinstance(config, dict):
        raise TypeError("Data config must be a dictionary")

    required_keys = {
        "inputs",
        "targets",
        "input_len",
        "output_len",
        "output_bin_size",
        "encoding",
        "max_jitter",
        "log_transform",
        "reverse_complement",
        "target_scale",
        "count_pseudocount",
        "use_sequence",
    }

    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Data config missing required keys: {missing}")

    inputs = _validate_file_dict(config["inputs"], "inputs", search_paths=search_paths)
    targets = _validate_file_dict(config["targets"], "targets", search_paths=search_paths)

    # Type and Value checks
    if not isinstance(config["input_len"], int) or config["input_len"] <= 0:
        raise ValueError("input_len must be a positive integer")

    if not isinstance(config["output_len"], int) or config["output_len"] <= 0:
        raise ValueError("output_len must be a positive integer")

    if not isinstance(config["max_jitter"], int) or config["max_jitter"] < 0:
        raise ValueError("max_jitter must be a non-negative integer")

    if not isinstance(config["output_bin_size"], int) or config["output_bin_size"] <= 0:
        raise ValueError("output_bin_size must be a positive integer")

    if not isinstance(config["encoding"], str):
        raise TypeError("encoding must be a string")

    if not isinstance(config["log_transform"], bool):
        raise TypeError("log_transform must be a boolean")

    if not isinstance(config["reverse_complement"], bool):
        raise TypeError("reverse_complement must be a boolean")

    if not isinstance(config["target_scale"], float) or config["target_scale"] <= 0:
        raise ValueError("target_scale must be a positive number")

    if not isinstance(config["count_pseudocount"], (int, float)) or config["count_pseudocount"] < 0:
        raise ValueError("count_pseudocount must be a non-negative number")

    if not isinstance(config["use_sequence"], bool):
        raise TypeError("use_sequence must be a boolean")

    if config["reverse_complement"] and not config["use_sequence"]:
        raise ValueError(
            "reverse_complement=True requires use_sequence=True. "
            "Reverse complement operates on DNA sequence channels."
        )

    return {
        "inputs": inputs,
        "targets": targets,
        "input_len": config["input_len"],
        "output_len": config["output_len"],
        "max_jitter": config["max_jitter"],
        "output_bin_size": config["output_bin_size"],
        "encoding": config["encoding"],
        "log_transform": config["log_transform"],
        "reverse_complement": config["reverse_complement"],
        "target_scale": config["target_scale"],
        "count_pseudocount": float(config["count_pseudocount"]),
        "use_sequence": config["use_sequence"],
    }


def validate_sampler_config(
    config: SamplerConfig,
    search_paths: list[Path] | None = None
) -> SamplerConfig:
    """
    Validates the sampler configuration dictionary.
    
    Checks for required keys and correct types. Performs specific validation based on `sampler_type`.
    
    Args:
        config: Dictionary containing sampler configuration.
        
    Returns:
        SamplerConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Sampler config must be a dictionary")

    required_keys = {
        "sampler_type",
        "padded_size",
        "sampler_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Sampler config missing required keys: {missing}")

    if not isinstance(config["sampler_type"], str):
        raise TypeError("sampler_type must be a string")

    if not isinstance(config["padded_size"], int) or config["padded_size"] <= 0:
        raise ValueError("padded_size must be a positive integer")

    if not isinstance(config["sampler_args"], dict):
        raise TypeError("sampler_args must be a dictionary")

    # Specialized validation based on sampler_type
    if config["sampler_type"] == "interval":
        required_args = {"intervals_path"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(f"IntervalSampler args missing required keys: {missing}")

    elif config["sampler_type"] == "sliding_window":
        required_args = {"stride"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(
                f"SlidingWindowSampler args missing required keys: {missing}"
            )

    elif config["sampler_type"] == "random":
        required_args = {"num_intervals"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(f"RandomSampler args missing required keys: {missing}")

    elif config["sampler_type"] == "peak":
        required_args = {"intervals_path"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(f"PeakSampler args missing required keys: {missing}")

    elif config["sampler_type"] == "negative_peak":
        required_args = {"intervals_path"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(f"NegativePeakSampler args missing required keys: {missing}")

    return {
        "sampler_type": config["sampler_type"],
        "padded_size": config["padded_size"],
        "sampler_args": config["sampler_args"],
    }


def validate_data_and_sampler_compatibility(
    data_config: DataConfig, sampler_config: SamplerConfig
) -> None:
    """
    Validates compatibility between data and sampler configurations.

    Checks if the sampler's padded_size is sufficient to cover the input window
    plus the maximum jitter range.

    Args:
        data_config: Validated DataConfig.
        sampler_config: Validated SamplerConfig.

    Raises:
        ValueError: If padded_size is too small for the requested input_len and max_jitter.
    """
    input_len = data_config["input_len"]
    max_jitter = data_config["max_jitter"]
    padded_size = sampler_config["padded_size"]
    
    required_size = input_len + 2 * max_jitter
    
    if padded_size < required_size:
        raise ValueError(
            f"Sampler padded_size ({padded_size}) is smaller than required size "
            f"({required_size} = input_len {input_len} + 2 * max_jitter {max_jitter}). "
            "Please increase padded_size or decrease input_len/max_jitter."
        )


def validate_train_config(config: TrainConfig) -> TrainConfig:
    """
    Validates the training configuration dictionary.

    Args:
        config: Dictionary containing training configuration.

    Returns:
        TrainConfig: Validated and typed configuration object.

    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Train config must be a dictionary")

    required_keys = {
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "patience",
        "optimizer",
        "filter_bias_and_bn",
        "adam_eps",
        "gradient_clip_val",
        "scheduler_type",
        "scheduler_args",
        "reload_dataloaders_every_n_epochs",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Train config missing required keys: {missing}")

    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(config["max_epochs"], int) or config["max_epochs"] <= 0:
        raise ValueError("max_epochs must be a positive integer")

    if not isinstance(config["learning_rate"], float) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be a positive float")
        
    if not isinstance(config["weight_decay"], float) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be a non-negative float")

    if not isinstance(config["patience"], int) or config["patience"] < 0:
        raise ValueError("patience must be a non-negative integer")

    if not isinstance(config["optimizer"], str):
        raise TypeError("optimizer must be a string")

    if not isinstance(config["filter_bias_and_bn"], bool):
        raise TypeError("filter_bias_and_bn must be a boolean")

    if not isinstance(config["scheduler_type"], str):
        raise TypeError("scheduler_type must be a string")

    if not isinstance(config["scheduler_args"], dict):
        raise TypeError("scheduler_args must be a dictionary")

    if not isinstance(config["reload_dataloaders_every_n_epochs"], int) or config["reload_dataloaders_every_n_epochs"] < 0:
        raise ValueError("reload_dataloaders_every_n_epochs must be a non-negative integer")

    if not isinstance(config["adam_eps"], float) or config["adam_eps"] <= 0:
        raise ValueError("adam_eps must be a positive float")

    gradient_clip_val = config["gradient_clip_val"]
    if gradient_clip_val is not None and (not isinstance(gradient_clip_val, float) or gradient_clip_val <= 0):
        raise ValueError("gradient_clip_val must be a positive float or None")

    return {
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "patience": config["patience"],
        "optimizer": config["optimizer"],
        "scheduler_type": config["scheduler_type"],
        "scheduler_args": config["scheduler_args"],
        "filter_bias_and_bn": config["filter_bias_and_bn"],
        "reload_dataloaders_every_n_epochs": config["reload_dataloaders_every_n_epochs"],
        "adam_eps": config["adam_eps"],
        "gradient_clip_val": config["gradient_clip_val"],
    }


def validate_model_config(config: ModelConfig) -> ModelConfig:
    """
    Validates the model configuration dictionary.

    Args:
        config: Dictionary containing model configuration.

    Returns:
        ModelConfig: Validated and typed configuration object.

    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Model config must be a dictionary")

    required_keys = {
        "name",
        "model_cls",
        "loss_cls",
        "loss_args",
        "metrics_cls",
        "metrics_args",
        "model_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Model config missing required keys: {missing}")

    if not isinstance(config["name"], str):
        raise TypeError("name must be a string")

    # Strict validation for class strings
    if not isinstance(config["model_cls"], str):
        raise TypeError("model_cls must be a string (fully qualified class name)")
    
    if not isinstance(config["loss_cls"], str):
        raise TypeError("loss_cls must be a string (fully qualified class name)")
        
    if not isinstance(config["metrics_cls"], str):
        raise TypeError("metrics_cls must be a string (fully qualified class name)")

    # Validate optional arguments in model_args if present
    model_args = config["model_args"]
    if not isinstance(model_args, dict):
        raise TypeError("model_args must be a dictionary")

    # Validate input_channels if present
    if "input_channels" in model_args:
        if not isinstance(model_args["input_channels"], (list, tuple)):
            raise TypeError("model_args['input_channels'] must be a list or tuple of strings")
        if not all(isinstance(c, str) for c in model_args["input_channels"]):
            raise TypeError("model_args['input_channels'] must contain only strings")
        if len(model_args["input_channels"]) == 0:
            raise ValueError("model_args['input_channels'] must not be empty")

    # Validate output_channels if present
    if "output_channels" in model_args:
        if not isinstance(model_args["output_channels"], (list, tuple)):
            raise TypeError("model_args['output_channels'] must be a list or tuple of strings")
        if not all(isinstance(c, str) for c in model_args["output_channels"]):
            raise TypeError("model_args['output_channels'] must contain only strings")
        if len(model_args["output_channels"]) == 0:
            raise ValueError("model_args['output_channels'] must not be empty")

    # Validate output_type if present
    if "output_type" in model_args:
        if not isinstance(model_args["output_type"], str):
            raise TypeError("model_args['output_type'] must be a string")
        valid_types = {"signal", "decoupled"}
        if model_args["output_type"] not in valid_types:
            raise ValueError(f"model_args['output_type'] must be one of {valid_types}")

    return {
        "name": config["name"],
        "model_cls": config["model_cls"],
        "loss_cls": config["loss_cls"],
        "loss_args": config["loss_args"],
        "metrics_cls": config["metrics_cls"],
        "metrics_args": config["metrics_args"],
        "model_args": config["model_args"],
        "pretrained": config["pretrained"],
    }


def validate_data_and_model_compatibility(
    data_config: DataConfig, model_config: ModelConfig
) -> None:
    """
    Validates compatibility between data and model configurations.

    Checks if the model's input/output channels match the data configuration.

    Args:
        data_config: Validated DataConfig.
        model_config: Validated ModelConfig.

    Raises:
        ValueError: If channel names do not match between model and data.
    """
    # Check output channels vs targets
    target_channels = set(data_config["targets"].keys())
    
    # Look in model_args for channels
    model_args = model_config["model_args"]
    
    if "output_channels" in model_args:
        model_outputs = set(model_args["output_channels"])
        if target_channels != model_outputs:
            raise ValueError(
                f"Model output channels {model_outputs} do not match data targets {target_channels}"
            )

    # Ensure all data input tracks are present in model input channels
    if "input_channels" in model_args:
        input_tracks = set(data_config["inputs"].keys())
        # Assuming model_config inputs includes both sequence and tracks?
        # Or just tracks? User said "('A','C','G,'T',"name of signal trak")". So it includes sequence.
        
        # So model inputs should be superset of data inputs keys?
        model_inputs = set(model_args["input_channels"])
        if not input_tracks.issubset(model_inputs):
            missing = input_tracks - model_inputs
            raise ValueError(f"Data inputs {missing} are not in model input channels")


def get_log_count_params(model_config: ModelConfig) -> tuple[bool, float]:
    """Determines log-count transform parameters from the model configuration.

    Losses with ``uses_count_pseudocount = True`` (MSE-family, Dalmatian) train
    log_counts in log(count + pseudocount) space, while Poisson/NB losses use
    log(count) directly.  This function inspects the loss class attribute and
    returns the two parameters needed by ``compute_total_log_counts`` and
    observed-count transforms.

    The returned pseudocount is in **scaled** units (i.e. already multiplied by
    target_scale via ``propagate_pseudocount``), matching the space that model
    predictions operate in.

    Args:
        model_config: Model configuration dict (must contain ``loss_cls`` and
            ``loss_args`` keys).

    Returns:
        Tuple of (log_counts_include_pseudocount, count_pseudocount):
            - log_counts_include_pseudocount: True if the loss uses
              log(count + pseudocount) space.
            - count_pseudocount: The pseudocount value from loss_args
              (scaled units), or 0.0 for losses that don't use pseudocount.
    """
    loss_cls = import_class(model_config["loss_cls"])
    log_counts_include_pseudocount = loss_cls.uses_count_pseudocount
    if log_counts_include_pseudocount:
        count_pseudocount = model_config["loss_args"]["count_pseudocount"]
    else:
        count_pseudocount = 0.0
    return log_counts_include_pseudocount, count_pseudocount


def propagate_pseudocount(data_config: DataConfig, model_config: ModelConfig) -> ModelConfig:
    """
    Propagate the scaled count_pseudocount from data_config into model_config's
    loss_args and metrics_args.

    The user specifies count_pseudocount in raw coverage units (e.g. read length);
    scaling by target_scale converts it to the units that the loss and metrics
    actually operate on. Uses setdefault so an explicitly provided value in
    loss_args/metrics_args is never overwritten.

    Returns a new ModelConfig; the input is not modified so callers can safely
    reuse the same model_config dict across folds.

    Args:
        data_config: Validated data configuration containing count_pseudocount
            and target_scale.
        model_config: Model configuration containing loss_args and metrics_args.

    Returns:
        A new ModelConfig with count_pseudocount set in loss_args and metrics_args.
    """
    loss_cls = import_class(model_config["loss_cls"])
    raw_pseudocount = data_config["count_pseudocount"]

    if not loss_cls.uses_count_pseudocount and raw_pseudocount > 0:
        logger.warning(
            "count_pseudocount=%.4g has no effect with %s (uses log(count) directly); "
            "consider setting count_pseudocount to 0.0",
            raw_pseudocount,
            loss_cls.__name__,
        )

    scaled_pseudocount = raw_pseudocount * data_config["target_scale"]
    loss_args = {**model_config["loss_args"]}
    loss_args.setdefault("count_pseudocount", scaled_pseudocount)
    metrics_args = {**model_config["metrics_args"]}
    metrics_args.setdefault("count_pseudocount", scaled_pseudocount)
    metrics_args.setdefault("log_counts_include_pseudocount", loss_cls.uses_count_pseudocount)
    return {**model_config, "loss_args": loss_args, "metrics_args": metrics_args}


def parse_hparams_config(
    path: str | Path, 
    search_paths: list[Path] | None = None
) -> CerberusConfig:
    """
    Parses a hparams.yaml file and returns validated configuration objects.
    
    Args:
        path: Path to the hparams.yaml file.
        search_paths: List of directories to search for referenced files if not found at original paths.
        
    Returns:
        CerberusConfig: Dictionary containing all validated configurations.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is invalid or missing required sections.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hparams file not found at: {p}")
        
    # Add hparams directory to search paths by default
    if search_paths is None:
        search_paths = []
    if p.parent not in search_paths:
        search_paths.append(p.parent)
        
    with open(p, 'r') as f:
        data = yaml.safe_load(f)
        
    if not isinstance(data, dict):
        raise ValueError("hparams file must contain a dictionary")
        
    # Required top-level keys
    required_keys = {
        "train_config",
        "genome_config", 
        "data_config",
        "sampler_config",
        "model_config"
    }
    
    # Check if all required keys are present
    # We allow extra keys (like other PL hparams), but ensure we have ours
    if not all(key in data for key in required_keys):
        missing = required_keys - data.keys()
        raise ValueError(f"hparams missing required sections: {missing}")
        
    # Validate each section
    train_conf = validate_train_config(data["train_config"])
    genome_conf = validate_genome_config(data["genome_config"], search_paths=search_paths)
    data_conf = validate_data_config(data["data_config"], search_paths=search_paths)
    sampler_conf = validate_sampler_config(data["sampler_config"], search_paths=search_paths)
    # Backfill pretrained for YAML files saved before the field existed.
    # This backwards compatibility shim will be removed in a future release —
    # retrain models to generate hparams.yaml files with the pretrained field.
    raw_model_config = data["model_config"]
    if "pretrained" not in raw_model_config:
        logger.warning(
            "hparams.yaml at %s is missing 'pretrained' field in model_config. "
            "Defaulting to pretrained=[]. Retrain the model to update the config. "
            "This backwards compatibility shim will be removed in a future release.",
            p,
        )
        raw_model_config["pretrained"] = []
    model_conf = validate_model_config(raw_model_config)
    
    # Cross-validation
    validate_data_and_sampler_compatibility(data_conf, sampler_conf)
    validate_data_and_model_compatibility(data_conf, model_conf)

    config: CerberusConfig = {
        "train_config": train_conf,
        "genome_config": genome_conf,
        "data_config": data_conf,
        "sampler_config": sampler_conf,
        "model_config": model_conf,
    }
    
    logger.info(f"Successfully parsed hparams from {p}")
        
    return config

def _sanitize_config(config: Any) -> Any:
    """Recursively convert Path objects to strings for clean serialization."""
    if isinstance(config, dict):
        return {k: _sanitize_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_sanitize_config(v) for v in config]
    elif isinstance(config, Path):
        return str(config)
    return config
